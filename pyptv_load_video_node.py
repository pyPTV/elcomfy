import os
import re
import io
import time
import subprocess
import numpy as np
import torch
import folder_paths
from comfy.utils import ProgressBar
from .pyptv_utils import ffmpeg_path, ENCODE_ARGS, strip_path, hash_path

VIDEO_EXTENSIONS = {"mp4", "mkv", "webm", "mov", "gif"}
PYPTV_CODECS_DECODE = ["auto", "h264", "hevc", "av1", "vp9"]

# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _probe_video(video_path, decode_codec="auto"):
    args = [ffmpeg_path]
    if decode_codec == "vp9":
        args += ["-c:v", "libvpx-vp9"]
    elif decode_codec != "auto":
        args += ["-c:v", decode_codec]
    args += ["-i", video_path, "-c", "copy", "-frames:v", "1", "-f", "null", "-"]

    try:
        res = subprocess.run(args, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg probe failed:\n" + e.stderr.decode(*ENCODE_ARGS))

    lines = res.stderr.decode(*ENCODE_ARGS)
    width = height = fps = duration = None
    alpha = False

    for line in lines.split("\n"):
        m = re.search(r"^ *Stream .* Video.*, ([1-9]|\d{2,})x(\d+)", line)
        if m:
            width, height = int(m.group(1)), int(m.group(2))
            fps_m = re.search(r", ([\d\.]+) fps", line)
            fps = float(fps_m.group(1)) if fps_m else 1.0
            alpha = bool(re.search(r"(yuva|rgba|bgra|gbra)", line))
            break

    if width is None:
        raise RuntimeError("Failed to parse video info.\nFFMPEG output:\n" + lines)

    dur_m = re.search(r"Duration: (\d+):(\d+):([\d\.]+),", lines)
    duration = (int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60
                + float(dur_m.group(3))) if dur_m else 0.0

    return width, height, fps, duration, alpha


# ---------------------------------------------------------------------------
# Frame generator
# ---------------------------------------------------------------------------

def _ffmpeg_frame_generator(video_path, width, height, alpha, decode_codec):
    args = [ffmpeg_path, "-v", "error", "-an"]

    if decode_codec == "vp9":
        args += ["-c:v", "libvpx-vp9"]
    elif decode_codec != "auto":
        args += ["-c:v", decode_codec]

    args += ["-i", video_path, "-pix_fmt", "rgba64le", "-f", "rawvideo", "-"]

    bpi = width * height * 8  # rgba64le: 4ch * 2 bytes
    pbar = ProgressBar(1)
    frames_yielded = 0

    try:
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            buf = bytearray(bpi)
            offset = 0
            prev = None

            while True:
                chunk = proc.stdout.read(bpi - offset)
                if chunk is None:
                    time.sleep(0.05)
                    continue
                if len(chunk) == 0:
                    break

                buf[offset:offset + len(chunk)] = chunk
                offset += len(chunk)

                if offset == bpi:
                    frame = (
                        np.frombuffer(buf, dtype=np.dtype(np.uint16).newbyteorder("<"))
                        .reshape(height, width, 4)
                        .astype(np.float32) / 65535.0
                    )
                    if not alpha:
                        frame = frame[:, :, :3]

                    if prev is not None:
                        sig = yield prev
                        frames_yielded += 1
                        pbar.update_absolute(frames_yielded, frames_yielded + 1)
                        if sig is not None:
                            return
                    prev = frame
                    offset = 0

    except BrokenPipeError:
        raise RuntimeError("ffmpeg process broke pipe unexpectedly.")

    if prev is not None:
        yield prev


# ---------------------------------------------------------------------------
# Audio extractor
# ---------------------------------------------------------------------------

def _extract_audio(video_path):
    """Return a callable that extracts audio on demand — same as VHS lazy_get_audio."""
    def _get():
        args = [
            ffmpeg_path, "-v", "error",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "pipe:1",
        ]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or len(res.stdout) < 44:
            raise RuntimeError("[pyPTV] Audio extraction failed:\n"
                               + res.stderr.decode(*ENCODE_ARGS))

        import wave as _wave
        buf = io.BytesIO(res.stdout)
        with _wave.open(buf, "rb") as wf:
            n_channels  = wf.getnchannels()
            sample_rate = wf.getframerate()
            raw         = wf.readframes(wf.getnframes())

        samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
        waveform = samples.reshape(n_channels, -1).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    return _get


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _load_video_ffmpeg(video_path, decode_codec):
    video_path = strip_path(video_path)
    width, height, fps, duration, alpha = _probe_video(video_path, decode_codec)

    gen = _ffmpeg_frame_generator(video_path, width, height, alpha, decode_codec)

    channels = 4 if alpha else 3
    images = torch.from_numpy(
        np.fromiter(gen, np.dtype((np.float32, (height, width, channels))))
    )

    if len(images) == 0:
        raise RuntimeError("No frames were loaded from the video.")

    audio = _extract_audio(video_path)
    return images, fps, audio


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LoadVideoFFmpeg_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted([
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS
        ])
        return {
            "required": {
                "video": (files, {"video_upload": True}),
                "decode_codec": (PYPTV_CODECS_DECODE, {"default": "auto"}),
            },
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "load_video"

    def load_video(self, video, decode_codec):
        video_path = folder_paths.get_annotated_filepath(strip_path(video))
        images, fps, audio = _load_video_ffmpeg(video_path, decode_codec)
        return (images, fps, audio)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return hash_path(folder_paths.get_annotated_filepath(video))

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LoadVideoFFmpeg_pyPTV": LoadVideoFFmpeg_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoFFmpeg_pyPTV": "Load Video FFMPEG (pyPTV)",
}
