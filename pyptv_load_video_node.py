import os
import re
import time
import subprocess
import numpy as np
import torch
import folder_paths
from comfy.utils import ProgressBar
from .pyptv_utils import (
    ffmpeg_path, ENCODE_ARGS, BIGMAX,
    lazy_get_audio, strip_path, validate_path,
    hash_path, is_url
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYPTV_CODECS_DECODE = ["auto", "h264", "hevc", "av1", "vp9"]

VIDEO_EXTENSIONS = {"mp4", "mkv", "webm", "mov", "gif"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_video(video_path, decode_codec="auto"):
    """Run a dummy ffmpeg pass to extract width, height, fps, duration, alpha."""
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
    if dur_m:
        duration = (int(dur_m.group(1)) * 3600
                    + int(dur_m.group(2)) * 60
                    + float(dur_m.group(3)))
    else:
        duration = 0.0

    return width, height, fps, duration, alpha


def _ffmpeg_frame_generator(video_path, width, height, alpha,
                             force_rate, frame_load_cap, decode_codec):
    """Yield raw float32 frames [H, W, C] from ffmpeg stdout."""
    args = [ffmpeg_path, "-v", "error", "-an"]

    if decode_codec == "vp9":
        args += ["-c:v", "libvpx-vp9"]
    elif decode_codec != "auto":
        args += ["-c:v", decode_codec]

    args += ["-i", video_path, "-pix_fmt", "rgba64le"]

    if force_rate > 0:
        args += ["-vf", f"fps=fps={force_rate}"]

    if frame_load_cap > 0:
        args += ["-frames:v", str(frame_load_cap)]

    args += ["-f", "rawvideo", "-"]

    # rgba64le: 4 channels * 2 bytes = 8 bytes per pixel
    bpi = width * height * 8
    total = frame_load_cap if frame_load_cap > 0 else 0
    pbar = ProgressBar(total or 1)
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
                        pbar.update_absolute(frames_yielded, total or frames_yielded + 1)
                        if sig is not None:
                            return
                    prev = frame
                    offset = 0

    except BrokenPipeError:
        raise RuntimeError("ffmpeg process broke pipe unexpectedly.")

    if prev is not None:
        yield prev


def _load_video_ffmpeg(video_path, force_rate, frame_load_cap, decode_codec):
    """Core loader — returns (images, fps, audio)."""
    video_path = strip_path(video_path)
    width, height, fps, duration, alpha = _probe_video(video_path, decode_codec)

    gen = _ffmpeg_frame_generator(
        video_path, width, height, alpha,
        force_rate, frame_load_cap, decode_codec
    )

    channels = 4 if alpha else 3
    images = torch.from_numpy(
        np.fromiter(gen, np.dtype((np.float32, (height, width, channels))))
    )

    if len(images) == 0:
        raise RuntimeError("No frames were loaded from the video.")

    out_fps = float(force_rate) if force_rate > 0 else fps
    duration_loaded = len(images) / out_fps
    audio = lazy_get_audio(video_path, 0, duration_loaded)

    return images, out_fps, audio


# ---------------------------------------------------------------------------
# NODE: LoadVideoFFmpeg_pyPTV  (upload picker)
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
                "video": (files,),
                "decode_codec": (PYPTV_CODECS_DECODE, {"default": "auto"}),
                "force_rate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Override FPS. 0 = use source FPS."
                }),
                "frame_load_cap": ("INT", {
                    "default": 0, "min": 0, "max": BIGMAX, "step": 1,
                    "tooltip": "Max frames to load. 0 = all."
                }),
            },
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "load_video"

    def load_video(self, video, decode_codec, force_rate, frame_load_cap):
        video_path = folder_paths.get_annotated_filepath(strip_path(video))
        images, fps, audio = _load_video_ffmpeg(
            video_path, force_rate, frame_load_cap, decode_codec
        )
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
# NODE: LoadVideoFFmpegPath_pyPTV  (path string)
# ---------------------------------------------------------------------------

class LoadVideoFFmpegPath_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {
                    "placeholder": "X://insert/path/here.mp4",
                    "vhs_path_extensions": list(VIDEO_EXTENSIONS)
                }),
                "decode_codec": (PYPTV_CODECS_DECODE, {"default": "auto"}),
                "force_rate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Override FPS. 0 = use source FPS."
                }),
                "frame_load_cap": ("INT", {
                    "default": 0, "min": 0, "max": BIGMAX, "step": 1,
                    "tooltip": "Max frames to load. 0 = all."
                }),
            },
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "load_video"

    def load_video(self, video, decode_codec, force_rate, frame_load_cap):
        if not video or validate_path(video) is not True:
            raise ValueError(f"Not a valid path: {video}")
        if is_url(video):
            video = try_download_video(video) or video
        images, fps, audio = _load_video_ffmpeg(
            video, force_rate, frame_load_cap, decode_codec
        )
        return (images, fps, audio)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        return validate_path(video, allow_none=True)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LoadVideoFFmpeg_pyPTV":     LoadVideoFFmpeg_pyPTV,
    "LoadVideoFFmpegPath_pyPTV": LoadVideoFFmpegPath_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoFFmpeg_pyPTV":     "Load Video FFMPEG (pyPTV)",
    "LoadVideoFFmpegPath_pyPTV": "Load Video FFMPEG Path (pyPTV)",
}
