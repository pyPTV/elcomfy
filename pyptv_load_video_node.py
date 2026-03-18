import os
import re
import time
import subprocess
import numpy as np
import torch
import folder_paths
from comfy.utils import ProgressBar
from .pyptv_utils import ffmpeg_path, ENCODE_ARGS, strip_path, hash_path, lazy_get_audio

VIDEO_EXTENSIONS = {"mp4", "mkv", "webm", "mov", "gif"}

# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _probe_video(video_path):
    args = [ffmpeg_path, "-i", video_path, "-c", "copy", "-frames:v", "1", "-f", "null", "-"]

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

def _ffmpeg_frame_generator(video_path, width, height, alpha):
    args = [ffmpeg_path, "-v", "error", "-an",
            "-i", video_path, "-pix_fmt", "rgba64le", "-f", "rawvideo", "-"]

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
# Core loader
# ---------------------------------------------------------------------------

def _load_video_ffmpeg(video_path):
    video_path = strip_path(video_path)
    width, height, fps, duration, alpha = _probe_video(video_path)

    gen = _ffmpeg_frame_generator(video_path, width, height, alpha)

    channels = 4 if alpha else 3
    images = torch.from_numpy(
        np.fromiter(gen, np.dtype((np.float32, (height, width, channels))))
    )

    if len(images) == 0:
        raise RuntimeError("No frames were loaded from the video.")

    audio = lazy_get_audio(video_path, 0, duration)
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
            },
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = folder_paths.get_annotated_filepath(strip_path(video))
        images, fps, audio = _load_video_ffmpeg(video_path)
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
