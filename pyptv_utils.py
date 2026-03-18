import os
import subprocess
import shutil

# ---------------------------------------------------------------------------
# ffmpeg path
# ---------------------------------------------------------------------------

def _find_ffmpeg():
    # 1. env override
    env = os.environ.get("VHS_FFMPEG_PATH") or os.environ.get("FFMPEG_PATH")
    if env and os.path.isfile(env):
        return env
    # 2. next to ComfyUI (common portable installs)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for candidate in [
        os.path.join(base, "ffmpeg.exe"),
        os.path.join(base, "ffmpeg"),
        os.path.join(base, "bin", "ffmpeg.exe"),
        os.path.join(base, "bin", "ffmpeg"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    # 3. system PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg and add it to PATH, "
        "or set the FFMPEG_PATH environment variable."
    )

ffmpeg_path = _find_ffmpeg()

# ---------------------------------------------------------------------------
# Encode args (encoding hint for stderr decoding)
# ---------------------------------------------------------------------------

ENCODE_ARGS = ("utf-8", "replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIGMAX = 2 ** 31 - 1

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def strip_path(path: str) -> str:
    """Remove surrounding quotes and whitespace."""
    if path is None:
        return path
    path = path.strip()
    if len(path) >= 2 and path[0] in ('"', "'") and path[-1] == path[0]:
        path = path[1:-1]
    return path


def validate_path(path: str, allow_none=False) -> bool | str:
    if path is None:
        return allow_none
    path = strip_path(path)
    if is_url(path):
        return True
    if not os.path.isfile(path):
        return f"Path does not exist: {path}"
    return True


def hash_path(path: str) -> str:
    """Return mtime+size as a cheap change-detection hash."""
    if path is None or not os.path.isfile(path):
        return ""
    stat = os.stat(path)
    return f"{stat.st_mtime}_{stat.st_size}"


def is_url(path: str) -> bool:
    return path is not None and (
        path.startswith("http://") or path.startswith("https://")
    )

# ---------------------------------------------------------------------------
# lazy_get_audio
# ---------------------------------------------------------------------------

def lazy_get_audio(video_path: str, start_time: float = 0.0,
                   duration: float = 0.0):
    """
    Returns a callable that extracts audio on demand.
    The callable returns a dict {"waveform": tensor, "sample_rate": int}
    or None if audio extraction fails.
    """
    def _extract():
        try:
            import torchaudio
            import torch
            import io

            args = [ffmpeg_path, "-v", "error"]
            if start_time > 0:
                args += ["-ss", str(start_time)]
            args += ["-i", video_path]
            if duration > 0:
                args += ["-t", str(duration)]
            args += ["-vn", "-f", "wav", "pipe:1"]

            res = subprocess.run(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            if res.returncode != 0 or len(res.stdout) == 0:
                return None

            buf = io.BytesIO(res.stdout)
            waveform, sample_rate = torchaudio.load(buf)
            # ComfyUI AUDIO format: waveform shape [1, channels, samples]
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except Exception:
            return None

    return _extract
