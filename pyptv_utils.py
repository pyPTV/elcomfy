import os
import re
import subprocess
import shutil
from collections.abc import Mapping
import torch

# ---------------------------------------------------------------------------
# ffmpeg path
# ---------------------------------------------------------------------------

def _find_ffmpeg():
    env = os.environ.get("VHS_FFMPEG_PATH") or os.environ.get("FFMPEG_PATH")
    if env and os.path.isfile(env):
        return env
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for candidate in [
        os.path.join(base, "ffmpeg.exe"),
        os.path.join(base, "ffmpeg"),
        os.path.join(base, "bin", "ffmpeg.exe"),
        os.path.join(base, "bin", "ffmpeg"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg and add it to PATH, "
        "or set the FFMPEG_PATH environment variable."
    )

ffmpeg_path = _find_ffmpeg()

# ---------------------------------------------------------------------------
# Constants — match VHS exactly
# ---------------------------------------------------------------------------

ENCODE_ARGS = ("utf-8", "backslashreplace")
BIGMAX = 2 ** 53 - 1

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def strip_path(path: str) -> str:
    if path is None:
        return path
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path

def hash_path(path: str) -> str:
    if path is None or not os.path.isfile(path):
        return ""
    stat = os.stat(path)
    return f"{stat.st_mtime}_{stat.st_size}"

def is_url(path: str) -> bool:
    return path is not None and path.split("://")[0] in ["http", "https"]

def validate_path(path: str, allow_none=False):
    if path is None:
        return allow_none
    path = strip_path(path)
    if is_url(path):
        return True
    if not os.path.isfile(path):
        return f"Path does not exist: {path}"
    return True

# ---------------------------------------------------------------------------
# Audio — exact VHS pattern
# ---------------------------------------------------------------------------

def get_audio(file, start_time=0, duration=0):
    """Extract audio via ffmpeg f32le — same as VHS get_audio."""
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"],
                             capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(r', (\d+) Hz, (\w+), ', res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[pyPTV] Failed to extract audio from {file}:\n"
                           + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {"waveform": audio, "sample_rate": ar}


class LazyAudioMap(Mapping):
    """Lazy audio loader — identical to VHS LazyAudioMap."""
    def __init__(self, file, start_time=0, duration=0):
        self.file = file
        self.start_time = start_time
        self.duration = duration
        self._dict = None

    def _load(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)

    def __getitem__(self, key):
        self._load()
        return self._dict[key]

    def __iter__(self):
        self._load()
        return iter(self._dict)

    def __len__(self):
        self._load()
        return len(self._dict)


def lazy_get_audio(file, start_time=0, duration=0):
    return LazyAudioMap(file, start_time, duration)
