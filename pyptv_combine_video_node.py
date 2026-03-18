import os
import tempfile
import subprocess
import numpy as np
import folder_paths
from comfy.utils import ProgressBar
from .pyptv_utils import ffmpeg_path, ENCODE_ARGS

PYPTV_CODECS_ENCODE = ["h264", "nvenc_hevc", "hevc", "av1"]

PYPTV_PIX_FMTS = [
    "auto",
    "yuv420p",
    "yuv420p10le",
    "p010le",
    "yuv444p",
    "yuv444p10le",
]

_CODEC_DEFAULT_PIX_FMT = {
    "h264":       "yuv420p",
    "nvenc_hevc": "p010le",
    "hevc":       "yuv420p10le",
    "av1":        "yuv420p10le",
}

_CODEC_LIB = {
    "h264":       "libx264",
    "nvenc_hevc": "hevc_nvenc",
    "hevc":       "libx265",
    "av1":        "libsvtav1",
}

# ---------------------------------------------------------------------------
# Audio helper — waveform tensor back to wav temp file, no torchaudio needed
# ---------------------------------------------------------------------------

def _audio_to_temp_wav(audio) -> str | None:
    """Convert ComfyUI AUDIO dict to a temp wav file. Returns path or None."""
    try:
        import wave
        import io

        waveform = audio["waveform"]    # [1, channels, samples]
        sample_rate = audio["sample_rate"]

        # squeeze batch dim -> [channels, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        # mix down to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        samples_np = (waveform[0].clamp(-1.0, 1.0).cpu().numpy() * 32767).astype("int16")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples_np.tobytes())
        tmp.close()
        return tmp.name

    except Exception as e:
        print(f"[pyPTV] Audio conversion failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class VideoCombine_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "fps":             ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Base FPS. Connect to 'fps' output of Load Video node."
                }),
                "fps_multiplier":  ("INT", {
                    "default": 1, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Multiplies fps. 2 = double speed output."
                }),
                "encode_codec":    (PYPTV_CODECS_ENCODE, {"default": "h264"}),
                "pix_fmt":         (PYPTV_PIX_FMTS, {"default": "auto"}),
                "bitrate_mbit":    ("INT", {
                    "default": 5, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Output video bitrate in Mbit/s."
                }),
                "filename_prefix": ("STRING", {"default": "pyPTV_video"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "combine"

    def combine(self, images, fps, fps_multiplier, encode_codec, pix_fmt,
                bitrate_mbit, filename_prefix, audio=None):

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # unique output filename
        i = 1
        while True:
            out_path = os.path.join(output_dir, f"{filename_prefix}_{i:05d}.mp4")
            if not os.path.exists(out_path):
                break
            i += 1

        effective_fps = fps * fps_multiplier
        N, H, W, C = images.shape

        resolved_pix_fmt = (
            _CODEC_DEFAULT_PIX_FMT[encode_codec] if pix_fmt == "auto" else pix_fmt
        )

        # 16-bit raw frames into ffmpeg
        raw = (images.numpy() * 65535).clip(0, 65535).astype(np.uint16)
        in_pix_fmt = "rgba64le" if C == 4 else "rgb48le"

        # write audio to temp file if present (avoids pipe:3 Windows issues)
        audio_tmp = None
        if audio is not None:
            audio_tmp = _audio_to_temp_wav(audio)

        # build ffmpeg args
        args = [
            ffmpeg_path, "-y",
            "-f", "rawvideo",
            "-pix_fmt", in_pix_fmt,
            "-s", f"{W}x{H}",
            "-r", str(effective_fps),
            "-i", "pipe:0",
        ]

        if audio_tmp is not None:
            args += ["-i", audio_tmp]

        args += ["-c:v", _CODEC_LIB[encode_codec], "-pix_fmt", resolved_pix_fmt,
                 "-b:v", f"{bitrate_mbit}M", "-maxrate", f"{bitrate_mbit}M",
                 "-bufsize", f"{bitrate_mbit * 2}M"]

        if audio_tmp is not None:
            args += ["-c:a", "aac", "-shortest"]

        args += [out_path]

        # encode
        pbar = ProgressBar(N)
        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE)

            for idx in range(N):
                proc.stdin.write(raw[idx].tobytes())
                pbar.update_absolute(idx + 1, N)

            proc.stdin.close()
            proc.wait()

        except BrokenPipeError:
            raise RuntimeError("ffmpeg encode process broke pipe.")
        finally:
            if audio_tmp is not None and os.path.exists(audio_tmp):
                os.unlink(audio_tmp)

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed with return code {proc.returncode}"
            )

        result = {"filename": os.path.basename(out_path), "subfolder": "", "type": "output"}
        return {"ui": {"gifs": [result]}}

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VideoCombine_pyPTV": VideoCombine_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCombine_pyPTV": "Video Combine (pyPTV)",
}
