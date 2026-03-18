import os
import io
import subprocess
import numpy as np
import folder_paths
from comfy.utils import ProgressBar
from .pyptv_utils import ffmpeg_path, ENCODE_ARGS

PYPTV_CODECS_ENCODE = ["h264", "nvenc_hevc", "hevc", "av1"]

# pix_fmt choices exposed to user
PYPTV_PIX_FMTS = [
    "auto",         # pick best for chosen codec (default)
    "yuv420p",      # 8-bit, max compatibility
    "yuv420p10le",  # 10-bit, good quality
    "p010le",       # 10-bit, NVENC/DXVA friendly
    "yuv444p",      # 8-bit, no chroma subsampling
    "yuv444p10le",  # 10-bit, no chroma subsampling
]

# default pix_fmt per codec when user picks "auto"
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
                filename_prefix, audio=None):

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # unique filename
        i = 1
        while True:
            out_path = os.path.join(output_dir, f"{filename_prefix}_{i:05d}.mp4")
            if not os.path.exists(out_path):
                break
            i += 1

        effective_fps = fps * fps_multiplier
        N, H, W, C = images.shape

        # resolve pix_fmt
        resolved_pix_fmt = (
            _CODEC_DEFAULT_PIX_FMT[encode_codec]
            if pix_fmt == "auto"
            else pix_fmt
        )

        # keep 16-bit going into ffmpeg, let ffmpeg do the conversion
        raw = (images.numpy() * 65535).clip(0, 65535).astype(np.uint16)
        in_pix_fmt = "rgba64le" if C == 4 else "rgb48le"

        args = [
            ffmpeg_path, "-y",
            "-f", "rawvideo",
            "-pix_fmt", in_pix_fmt,
            "-s", f"{W}x{H}",
            "-r", str(effective_fps),
            "-i", "pipe:0",
        ]

        # audio
        audio_bytes = None
        if audio is not None:
            try:
                import torchaudio
                waveform = audio["waveform"].squeeze(0)
                sample_rate = audio["sample_rate"]
                buf = io.BytesIO()
                torchaudio.save(buf, waveform, sample_rate, format="wav")
                audio_bytes = buf.getvalue()
            except Exception:
                audio_bytes = None

        if audio_bytes is not None:
            args += ["-f", "wav", "-i", "pipe:3"]

        args += ["-c:v", _CODEC_LIB[encode_codec], "-pix_fmt", resolved_pix_fmt]

        if audio_bytes is not None:
            args += ["-c:a", "aac", "-shortest"]

        args += [out_path]

        pbar = ProgressBar(N)

        try:
            if audio_bytes is not None:
                r_fd, w_fd = os.pipe()
                proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                        pass_fds=(r_fd,))
                os.close(r_fd)
                os.write(w_fd, audio_bytes)
                os.close(w_fd)
            else:
                proc = subprocess.Popen(args, stdin=subprocess.PIPE)

            for idx in range(N):
                proc.stdin.write(raw[idx].tobytes())
                pbar.update_absolute(idx + 1, N)

            proc.stdin.close()
            proc.wait()

        except BrokenPipeError:
            raise RuntimeError("ffmpeg encode process broke pipe.")

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed with return code {proc.returncode}"
            )

        return {"ui": {"videos": [{"filename": os.path.basename(out_path),
                                   "subfolder": "",
                                   "type": "output"}]}}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "VideoCombine_pyPTV": VideoCombine_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCombine_pyPTV": "Video Combine (pyPTV)",
}
