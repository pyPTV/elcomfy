"""
pyPTV — RIFE VFI node
Frame interpolation using RIFE model (v4.25 / v4.26 recommended)
"""

import os
import torch
import numpy as np
import folder_paths
from comfy.utils import ProgressBar
from comfy.model_management import get_torch_device, soft_empty_cache

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_rife_cache = {}


def _load_rife(ckpt_name: str, dtype: str):
    key = (ckpt_name, dtype)
    if key in _rife_cache:
        return _rife_cache[key]

    # поддерживаем .pth и .pkl
    model_path = os.path.join(folder_paths.models_dir, "rife", ckpt_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[pyPTV] RIFE model not found: {model_path}")

    print(f"[pyPTV] Loading RIFE v4.25/4.26 from {ckpt_name}")

    from .ifnet import IFNet
    model = IFNet()

    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    # strip "module." prefix if DDP checkpoint
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    device = get_torch_device()
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = model.to(device=device, dtype=torch_dtype).eval()

    _rife_cache[key] = (model, device, torch_dtype)
    return model, device, torch_dtype


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _interpolate_pair(model, img0: torch.Tensor, img1: torch.Tensor,
                      multiplier: int, scale_factor: float,
                      fast_mode: bool,
                      dtype: torch.dtype, device: torch.device) -> list:
    """
    Recursively interpolate between img0 and img1.
    Returns list of (multiplier - 1) intermediate frames, NOT including img0/img1.
    img0, img1: [1, C, H, W] on device
    """
    if multiplier == 1:
        return []

    # v4.25 uses 5-level scale list
    scale_list = [
        16 / scale_factor,
        8  / scale_factor,
        4  / scale_factor,
        2  / scale_factor,
        1  / scale_factor,
    ]

    mid = model(
        img0, img1,
        timestep=0.5,
        scale_list=scale_list,
        training=False,
        fastmode=fast_mode,
        ensemble=False,
    )  # [1, C, H, W]

    if multiplier == 2:
        return [mid]

    left  = _interpolate_pair(model, img0, mid,  multiplier // 2,
                               scale_factor, fast_mode, dtype, device)
    right = _interpolate_pair(model, mid,  img1, multiplier // 2,
                               scale_factor, fast_mode, dtype, device)
    return left + [mid] + right


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RIFEInterpolate_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        rife_dir = os.path.join(folder_paths.models_dir, "rife")
        os.makedirs(rife_dir, exist_ok=True)
        models = sorted([
            f for f in os.listdir(rife_dir)
            if f.endswith(".pth") or f.endswith(".pkl")
        ]) or ["flownet.pkl"]

        return {
            "required": {
                "frames":       ("IMAGE",),
                "ckpt_name":    (models,),
                "multiplier":   ("INT",   {"default": 2, "min": 2, "max": 8, "step": 1}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25,
                                           "tooltip": "1.0 = standard. 0.5 = finer flow (more VRAM). 2.0 = coarser/faster."}),
                "fast_mode":    ("BOOLEAN", {"default": True}),
                "dtype":        (["float32", "float16"], {"default": "float32"}),
                "batch_size":   ("INT",   {"default": 30, "min": 1, "max": 256, "step": 1}),
                "clear_cache_after_n_frames": ("INT", {"default": 241, "min": 1, "max": 9999}),
            },
        }

    CATEGORY     = "pyPTV"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION     = "interpolate"

    def interpolate(self, frames, ckpt_name, multiplier, scale_factor,
                    fast_mode, dtype, batch_size, clear_cache_after_n_frames):

        model, device, torch_dtype = _load_rife(ckpt_name, dtype)

        N, H, W, C = frames.shape
        pbar = ProgressBar(N - 1)

        result = []
        frames_since_cache_clear = 0

        def to_gpu(img_np):
            t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return t.to(device=device, dtype=torch_dtype)

        imgs_gpu = [to_gpu(frames[i].numpy()) for i in range(N)]

        for i in range(N - 1):
            img0 = imgs_gpu[i]
            img1 = imgs_gpu[i + 1]

            result.append(frames[i].numpy())

            interp = _interpolate_pair(
                model, img0, img1, multiplier,
                scale_factor, fast_mode, torch_dtype, device
            )
            for mid in interp:
                arr = mid.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                arr = np.clip(arr, 0.0, 1.0)
                result.append(arr)

            frames_since_cache_clear += 1
            if frames_since_cache_clear >= clear_cache_after_n_frames:
                soft_empty_cache()
                frames_since_cache_clear = 0

            pbar.update_absolute(i + 1, N - 1)

        result.append(frames[N - 1].numpy())

        out = torch.from_numpy(np.stack(result, axis=0))
        return (out,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RIFEInterpolate_pyPTV": RIFEInterpolate_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFEInterpolate_pyPTV": "RIFE VFI (pyPTV)",
}
