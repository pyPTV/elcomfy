"""
pyPTV — RIFE VFI node
Frame interpolation using RIFE model (rife47, rife49 recommended)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
from comfy.utils import ProgressBar
from comfy.model_management import get_torch_device, soft_empty_cache

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_rife_cache = {}


def _detect_arch_ver(ckpt_name: str) -> str:
    name = ckpt_name.lower()

    # Явные числовые маппинги: rife47/48/49 → arch 4.7, rife410/411/412 → arch 4.10
    _NUM_MAP = {
        "47": "4.7", "48": "4.7", "49": "4.7",
        "410": "4.10", "411": "4.10", "412": "4.10",
        "45": "4.5",
        "46": "4.6",
        "43": "4.3", "44": "4.3",
        "42": "4.2",
        "40": "4.0", "41": "4.0",
    }
    for num, ver in _NUM_MAP.items():
        if f"rife{num}" in name:
            return ver

    # fallback по точечной версии в имени файла
    for ver in ["4.10", "4.7", "4.6", "4.5", "4.3", "4.2", "4.0"]:
        if ver in name:
            return ver

    return "4.7"  # rife47/49 — самые популярные, безопаснее чем 4.10


def _load_rife(ckpt_name: str, dtype: str):
    key = (ckpt_name, dtype)
    if key in _rife_cache:
        return _rife_cache[key]

    model_path = os.path.join(folder_paths.models_dir, "rife", ckpt_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[pyPTV] RIFE model not found: {model_path}")

    arch_ver = _detect_arch_ver(ckpt_name)
    print(f"[pyPTV] Loading RIFE arch={arch_ver} from {ckpt_name}")

    # Import IFNet — must be placed alongside this file as ifnet.py
    from .ifnet import IFNet
    model = IFNet(arch_ver=arch_ver)

    sd = torch.load(model_path, map_location="cpu", weights_only=True)
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
                      fast_mode: bool, ensemble: bool,
                      dtype: torch.dtype, device: torch.device) -> list:
    """
    Recursively interpolate between img0 and img1.
    Returns list of (multiplier - 1) intermediate frames, NOT including img0/img1.
    img0, img1: [1, C, H, W] float32/16 on device
    """
    if multiplier == 1:
        return []

    scale_list = [8 * scale_factor, 4 * scale_factor,
                  2 * scale_factor, 1 * scale_factor]

    mid = model(
        img0, img1,
        timestep=0.5,
        scale_list=scale_list,
        training=False,
        fastmode=fast_mode,
        ensemble=ensemble,
    )  # [1, C, H, W]

    if multiplier == 2:
        return [mid]

    # Recursive: fill left half and right half
    left  = _interpolate_pair(model, img0, mid,  multiplier // 2,
                               scale_factor, fast_mode, ensemble, dtype, device)
    right = _interpolate_pair(model, mid,  img1, multiplier // 2,
                               scale_factor, fast_mode, ensemble, dtype, device)
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
            if f.endswith(".pth")
        ]) or ["rife49.pth"]

        return {
            "required": {
                "frames":                   ("IMAGE",),
                "ckpt_name":                (models,),
                "multiplier":               ("INT",    {"default": 2,   "min": 2, "max": 8,   "step": 1}),
                "scale_factor":             ("FLOAT",  {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.25,
                                                        "tooltip": "1.0 = standard. 0.5 = finer flow (more VRAM). 2.0 = coarser/faster."}),
                "fast_mode":                ("BOOLEAN", {"default": False}),
                "ensemble":                 ("BOOLEAN", {"default": True,
                                                         "tooltip": "Run both directions and average — reduces artifacts."}),
                "dtype":                    (["float32", "float16"], {"default": "float32"}),
                "batch_size":               ("INT",    {"default": 30,  "min": 1,  "max": 256, "step": 1}),
                "clear_cache_after_n_frames": ("INT",  {"default": 241, "min": 1,  "max": 9999}),
            },
        }

    CATEGORY    = "pyPTV"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION    = "interpolate"

    def interpolate(self, frames, ckpt_name, multiplier, scale_factor,
                    fast_mode, ensemble, dtype, batch_size, clear_cache_after_n_frames):

        model, device, torch_dtype = _load_rife(ckpt_name, dtype)

        N, H, W, C = frames.shape
        total_out = (N - 1) * multiplier + 1
        pbar = ProgressBar(N - 1)

        result = []
        frames_since_cache_clear = 0

        # Convert to [1, C, H, W] tensors on device
        def to_gpu(img_np):
            # img_np: [H, W, C] float32
            t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return t.to(device=device, dtype=torch_dtype)

        imgs_gpu = []
        for i in range(N):
            imgs_gpu.append(to_gpu(frames[i].numpy()))

        for i in range(N - 1):
            img0 = imgs_gpu[i]
            img1 = imgs_gpu[i + 1]

            # Original frame
            result.append(frames[i].numpy())

            # Interpolated frames between i and i+1
            interp = _interpolate_pair(
                model, img0, img1, multiplier,
                scale_factor, fast_mode, ensemble, torch_dtype, device
            )
            for mid in interp:
                # [1, C, H, W] → [H, W, C]
                arr = mid.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                arr = np.clip(arr, 0.0, 1.0)
                result.append(arr)

            frames_since_cache_clear += 1
            if frames_since_cache_clear >= clear_cache_after_n_frames:
                soft_empty_cache()
                frames_since_cache_clear = 0

            pbar.update_absolute(i + 1, N - 1)

        # Last frame
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
