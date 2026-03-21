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

    model_path = os.path.join(folder_paths.models_dir, "rife", ckpt_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[pyPTV] RIFE model not found: {model_path}")

    print(f"[pyPTV] Loading RIFE v4.25/4.26 from {ckpt_name}")

    from .ifnet import IFNet
    model = IFNet()

    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    device = get_torch_device()
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = model.to(device=device, dtype=torch_dtype).eval()

    _rife_cache[key] = (model, device, torch_dtype)
    return model, device, torch_dtype


# ---------------------------------------------------------------------------
# Batch interpolation — один проход через модель для N пар
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _interp_batch(model, img0_batch: torch.Tensor, img1_batch: torch.Tensor,
                  scale_factor: float) -> torch.Tensor:
    """
    img0_batch, img1_batch: [B, C, H, W]
    returns: [B, C, H, W] — промежуточные кадры
    """
    scale_list = [
        16 / scale_factor,
        8  / scale_factor,
        4  / scale_factor,
        2  / scale_factor,
        1  / scale_factor,
    ]
    return model(
        img0_batch, img1_batch,
        timestep=0.5,
        scale_list=scale_list,
        training=False,
        ensemble=False,
    )


@torch.inference_mode()
def _interpolate_recursive(model, img0_batch: torch.Tensor, img1_batch: torch.Tensor,
                            multiplier: int, scale_factor: float) -> list:
    """
    Рекурсивно интерполирует между img0_batch и img1_batch.
    Возвращает список тензоров [B, C, H, W] промежуточных кадров
    в правильном порядке (без img0 и img1).
    """
    if multiplier == 1:
        return []

    mid = _interp_batch(model, img0_batch, img1_batch, scale_factor)

    if multiplier == 2:
        return [mid]

    left  = _interpolate_recursive(model, img0_batch, mid,  multiplier // 2, scale_factor)
    right = _interpolate_recursive(model, mid,  img1_batch, multiplier // 2, scale_factor)
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
                "multiplier":   ("INT",   {"default": 2,    "min": 2,    "max": 8,    "step": 1}),
                "scale_factor": ("FLOAT", {"default": 1.0,  "min": 0.25, "max": 4.0,  "step": 0.25,
                                           "tooltip": "1.0 = standard. 0.5 = finer flow (more VRAM). 2.0 = coarser/faster."}),
                "batch_size":   ("INT",   {"default": 16,   "min": 8,    "max": 64,   "step": 8,
                                           "tooltip": "Pairs processed per GPU call. Higher = faster but more VRAM."}),
                "dtype":        (["float32", "float16"], {"default": "float32"}),
                "clear_cache_after_n_frames": ("INT", {"default": 241, "min": 1, "max": 9999}),
            },
        }

    CATEGORY     = "pyPTV"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION     = "interpolate"

    def interpolate(self, frames, ckpt_name, multiplier, scale_factor,
                    batch_size, dtype, clear_cache_after_n_frames):

        model, device, torch_dtype = _load_rife(ckpt_name, dtype)

        N, H, W, C = frames.shape
        num_pairs = N - 1
        pbar = ProgressBar(num_pairs)

        # Конвертируем все кадры в GPU тензоры [C, H, W]
        all_frames = torch.from_numpy(
            frames.numpy()
        ).permute(0, 3, 1, 2).to(device=device, dtype=torch_dtype)  # [N, C, H, W]

        # Результат: список numpy массивов [H, W, C]
        result = []
        frames_since_cache_clear = 0

        # Обрабатываем пары батчами
        i = 0
        while i < num_pairs:
            b_end = min(i + batch_size, num_pairs)
            b_size = b_end - i

            img0_batch = all_frames[i:i + b_size]        # [B, C, H, W]
            img1_batch = all_frames[i + 1:i + b_size + 1]  # [B, C, H, W]

            # Получаем промежуточные кадры для каждого уровня рекурсии
            # interp_levels — список тензоров [B, C, H, W]
            interp_levels = _interpolate_recursive(
                model, img0_batch, img1_batch, multiplier, scale_factor
            )

            # Собираем результат попарно: img0[j], interp[j], ..., (img1 добавим на след итерации)
            for j in range(b_size):
                # оригинальный кадр
                result.append(
                    all_frames[i + j].permute(1, 2, 0).float().cpu().numpy()
                )
                # промежуточные кадры в правильном порядке
                for level_tensor in interp_levels:
                    arr = level_tensor[j].permute(1, 2, 0).float().cpu().numpy()
                    result.append(np.clip(arr, 0.0, 1.0))

            frames_since_cache_clear += b_size
            if frames_since_cache_clear >= clear_cache_after_n_frames:
                soft_empty_cache()
                frames_since_cache_clear = 0

            pbar.update_absolute(b_end, num_pairs)
            i += b_size

        # Последний кадр
        result.append(all_frames[N - 1].permute(1, 2, 0).float().cpu().numpy())

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
