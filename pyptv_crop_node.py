"""
pyPTV — Image Crop node
Crop a batch of images to a preset resolution from center.
"""

import torch

_PRESETS = {
    "1080p  →  1920×1080": (1920, 1080),
    "720p   →  1280×720":  (1280, 720),
}


class ImageCrop_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":     ("IMAGE",),
                "dimensions": (list(_PRESETS.keys()),),
            },
        }

    CATEGORY     = "pyPTV"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION     = "crop"

    def crop(self, images, dimensions):
        target_w, target_h = _PRESETS[dimensions]

        N, H, W, C = images.shape

        crop_h = min(target_h, H)
        crop_w = min(target_w, W)

        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2

        cropped = images[:, y0:y0 + crop_h, x0:x0 + crop_w, :]
        return (cropped,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ImageCrop_pyPTV": ImageCrop_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCrop_pyPTV": "Image Crop (pyPTV)",
}
