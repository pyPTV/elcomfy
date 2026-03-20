"""
pyPTV — Image Crop node
Crop a batch of images to a target resolution from center (or custom offset).
"""

import torch


class ImageCrop_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":    ("IMAGE",),
                "width":     ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 2,
                                      "tooltip": "Target crop width in pixels."}),
                "height":    ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 2,
                                      "tooltip": "Target crop height in pixels."}),
                "align":     (["center", "top-left", "custom"], {"default": "center"}),
            },
            "optional": {
                "x":         ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                      "tooltip": "X offset (used when align=custom)."}),
                "y":         ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                      "tooltip": "Y offset (used when align=custom)."}),
            },
        }

    CATEGORY     = "pyPTV"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION     = "crop"

    def crop(self, images, width, height, align, x=0, y=0):
        # images: [N, H, W, C]
        N, H, W, C = images.shape

        crop_h = min(height, H)
        crop_w = min(width,  W)

        if align == "center":
            y0 = (H - crop_h) // 2
            x0 = (W - crop_w) // 2
        elif align == "top-left":
            y0, x0 = 0, 0
        else:  # custom
            y0 = max(0, min(y, H - crop_h))
            x0 = max(0, min(x, W - crop_w))

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
