"""
pyPTV — Image Crop node
Crops 4 pixels from top/bottom center to 1472×828 (16:9).
"""
import torch

class ImageCrop_pyPTV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    CATEGORY     = "pyPTV"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION     = "crop"

    def crop(self, images):
        target_w, target_h = 1472, 828
        N, H, W, C = images.shape
        y0 = (H - target_h) // 2
        x0 = (W - target_w) // 2
        return (images[:, y0:y0 + target_h, x0:x0 + target_w, :],)

NODE_CLASS_MAPPINGS = {
    "ImageCrop_pyPTV": ImageCrop_pyPTV,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCrop_pyPTV": "Image Crop (pyPTV)",
}
