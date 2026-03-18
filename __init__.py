NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .elevenlabs_voice_changer_node import (
        NODE_CLASS_MAPPINGS       as _M1,
        NODE_DISPLAY_NAME_MAPPINGS as _D1,
    )
    NODE_CLASS_MAPPINGS.update(_M1)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D1)
except Exception as e:
    print(f"[pyPTV] Failed to load elevenlabs_voice_changer_node: {e}")

try:
    from .elevenlabs_fal_voice_changer_node import (
        NODE_CLASS_MAPPINGS       as _M2,
        NODE_DISPLAY_NAME_MAPPINGS as _D2,
    )
    NODE_CLASS_MAPPINGS.update(_M2)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D2)
except Exception as e:
    print(f"[pyPTV] Failed to load elevenlabs_fal_voice_changer_node: {e}")

try:
    from .pyptv_load_video_node import (
        NODE_CLASS_MAPPINGS       as _M3,
        NODE_DISPLAY_NAME_MAPPINGS as _D3,
    )
    NODE_CLASS_MAPPINGS.update(_M3)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D3)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_load_video_node: {e}")

try:
    from .pyptv_combine_video_node import (
        NODE_CLASS_MAPPINGS       as _M4,
        NODE_DISPLAY_NAME_MAPPINGS as _D4,
    )
    NODE_CLASS_MAPPINGS.update(_M4)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D4)
except Exception as e:
    print(f"[pyPTV] Failed to load pyptv_combine_video_node: {e}")

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
