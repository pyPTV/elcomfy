"""
training_ltx23_lora.py  —  pyPTV
Node: Training LTX-2.3 LoRA (pyPTV)

Assumes everything is pre-installed:
  - LTX-2 repo cloned at LTX2_REPO with `uv sync` already done
  - model checkpoint + text encoder already on disk
  - dataset images already in the folder passed via `dataset_path`

The node:
  1. Builds dataset.json from all files in dataset_path
  2. Runs process_dataset.py (precompute latents)
  3. Writes train_config.yaml
  4. Runs train.py
  5. Returns path to final lora_weights.safetensors + full log text
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

# ── paths (everything is pre-installed) ──────────────────────────────────────
LTX2_REPO      = Path("/comfyui/LTX-2")
TRAINER_PKG    = LTX2_REPO / "packages" / "ltx-trainer"
TRAIN_SCRIPT   = TRAINER_PKG / "scripts" / "train.py"
PREPROC_SCRIPT = TRAINER_PKG / "scripts" / "process_dataset.py"


# ── minimal YAML serialiser (no external dep at import time) ──────────────────
def _to_yaml(obj, indent: int = 0) -> str:
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                lines.append(f"{pad}{k}: null")
            elif isinstance(v, bool):
                lines.append(f"{pad}{k}: {'true' if v else 'false'}")
            elif isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            elif isinstance(v, str):
                lines.append(f"{pad}{k}: {v!r}")
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_to_yaml(item, indent + 1))
            elif isinstance(item, str):
                lines.append(f"{pad}- {item!r}")
            else:
                lines.append(f"{pad}- {item}")
    return "\n".join(lines)


# ── subprocess streamer ───────────────────────────────────────────────────────
def _run_streaming(
    cmd: list[str],
    cwd: str,
    env: dict,
    log_cb,
    done_event: threading.Event,
):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    done_event._rc = proc.returncode
    done_event.set()


# ── Node ──────────────────────────────────────────────────────────────────────
class TrainingLTX23LoRA_pyPTV:
    """
    Training LTX-2.3 LoRA (pyPTV)

    Inputs
    ──────
    dataset_path     — folder that already contains images (no filtering)
    model_checkpoint — /comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors
    text_encoder_dir — /comfyui/models/text_encoders/gemma_3_12B_it
    output_dir       — destination for lora_weights.safetensors
    trigger_word     — prepended to every caption
    default_caption  — caption text shared by all images
    resolution       — WxHxFrames  (1 frame for image-only dataset)
    steps / lora_rank / lora_alpha / learning_rate
    skip_preprocess  — reuse cached latents from a previous run

    Outputs
    ───────
    lora_path  (STRING) — absolute path to lora_weights.safetensors
    log_text   (STRING) — full stdout/stderr log (feed into Log Viewer node)
    """

    CATEGORY     = "pyPTV"
    FUNCTION     = "train"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "log_text")
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── paths ─────────────────────────────────────────────────────
                "dataset_path": ("STRING", {
                    "default": "/comfyui/datasets/my_character",
                    "multiline": False,
                }),
                "model_checkpoint": ("STRING", {
                    "default": "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors",
                    "multiline": False,
                }),
                "text_encoder_dir": ("STRING", {
                    "default": "/comfyui/models/text_encoders/gemma_3_12B_it",
                    "multiline": False,
                }),
                "output_dir": ("STRING", {
                    "default": "/comfyui/output/lora_training",
                    "multiline": False,
                }),
                # ── caption ───────────────────────────────────────────────────
                "trigger_word": ("STRING", {
                    "default": "JSRv1rpd",
                    "multiline": False,
                }),
                "default_caption": ("STRING", {
                    "default": "woman in a dark-blue suit, ash-brown shoulder-length hair, grey-blue eyes",
                    "multiline": True,
                }),
                # ── training params ───────────────────────────────────────────
                "resolution": ("STRING", {
                    "default": "1024x1024x1",
                    "multiline": False,
                    "tooltip": "WxHxFrames — use 1 frame for image-only dataset",
                }),
                "steps": ("INT", {
                    "default": 2000, "min": 100, "max": 20000, "step": 100,
                }),
                "lora_rank": ("INT", {
                    "default": 32, "min": 4, "max": 128, "step": 4,
                }),
                "lora_alpha": ("INT", {
                    "default": 16, "min": 4, "max": 128, "step": 4,
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4, "min": 1e-6, "max": 1e-2,
                    "step": 1e-6, "round": False,
                }),
                # ── misc ──────────────────────────────────────────────────────
                "skip_preprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip process_dataset.py if latents already cached",
                }),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        dataset_path: str,
        model_checkpoint: str,
        text_encoder_dir: str,
        output_dir: str,
        trigger_word: str,
        default_caption: str,
        resolution: str,
        steps: int,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        skip_preprocess: bool,
    ):
        log_lines: list[str] = []

        def log(msg: str):
            log_lines.append(msg)
            print(msg, flush=True)

        log("══ Training LTX-2.3 LoRA (pyPTV) ══")

        ds_dir  = Path(dataset_path)
        out_dir = Path(output_dir)
        preproc = ds_dir / ".precomputed"

        # ── 1. Build dataset.json ─────────────────────────────────────────────
        log("Building dataset.json …")
        images = sorted(p for p in ds_dir.iterdir() if p.is_file())
        if not images:
            raise RuntimeError(f"No files found in {ds_dir}")

        full_caption = f"{trigger_word} {default_caption}".strip()
        entries = [
            {"caption": full_caption, "media_path": str(img)}
            for img in images
        ]
        dataset_json = ds_dir.parent / f"{ds_dir.name}_dataset.json"
        dataset_json.write_text(json.dumps(entries, indent=2))
        log(f"dataset.json → {dataset_json}  ({len(entries)} images)")

        env = os.environ.copy()

        # ── 2. Preprocess ─────────────────────────────────────────────────────
        if not skip_preprocess:
            log("── Preprocessing (computing latents) ──")
            cmd = [
                "uv", "run", "python", str(PREPROC_SCRIPT),
                str(dataset_json),
                "--resolution-buckets", resolution,
                "--model-path",         model_checkpoint,
                "--text-encoder-path",  text_encoder_dir,
                "--lora-trigger",       trigger_word,
                "--output-dir",         str(preproc),
            ]
            log("CMD: " + " ".join(cmd))
            done = threading.Event()
            done._rc = -1
            t = threading.Thread(
                target=_run_streaming,
                args=(cmd, str(TRAINER_PKG), env, log, done),
                daemon=True,
            )
            t.start()
            t.join()
            if done._rc != 0:
                raise RuntimeError(f"process_dataset.py exited with code {done._rc}")
            log("Preprocessing done ✓")
        else:
            log(f"Skipping preprocessing — using cached latents in {preproc}")

        # ── 3. Write train_config.yaml ────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "output_dir": str(out_dir),
            "model": {
                "model_path":        model_checkpoint,
                "text_encoder_path": text_encoder_dir,
                "training_mode":     "lora",
                "load_checkpoint":   None,
            },
            "lora": {
                "rank":           lora_rank,
                "alpha":          lora_alpha,
                "dropout":        0.0,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            },
            "training_strategy": {
                "name":                       "text_to_video",
                "first_frame_conditioning_p": 0.0,
                "with_audio":                 False,
            },
            "optimization": {
                "learning_rate":                 learning_rate,
                "steps":                         steps,
                "batch_size":                    1,
                "gradient_accumulation_steps":   1,
                "max_grad_norm":                 1.0,
                "optimizer_type":                "adamw8bit",
                "scheduler_type":                "linear",
                "enable_gradient_checkpointing": True,
            },
            "acceleration": {
                "mixed_precision_mode":      "bf16",
                "quantization":              None,
                "load_text_encoder_in_8bit": True,
            },
            "data": {
                "preprocessed_data_root": str(preproc),
                "num_dataloader_workers": 2,
            },
            "validation": {
                "prompts":                 [f"{trigger_word} portrait"],
                "negative_prompt":         "worst quality, blurry",
                "video_dims":              [512, 512, 1],
                "frame_rate":              25.0,
                "seed":                    42,
                "inference_steps":         20,
                "interval":                None,
                "videos_per_prompt":       1,
                "guidance_scale":          3.0,
                "stg_scale":               0.0,
                "generate_audio":          False,
                "skip_initial_validation": True,
            },
            "checkpoints": {
                "interval":    None,
                "keep_last_n": 1,
                "precision":   "bfloat16",
            },
            "hub":           {"push_to_hub": False},
            "wandb":         {"enabled": False},
            "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal"},
        }

        cfg_path = out_dir / "train_config.yaml"
        cfg_path.write_text(_to_yaml(cfg))
        log(f"Config → {cfg_path}")

        # ── 4. Train ──────────────────────────────────────────────────────────
        log("── Starting training ──")
        train_cmd = [
            "uv", "run", "python", str(TRAIN_SCRIPT), str(cfg_path),
        ]
        log("CMD: " + " ".join(train_cmd))

        done2 = threading.Event()
        done2._rc = -1
        t2 = threading.Thread(
            target=_run_streaming,
            args=(train_cmd, str(TRAINER_PKG), env, log, done2),
            daemon=True,
        )
        t2.start()
        t2.join()

        if done2._rc != 0:
            raise RuntimeError(f"train.py exited with code {done2._rc}")

        # ── 5. Find output LoRA ───────────────────────────────────────────────
        lora_path = out_dir / "lora_weights.safetensors"
        if not lora_path.exists():
            candidates = list(out_dir.rglob("lora_weights.safetensors"))
            if not candidates:
                raise RuntimeError(
                    f"lora_weights.safetensors not found in {out_dir}"
                )
            lora_path = candidates[0]

        log(f"✓ LoRA ready → {lora_path}")

        log_text = "\n".join(log_lines)
        return (str(lora_path), log_text)


# ── Mappings ──────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "TrainingLTX23LoRA_pyPTV": TrainingLTX23LoRA_pyPTV,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainingLTX23LoRA_pyPTV": "Training LTX-2.3 LoRA (pyPTV)",
}


