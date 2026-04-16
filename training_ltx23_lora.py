"""
training_ltx23_lora.py  —  pyPTV
Node: Training LTX-2.3 LoRA (pyPTV)

Trains a character LoRA using the official Lightricks ltx-trainer
(https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer).

Dataset is downloaded from HuggingFace (images only, no .txt captions).
One local folder per dataset run.
Outputs a single final lora_weights.safetensors — no intermediate checkpoints.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
COMFY_ROOT     = Path(__file__).resolve().parents[2]   # comfyui/
LTX2_REPO      = Path("/comfyui/LTX-2")                # cloned repo
TRAINER_PKG    = LTX2_REPO / "packages" / "ltx-trainer"
TRAIN_SCRIPT   = TRAINER_PKG / "scripts" / "train.py"
PREPROC_SCRIPT = TRAINER_PKG / "scripts" / "process_dataset.py"

# model defaults (may be overridden by node inputs)
DEFAULT_MODEL_CKPT   = "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors"
DEFAULT_TEXT_ENCODER = "/comfyui/models/text_encoders/gemma_3_12B_it"


# ── helpers ───────────────────────────────────────────────────────────────────

def _uv_python() -> str:
    """Return 'uv run python' if uv is available, else sys.executable."""
    if shutil.which("uv"):
        return "uv run python"
    return sys.executable


def _ensure_ltx2_repo(log_cb):
    """Clone LTX-2 repo if not present, install with uv."""
    if not LTX2_REPO.exists():
        log_cb("[pyPTV] Cloning Lightricks/LTX-2 …")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Lightricks/LTX-2.git",
             str(LTX2_REPO)],
            check=True,
        )
    if not (LTX2_REPO / ".uv_synced").exists():
        log_cb("[pyPTV] Running 'uv sync' in LTX-2 repo …")
        subprocess.run(
            ["uv", "sync"],
            cwd=str(LTX2_REPO),
            check=True,
        )
        (LTX2_REPO / ".uv_synced").touch()


def _download_hf_dataset(repo_id: str, hf_token: str, local_dir: Path, log_cb):
    """Download image files from a HuggingFace dataset repo into local_dir."""
    log_cb(f"[pyPTV] Downloading dataset {repo_id} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # use huggingface_hub snapshot_download (images only)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")

    py = sys.executable
    patterns_arg = ",".join(IMAGE_EXTS)

    code = textwrap.dedent(f"""
import os, sys
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN")
path = snapshot_download(
    repo_id={repo_id!r},
    repo_type="dataset",
    local_dir={str(local_dir)!r},
    allow_patterns={list(IMAGE_EXTS)!r},
    token=token,
    ignore_patterns=["*.parquet","*.arrow","*.json","*.csv","*.txt","*.md"],
)
print("Downloaded to:", path)
""")
    result = subprocess.run(
        [py, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"HF download failed:\n{result.stderr}")
    log_cb(f"[pyPTV] {result.stdout.strip()}")


def _build_dataset_json(images_dir: Path, caption: str, trigger_word: str) -> Path:
    """Create dataset.json with one entry per image."""
    import json

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    full_caption = f"{trigger_word} {caption}".strip()
    entries = [{"caption": full_caption, "media_path": str(img)} for img in images]

    json_path = images_dir.parent / "dataset.json"
    json_path.write_text(json.dumps(entries, indent=2))
    return json_path


def _write_train_config(
    cfg_path: Path,
    model_ckpt: str,
    text_encoder: str,
    preproc_root: str,
    output_dir: str,
    trigger_word: str,
    lora_rank: int,
    lora_alpha: int,
    lr: float,
    steps: int,
    resolution: str,
    validation_prompt: str,
):
    """Write a minimal ltx-trainer YAML config."""
    import yaml   # bundled with ltx-trainer env; fall back to manual write

    cfg = {
        "output_dir": output_dir,
        "model": {
            "model_path": model_ckpt,
            "text_encoder_path": text_encoder,
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "training_strategy": {
            "name": "text_to_video",
            "first_frame_conditioning_p": 0.0,
            "with_audio": False,
        },
        "optimization": {
            "learning_rate": lr,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw8bit",
            "scheduler_type": "linear",
            "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            "quantization": None,
            "load_text_encoder_in_8bit": True,
        },
        "data": {
            "preprocessed_data_root": preproc_root,
            "num_dataloader_workers": 2,
        },
        "validation": {
            "prompts": [f"{trigger_word} {validation_prompt}".strip()],
            "negative_prompt": "worst quality, inconsistent motion, blurry",
            "video_dims": [512, 512, 1],   # single frame — fast
            "frame_rate": 25.0,
            "seed": 42,
            "inference_steps": 20,
            "interval": None,             # no intermediate validation
            "videos_per_prompt": 1,
            "guidance_scale": 3.0,
            "stg_scale": 0.0,
            "generate_audio": False,
            "skip_initial_validation": True,
        },
        "checkpoints": {
            "interval": None,    # no intermediate checkpoints
            "keep_last_n": 1,
            "precision": "bfloat16",
        },
        "hub": {"push_to_hub": False},
        "wandb": {"enabled": False},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal"},
    }

    try:
        import yaml as _yaml
        cfg_path.write_text(_yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
    except ImportError:
        # Fallback: write YAML manually (simple enough structure)
        import json as _json
        # convert via json round-trip is not proper YAML but readable enough
        # Use a simple recursive serialiser
        cfg_path.write_text(_simple_yaml_dump(cfg))


def _simple_yaml_dump(obj, indent=0) -> str:
    """Minimal YAML serialiser (no external dep)."""
    lines = []
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                lines.append(f"{pad}{k}: null")
            elif isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_simple_yaml_dump(v, indent + 1))
            elif isinstance(v, bool):
                lines.append(f"{pad}{k}: {'true' if v else 'false'}")
            elif isinstance(v, str):
                # quote strings that might be misread
                lines.append(f"{pad}{k}: {v!r}")
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_simple_yaml_dump(item, indent + 1))
            elif isinstance(item, str):
                lines.append(f"{pad}- {item!r}")
            else:
                lines.append(f"{pad}- {item}")
    return "\n".join(lines)


def _stream_subprocess(cmd: list[str], cwd: str, env: dict, log_cb, done_event: threading.Event):
    """Run cmd, stream output line-by-line via log_cb, set done_event when finished."""
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
    done_event._returncode = proc.returncode
    done_event.set()


# ── Node ──────────────────────────────────────────────────────────────────────

class TrainingLTX23LoRA_pyPTV:
    """
    Training LTX-2.3 LoRA (pyPTV)
    ─────────────────────────────
    Downloads an image dataset from HuggingFace, preprocesses it with
    ltx-trainer's process_dataset.py, then trains a LoRA with train.py.
    Outputs the path to the final lora_weights.safetensors.
    """

    CATEGORY     = "pyPTV"
    FUNCTION     = "train"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_NODE  = True

    # Shared log buffer (node instance → JS polling)
    _log_buffers: dict[str, list[str]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── Dataset ──────────────────────────────────────────────────
                "hf_dataset_repo_id": ("STRING", {
                    "default": "username/my-character-dataset",
                    "multiline": False,
                    "tooltip": "HuggingFace dataset repo, e.g. 'myuser/char-photos'",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Your HF_TOKEN for private repos",
                }),
                "local_dataset_folder": ("STRING", {
                    "default": "/comfyui/datasets/my_character",
                    "multiline": False,
                    "tooltip": "Local folder — one per character",
                }),
                # ── Caption / trigger ────────────────────────────────────────
                "trigger_word": ("STRING", {
                    "default": "JSRv1rpd",
                    "multiline": False,
                }),
                "default_caption": ("STRING", {
                    "default": "woman in a dark-blue suit, ash-brown shoulder-length hair, grey-blue eyes",
                    "multiline": True,
                }),
                # ── Output ───────────────────────────────────────────────────
                "output_dir": ("STRING", {
                    "default": "/comfyui/output/lora_training",
                    "multiline": False,
                }),
                # ── Model paths ──────────────────────────────────────────────
                "model_checkpoint": ("STRING", {
                    "default": DEFAULT_MODEL_CKPT,
                    "multiline": False,
                }),
                "text_encoder_dir": ("STRING", {
                    "default": DEFAULT_TEXT_ENCODER,
                    "multiline": False,
                }),
                # ── Training params ──────────────────────────────────────────
                "resolution": ("STRING", {
                    "default": "1024x1024x1",
                    "multiline": False,
                    "tooltip": "WxHxFrames — use 1 frame for image-only dataset",
                }),
                "steps": ("INT", {
                    "default": 2000,
                    "min": 100,
                    "max": 20000,
                    "step": 100,
                }),
                "lora_rank": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 128,
                    "step": 4,
                }),
                "lora_alpha": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 128,
                    "step": 4,
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-6,
                    "max": 1e-2,
                    "step": 1e-6,
                    "round": False,
                }),
                "validation_prompt": ("STRING", {
                    "default": "woman standing in a park, natural light",
                    "multiline": False,
                    "tooltip": "Used only at the very end to generate one validation frame",
                }),
                # ── Misc ─────────────────────────────────────────────────────
                "skip_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip HF download if images already in local_dataset_folder",
                }),
                "skip_preprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip process_dataset.py if latents already cached",
                }),
            },
        }

    # ── internal log helpers ──────────────────────────────────────────────────

    def _log(self, run_id: str, msg: str):
        buf = TrainingLTX23LoRA_pyPTV._log_buffers.setdefault(run_id, [])
        buf.append(msg)
        print(msg)   # also visible in ComfyUI server console

    def _get_log_cb(self, run_id: str):
        return lambda msg: self._log(run_id, msg)

    # ── main entry ────────────────────────────────────────────────────────────

    def train(
        self,
        hf_dataset_repo_id: str,
        hf_token: str,
        local_dataset_folder: str,
        trigger_word: str,
        default_caption: str,
        output_dir: str,
        model_checkpoint: str,
        text_encoder_dir: str,
        resolution: str,
        steps: int,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        validation_prompt: str,
        skip_download: bool,
        skip_preprocess: bool,
    ):
        import uuid
        run_id = uuid.uuid4().hex[:8]
        log = self._get_log_cb(run_id)

        log(f"[pyPTV] ══ Training LTX-2.3 LoRA  run={run_id} ══")

        local_dir   = Path(local_dataset_folder)
        out_dir     = Path(output_dir)
        preproc_dir = local_dir / ".precomputed"
        cfg_path    = local_dir / "train_config.yaml"

        # ── 1. Ensure ltx-trainer is available ────────────────────────────────
        try:
            _ensure_ltx2_repo(log)
        except Exception as e:
            log(f"[pyPTV] ERROR: could not set up LTX-2 repo: {e}")
            raise

        # ── 2. Download dataset ───────────────────────────────────────────────
        if not skip_download:
            try:
                _download_hf_dataset(hf_dataset_repo_id, hf_token, local_dir, log)
            except Exception as e:
                log(f"[pyPTV] ERROR downloading dataset: {e}")
                raise
        else:
            log(f"[pyPTV] Skipping download — using existing images in {local_dir}")

        # ── 3. Build dataset.json ─────────────────────────────────────────────
        log("[pyPTV] Building dataset.json …")
        dataset_json = _build_dataset_json(local_dir, default_caption, trigger_word)
        log(f"[pyPTV] dataset.json → {dataset_json}")

        # ── 4. Preprocess (compute latents + text embeddings) ─────────────────
        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"]                 = hf_token
            env["HUGGING_FACE_HUB_TOKEN"]   = hf_token

        if not skip_preprocess:
            log("[pyPTV] ── Preprocessing dataset (computing latents) ──")
            preproc_cmd = [
                "uv", "run", "python", str(PREPROC_SCRIPT),
                str(dataset_json),
                "--resolution-buckets", resolution,
                "--model-path",         model_checkpoint,
                "--text-encoder-path",  text_encoder_dir,
                "--lora-trigger",       trigger_word,
            ]
            log(f"[pyPTV] CMD: {' '.join(preproc_cmd)}")
            done = threading.Event()
            done._returncode = -1
            t = threading.Thread(
                target=_stream_subprocess,
                args=(preproc_cmd, str(TRAINER_PKG), env, log, done),
                daemon=True,
            )
            t.start()
            t.join()
            if done._returncode != 0:
                raise RuntimeError(f"process_dataset.py exited with code {done._returncode}")
            log("[pyPTV] Preprocessing complete ✓")
        else:
            log(f"[pyPTV] Skipping preprocessing — using cached latents in {preproc_dir}")

        # ── 5. Write train config ─────────────────────────────────────────────
        log("[pyPTV] Writing train_config.yaml …")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_train_config(
            cfg_path        = cfg_path,
            model_ckpt      = model_checkpoint,
            text_encoder    = text_encoder_dir,
            preproc_root    = str(preproc_dir),
            output_dir      = str(out_dir),
            trigger_word    = trigger_word,
            lora_rank       = lora_rank,
            lora_alpha      = lora_alpha,
            lr              = learning_rate,
            steps           = steps,
            resolution      = resolution,
            validation_prompt = validation_prompt,
        )
        log(f"[pyPTV] Config written → {cfg_path}")

        # ── 6. Run training ───────────────────────────────────────────────────
        log("[pyPTV] ── Starting training ──")
        train_cmd = [
            "uv", "run", "python", str(TRAIN_SCRIPT),
            str(cfg_path),
        ]
        log(f"[pyPTV] CMD: {' '.join(train_cmd)}")

        done2 = threading.Event()
        done2._returncode = -1
        t2 = threading.Thread(
            target=_stream_subprocess,
            args=(train_cmd, str(TRAINER_PKG), env, log, done2),
            daemon=True,
        )
        t2.start()
        t2.join()

        if done2._returncode != 0:
            raise RuntimeError(f"train.py exited with code {done2._returncode}")

        # ── 7. Locate output LoRA ─────────────────────────────────────────────
        lora_path = out_dir / "lora_weights.safetensors"
        if not lora_path.exists():
            # Search recursively in case ltx-trainer puts it in a subfolder
            candidates = list(out_dir.rglob("lora_weights.safetensors"))
            if candidates:
                lora_path = candidates[0]
            else:
                raise RuntimeError(
                    f"Training finished but lora_weights.safetensors not found in {out_dir}"
                )

        log(f"[pyPTV] ✓ LoRA saved → {lora_path}")
        return (str(lora_path),)


# ── Log polling node ──────────────────────────────────────────────────────────

class TrainingLogViewer_pyPTV:
    """
    Training Log Viewer (pyPTV)
    ───────────────────────────
    Displays the live training log from TrainingLTX23LoRA_pyPTV.
    Connect lora_path output → lora_path input here to auto-refresh.
    """

    CATEGORY     = "pyPTV"
    FUNCTION     = "show_log"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_text",)
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"forceInput": True}),
            }
        }

    def show_log(self, lora_path: str):
        # Collect all log lines from every run
        all_lines = []
        for buf in TrainingLTX23LoRA_pyPTV._log_buffers.values():
            all_lines.extend(buf)
        text = "\n".join(all_lines[-300:])   # last 300 lines
        return {"ui": {"text": text}, "result": (text,)}


# ── Mappings ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "TrainingLTX23LoRA_pyPTV":   TrainingLTX23LoRA_pyPTV,
    "TrainingLogViewer_pyPTV":   TrainingLogViewer_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainingLTX23LoRA_pyPTV":   "Training LTX-2.3 LoRA (pyPTV)",
    "TrainingLogViewer_pyPTV":   "Training Log Viewer (pyPTV)",
}
