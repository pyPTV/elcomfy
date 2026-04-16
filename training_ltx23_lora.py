"""
pyPTV — Training LTX-2.3 LoRA node
Downloads dataset from HuggingFace, trains LoRA, outputs final .safetensors
"""

import os
import sys
import json
import time
import shutil
import subprocess
import threading
import tempfile
from pathlib import Path


# ─── helpers ──────────────────────────────────────────────────────────────────

def _write_config(config_path: str, cfg: dict) -> None:
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def _build_aitk_config(
    dataset_local_dir: str,
    output_dir: str,
    trigger_word: str,
    default_caption: str,
    steps: int,
    lr: float,
    lora_rank: int,
    lora_alpha: int,
    model_path: str,
    resolution: int,
) -> dict:
    return {
        "job": "extension",
        "config": {
            "name": "ltx23_lora_pyptvw",
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": output_dir,
                    "sqlite_db_path": os.path.join(output_dir, "aitk_db.db"),
                    "device": "cuda",
                    "trigger_word": trigger_word,
                    "performance_log_every": 10,
                    "network": {
                        "type": "lora",
                        "linear": lora_rank,
                        "linear_alpha": lora_alpha,
                        "conv": 16,
                        "conv_alpha": 8,
                        "lokr_full_rank": False,
                        "lokr_factor": -1,
                        "network_kwargs": {"ignore_if_contains": []},
                    },
                    "save": {
                        "dtype": "bf16",
                        "save_every": steps + 9999,   # no intermediate saves
                        "max_step_saves_to_keep": 1,
                        "save_format": "safetensors",
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_local_dir,
                            "mask_path": None,
                            "mask_min_value": 0.1,
                            "default_caption": default_caption,
                            "caption_dropout_rate": 0.1,
                            "cache_latents_to_disk": True,
                            "is_reg": False,
                            "network_weight": 1,
                            "resolution": [resolution],
                            "controls": [],
                            "shrink_video_to_frames": True,
                            "num_frames": 1,
                            "flip_x": False,
                            "flip_y": False,
                            "num_repeats": 5,
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "bypass_guidance_embedding": False,
                        "steps": steps,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": True,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "timestep_type": "weighted",
                        "content_or_style": "balanced",
                        "optimizer_params": {"weight_decay": 0.01},
                        "unload_text_encoder": False,
                        "cache_text_embeddings": False,
                        "lr": lr,
                        "ema_config": {"use_ema": False, "ema_decay": 0.99},
                        "skip_first_sample": False,
                        "force_first_sample": False,
                        "disable_sampling": True,
                        "dtype": "bf16",
                        "diff_output_preservation": False,
                        "diff_output_preservation_multiplier": 1,
                        "diff_output_preservation_class": "person",
                        "switch_boundary_every": 1,
                        "loss_type": "mse",
                    },
                    "logging": {
                        "log_every": 1,
                        "use_ui_logger": False,
                    },
                    "model": {
                        "name_or_path": model_path,
                        "quantize": False,
                        "qtype": "qfloat8",
                        "quantize_te": False,
                        "qtype_te": "bf16",
                        "arch": "ltx2.3",
                        "low_vram": True,
                        "dtype": "bf16",
                        "model_kwargs": {"match_target_res": False},
                        "layer_offloading": False,
                        "layer_offloading_text_encoder_percent": 1,
                        "layer_offloading_transformer_percent": 1,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": steps + 9999,  # no intermediate samples
                        "width": resolution,
                        "height": resolution,
                        "samples": [],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 3,
                        "sample_steps": 25,
                        "num_frames": 1,
                        "fps": 24,
                    },
                }
            ],
        },
        "meta": {"name": "ltx23_lora_pyptvw", "version": "1.0"},
    }


# ─── node ─────────────────────────────────────────────────────────────────────

class TrainingLTX23LoRA_pyPTV:
    """
    Downloads a HuggingFace dataset (images only, no .txt required),
    trains an LTX-2.3 LoRA via ai-toolkit, streams progress to log,
    and returns the path to the final .safetensors file.
    """

    # Shared log buffer accessible from the log node
    _log_buffers: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_dataset_repo": (
                    "STRING",
                    {
                        "default": "username/my-dataset",
                        "multiline": False,
                        "tooltip": "HuggingFace repo id, e.g. myuser/portraits",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Your HuggingFace access token (HF_TOKEN)",
                    },
                ),
                "trigger_word": (
                    "STRING",
                    {"default": "JSRv1rpd", "multiline": False},
                ),
                "default_caption": (
                    "STRING",
                    {
                        "default": "JSRv1rpd person",
                        "multiline": True,
                        "tooltip": "Caption applied to every image (no .txt files needed)",
                    },
                ),
                "model_path": (
                    "STRING",
                    {
                        "default": "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
                        "multiline": False,
                        "tooltip": "Local path or HF path to the LTX-2.3 model checkpoint",
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "/workspace/lora_output",
                        "multiline": False,
                        "tooltip": "Directory where the final LoRA will be saved",
                    },
                ),
                "steps": ("INT", {"default": 2000, "min": 100, "max": 20000, "step": 100}),
                "learning_rate": (
                    "FLOAT",
                    {"default": 0.0001, "min": 1e-6, "max": 0.01, "step": 1e-5},
                ),
                "lora_rank": ("INT", {"default": 32, "min": 4, "max": 128, "step": 4}),
                "lora_alpha": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4}),
                "resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "aitk_path": (
                    "STRING",
                    {
                        "default": "/app/ai-toolkit",
                        "multiline": False,
                        "tooltip": "Path to ai-toolkit repo root (where run.py lives)",
                    },
                ),
            }
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "log")
    FUNCTION = "train"
    OUTPUT_NODE = True

    # ── main entry ────────────────────────────────────────────────────────────

    def train(
        self,
        hf_dataset_repo: str,
        hf_token: str,
        trigger_word: str,
        default_caption: str,
        model_path: str,
        output_dir: str,
        steps: int,
        learning_rate: float,
        lora_rank: int,
        lora_alpha: int,
        resolution: int,
        aitk_path: str,
    ):
        session_id = str(int(time.time()))
        log_lines: list[str] = []
        TrainingLTX23LoRA_pyPTV._log_buffers[session_id] = log_lines

        def log(msg: str):
            log_lines.append(msg)
            print(f"[pyPTV-Train] {msg}", flush=True)

        try:
            # ── 0. sanity checks ──────────────────────────────────────────────
            aitk_run = os.path.join(aitk_path, "run.py")
            if not os.path.isfile(aitk_run):
                raise FileNotFoundError(
                    f"ai-toolkit run.py not found at {aitk_run}. "
                    "Set 'aitk_path' to the root of the ai-toolkit repo."
                )

            # ── 1. install pyyaml if needed ───────────────────────────────────
            try:
                import yaml  # noqa: F401
            except ImportError:
                log("Installing pyyaml …")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
                import yaml  # noqa: F401

            # ── 2. dataset directory (one per repo) ───────────────────────────
            safe_name = hf_dataset_repo.replace("/", "__")
            datasets_root = os.path.join(aitk_path, "datasets")
            dataset_local_dir = os.path.join(datasets_root, safe_name)
            os.makedirs(dataset_local_dir, exist_ok=True)

            # ── 3. download images from HuggingFace ───────────────────────────
            log(f"Downloading dataset {hf_dataset_repo} → {dataset_local_dir} …")
            self._download_hf_dataset(
                repo_id=hf_dataset_repo,
                local_dir=dataset_local_dir,
                hf_token=hf_token,
                log=log,
            )

            img_count = len(
                [
                    f
                    for f in os.listdir(dataset_local_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ]
            )
            log(f"Dataset ready: {img_count} images in {dataset_local_dir}")
            if img_count == 0:
                raise RuntimeError(
                    "No images found in dataset after download. "
                    "Check the repo id and HF_TOKEN permissions."
                )

            # ── 4. build config ───────────────────────────────────────────────
            os.makedirs(output_dir, exist_ok=True)
            cfg = _build_aitk_config(
                dataset_local_dir=dataset_local_dir,
                output_dir=output_dir,
                trigger_word=trigger_word,
                default_caption=default_caption,
                steps=steps,
                lr=learning_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                model_path=model_path,
                resolution=resolution,
            )
            config_path = os.path.join(output_dir, "train_config.yaml")
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            log(f"Config written → {config_path}")

            # ── 5. run ai-toolkit ─────────────────────────────────────────────
            log(f"Starting training: {steps} steps …")
            env = os.environ.copy()
            if hf_token:
                env["HF_TOKEN"] = hf_token
                env["HUGGING_FACE_HUB_TOKEN"] = hf_token

            lora_path = self._run_training(
                aitk_path=aitk_path,
                config_path=config_path,
                output_dir=output_dir,
                steps=steps,
                env=env,
                log=log,
            )

            log(f"✅ Training complete! LoRA saved → {lora_path}")

        except Exception as exc:
            log(f"❌ ERROR: {exc}")
            lora_path = ""

        full_log = "\n".join(log_lines)
        TrainingLTX23LoRA_pyPTV._log_buffers.pop(session_id, None)
        return (lora_path, full_log)

    # ── private helpers ───────────────────────────────────────────────────────

    def _download_hf_dataset(
        self,
        repo_id: str,
        local_dir: str,
        hf_token: str,
        log,
    ):
        """Download only image files from a HuggingFace dataset repo."""
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
        except ImportError:
            log("Installing huggingface_hub …")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"]
            )
            from huggingface_hub import list_repo_files, hf_hub_download

        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        token = hf_token if hf_token else None

        all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
        img_files = [f for f in all_files if Path(f).suffix.lower() in image_exts]

        log(f"Found {len(img_files)} image files in repo.")

        for i, fname in enumerate(img_files, 1):
            dest = os.path.join(local_dir, os.path.basename(fname))
            if os.path.exists(dest):
                log(f"  [{i}/{len(img_files)}] skip (cached): {fname}")
                continue
            log(f"  [{i}/{len(img_files)}] downloading: {fname}")
            src = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                token=token,
                local_dir=local_dir,
            )
            # hf_hub_download may put file in subdir — flatten to local_dir
            if os.path.dirname(src) != local_dir:
                shutil.move(src, dest)

    def _run_training(
        self,
        aitk_path: str,
        config_path: str,
        output_dir: str,
        steps: int,
        env: dict,
        log,
    ) -> str:
        """
        Runs `python run.py <config>` inside ai-toolkit and streams stdout/stderr.
        Returns path to final .safetensors lora file.
        """
        cmd = [sys.executable, "run.py", config_path]

        proc = subprocess.Popen(
            cmd,
            cwd=aitk_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ai-toolkit exited with code {proc.returncode}")

        # Locate the final lora file
        lora_path = self._find_lora_file(output_dir)
        return lora_path

    def _find_lora_file(self, output_dir: str) -> str:
        """Walk output_dir looking for the final lora .safetensors."""
        candidates = sorted(
            Path(output_dir).rglob("*.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Prefer files with 'lora' in name
        for p in candidates:
            if "lora" in p.name.lower():
                return str(p)
        if candidates:
            return str(candidates[0])
        return ""


# ─── companion log node ───────────────────────────────────────────────────────

class TrainingLog_pyPTV:
    """
    Simple passthrough node: displays the training log string in the UI.
    Connect 'log' output from TrainingLTX23LoRA_pyPTV here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "log": ("STRING", {"forceInput": True}),
            }
        }

    CATEGORY = "pyPTV"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "show"
    OUTPUT_NODE = True

    def show(self, log: str):
        return {"ui": {"text": [log]}, "result": (log,)}


# ─── registrations ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "TrainingLTX23LoRA_pyPTV": TrainingLTX23LoRA_pyPTV,
    "TrainingLog_pyPTV": TrainingLog_pyPTV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainingLTX23LoRA_pyPTV": "Training LTX-2.3 LoRA (pyPTV)",
    "TrainingLog_pyPTV": "Training Log (pyPTV)",
}
