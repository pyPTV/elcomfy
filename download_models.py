import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Цвета для терминала ───────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def log(msg):    print(f"{GREEN}[✓]{RESET} {msg}", flush=True)
def warn(msg):   print(f"{YELLOW}[!]{RESET} {msg}", flush=True)
def error(msg):  print(f"{RED}[✗]{RESET} {msg}", flush=True)
def info(msg):   print(f"{CYAN}[→]{RESET} {msg}", flush=True)
def header(msg): print(f"\n{'─'*55}\n  {msg}\n{'─'*55}", flush=True)


def download_hf_file(repo_id: str, filename: str, dest_path: str, token: str = None):
    """Скачивает файл с HuggingFace."""
    dest = Path(dest_path)

    info(f"Файл:       {filename}")
    info(f"Репо:       {repo_id}")
    info(f"Назначение: {dest}")

    if dest.exists() and dest.stat().st_size > 0:
        size_gb = dest.stat().st_size / 1024**3
        warn(f"Уже существует, пропускаем: {dest.name} ({size_gb:.2f} GB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(f"/tmp/hf_{dest.stem}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        log(f"Скачиваю...")
        t0 = time.time()
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(tmp_dir),
            token=token,
        )
        shutil.move(str(downloaded), str(dest))
        elapsed = time.time() - t0
        size_gb = dest.stat().st_size / 1024**3
        speed = size_gb / elapsed * 1024 if elapsed > 0 else 0
        log(f"Готово: {dest.name}  ({size_gb:.2f} GB, {elapsed:.0f}s, ~{speed:.0f} MB/s)")

    except Exception as e:
        error(f"Ошибка при скачивании {filename}: {e}")
        # НЕ делаем raise — продолжаем скрипт
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_hf_repo(repo_id: str, dest_dir: str, token: str = None):
    """Скачивает весь репозиторий с HuggingFace."""
    dest = Path(dest_dir)

    info(f"Репо:       {repo_id}")
    info(f"Назначение: {dest}")

    dest.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/hf_repo_download")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cmd = ["hf", "download", repo_id, "--local-dir", str(tmp_dir)]
    if token:
        cmd += ["--token", token]

    log(f"Скачиваю репозиторий {repo_id} ...")
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        error(f"hf download упал с кодом {result.returncode} для {repo_id}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return  # НЕ raise — продолжаем

    moved = 0
    for item in tmp_dir.iterdir():
        target = dest / item.name
        if target.exists():
            warn(f"  Уже есть, пропускаем: {item.name}")
            continue
        shutil.move(str(item), str(target))
        log(f"  Перемещено: {item.name}")
        moved += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.time() - t0
    log(f"Репо {repo_id} — {moved} файлов за {elapsed:.0f}s → {dest}")


def install_custom_node(repo_url: str, node_dir: str):
    """Клонирует custom node и устанавливает зависимости."""
    dest = Path(node_dir)
    name = dest.name

    info(f"Нода:   {name}")
    info(f"URL:    {repo_url}")
    info(f"Путь:   {dest}")

    if dest.exists():
        warn(f"Уже установлен: {name}")
        return

    log(f"Клонирую {name} ...")
    t0 = time.time()
    result = subprocess.run(
        ["git", "clone", repo_url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        error(f"git clone упал для {name}:")
        error(f"  stdout: {result.stdout.strip()}")
        error(f"  stderr: {result.stderr.strip()}")
        return

    elapsed = time.time() - t0
    log(f"Клонировано: {name} за {elapsed:.1f}s")

    # Считаем файлы
    file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
    info(f"Файлов в репо: {file_count}")

    req_file = dest / "requirements.txt"
    if req_file.exists():
        log(f"Устанавливаю зависимости {name} ...")
        pip_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            capture_output=True, text=True
        )
        if pip_result.returncode == 0:
            log(f"Зависимости {name} установлены")
        else:
            error(f"pip install упал для {name}:")
            error(f"  {pip_result.stderr[-500:]}")
    else:
        info(f"requirements.txt не найден для {name}, пропускаем pip")

    log(f"✅ Custom node готов: {name}")


def main():
    token = os.getenv("HF_TOKEN")

    print(f"\n{'═'*55}")
    print("  ComfyUI Setup & Model Downloader")
    print(f"{'═'*55}")

    if not token:
        warn("HF_TOKEN не найден в переменных окружения.")
        warn("Задай его в настройках Pod: Environment Variables → HF_TOKEN")
    else:
        log(f"HF_TOKEN найден: {token[:8]}...")

    # ════════════════════════════════════════════════════════
    # СНАЧАЛА — все custom nodes (пока сеть свободна)
    # ════════════════════════════════════════════════════════

    header("STEP 1/2  Custom Nodes")

    install_custom_node(
        repo_url = "https://github.com/gseth/ControlAltAI-Nodes.git",
        node_dir = "/comfyui/custom_nodes/ControlAltAI-Nodes",
    )
    install_custom_node(
        repo_url = "https://github.com/pyPTV/elcomfy.git",
        node_dir = "/comfyui/custom_nodes/elcomfy",
    )
    install_custom_node(
        repo_url = "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
        node_dir = "/comfyui/custom_nodes/ComfyUI-LTXVideo",
    )
    install_custom_node(
        repo_url = "https://github.com/evanspearman/ComfyMath.git",
        node_dir = "/comfyui/custom_nodes/ComfyMath",
    )
    install_custom_node(
        repo_url = "https://github.com/ClownsharkBatwing/RES4LYF.git",
        node_dir = "/comfyui/custom_nodes/RES4LYF",
    )
    install_custom_node(
        repo_url = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git",
        node_dir = "/comfyui/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler",
    )
    install_custom_node(
        repo_url = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        node_dir = "/comfyui/custom_nodes/ComfyUI-VideoHelperSuite",
    )
    install_custom_node(
        repo_url = "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git",
        node_dir = "/comfyui/custom_nodes/ComfyUI-Frame-Interpolation",
    )

    # ════════════════════════════════════════════════════════
    # ПОТОМ — все модели (тяжёлые файлы)
    # ════════════════════════════════════════════════════════

    header("STEP 2/2  Models")

    header("FLUX.2-dev (~64 GB)")
    download_hf_file(
        repo_id  = "black-forest-labs/FLUX.2-dev",
        filename = "flux2-dev.safetensors",
        dest_path= "/comfyui/models/diffusion_models/flux2-dev.safetensors",
        token    = token,
    )

    header("FLUX.2 VAE (~336 MB)")
    download_hf_file(
        repo_id  = "Comfy-Org/flux2-dev",
        filename = "split_files/vae/flux2-vae.safetensors",
        dest_path= "/comfyui/models/vae/flux2-vae.safetensors",
        token    = token,
    )

    header("Mistral CLIP fp8 (~2.5 GB)")
    download_hf_file(
        repo_id  = "Comfy-Org/flux2-dev",
        filename = "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
        dest_path= "/comfyui/models/text_encoders/mistral_3_small_flux2_fp8.safetensors",
        token    = token,
    )

    header("Jane Soren LoRA repo")
    download_hf_repo(
        repo_id  = "avidscreator/jane-soren-flux2",
        dest_dir = "/comfyui/models/loras",
        token    = token,
    )

    header("LTX-2 Text Encoder — Gemma 3 12B (~24 GB)")
    download_hf_file(
        repo_id  = "Comfy-Org/ltx-2",
        filename = "split_files/text_encoders/gemma_3_12B_it.safetensors",
        dest_path= "/comfyui/models/text_encoders/gemma_3_12B_it.safetensors",
        token    = token,
    )

    header("LTX-2.3 22B dev (~44 GB)")
    download_hf_file(
        repo_id  = "Lightricks/LTX-2.3",
        filename = "ltx-2.3-22b-dev.safetensors",
        dest_path= "/comfyui/models/checkpoints/ltx-2.3-22b-dev.safetensors",
        token    = token,
    )

    header("LTX-2.3 distilled LoRA (~1.5 GB)")
    download_hf_file(
        repo_id  = "Lightricks/LTX-2.3",
        filename = "ltx-2.3-22b-distilled-lora-384.safetensors",
        dest_path= "/comfyui/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors",
        token    = token,
    )

    header("LTX-2.3 spatial upscaler x2 (~1 GB)")
    download_hf_file(
        repo_id  = "Lightricks/LTX-2.3",
        filename = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        dest_path= "/comfyui/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        token    = token,
    )

    header("LTX-2.3 temporal upscaler x2 (~1 GB)")
    download_hf_file(
        repo_id  = "Lightricks/LTX-2.3",
        filename = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        dest_path= "/comfyui/models/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        token    = token,
    )

    header("SeedVR2 model (~15 GB)")
    download_hf_file(
        repo_id  = "numz/SeedVR2_comfyUI",
        filename = "seedvr2_ema_7b_fp16.safetensors",
        dest_path= "/comfyui/models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors",
        token    = token,
    )

    header("SeedVR2 VAE (~470 MB)")
    download_hf_file(
        repo_id  = "numz/SeedVR2_comfyUI",
        filename = "ema_vae_fp16.safetensors",
        dest_path= "/comfyui/models/SEEDVR2/ema_vae_fp16.safetensors",
        token    = token,
    )

    header("RIFE 4.9 frame interpolation (~20 MB)")
    download_hf_file(
        repo_id  = "Isi99999/Frame_Interpolation_Models",
        filename = "rife49.pth",
        dest_path= "/comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth",
        token    = token,
    )

    print(f"\n{'═'*55}")
    log("Всё готово!")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
