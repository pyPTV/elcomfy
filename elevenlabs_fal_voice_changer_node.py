"""
ComfyUI Custom Node: ElevenLabs Voice Changer via fal.ai
AUDIO → fal.ai (fal-ai/elevenlabs/voice-changer) → AUDIO
Requires: pip install fal-client
"""

import io
import os
import wave
import tempfile
import requests
import torch


# ─── audio helpers ───────────────────────────────────────────────────────────

def tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    samples = waveform[0]
    samples_np = (samples.clamp(-1.0, 1.0).cpu().numpy() * 32767).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_np.tobytes())
    return buf.getvalue()


def wav_bytes_to_tensor(wav_bytes: bytes):
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)
    samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    samples = samples.reshape(n_channels, -1)
    return samples.unsqueeze(0), sample_rate


def mp3_bytes_to_wav_bytes(mp3_bytes: bytes) -> bytes:
    import torchaudio
    buf_in = io.BytesIO(mp3_bytes)
    waveform, sr = torchaudio.load(buf_in, format="mp3")
    buf_out = io.BytesIO()
    torchaudio.save(buf_out, waveform, sr, format="wav")
    return buf_out.getvalue()


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ─── node ────────────────────────────────────────────────────────────────────

class ElevenLabsFalVoiceChangerNode:
    """AUDIO → fal.ai ElevenLabs Voice Changer → AUDIO"""

    CATEGORY = "audio/elevenlabs"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":   ("AUDIO",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "fal.ai API key (FAL_KEY)"}),
                "voice":   ("STRING", {"default": "Rachel", "multiline": False, "placeholder": "Voice name or voice_id"}),
            },
            "optional": {
                "remove_background_noise": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": "0 = случайный seed. Число = детерминированный результат.",
                }),
                "output_format": (
                    [
                        "mp3_44100_128", "mp3_44100_192", "mp3_44100_96",
                        "mp3_44100_64",  "mp3_44100_32",  "mp3_22050_32",
                        "pcm_44100", "pcm_48000", "pcm_24000", "pcm_22050",
                        "pcm_16000", "pcm_8000",
                    ],
                    {"default": "mp3_44100_128"},
                ),
            },
        }

    def process(self, audio, api_key, voice,
                remove_background_noise=False, seed=0,
                output_format="mp3_44100_128"):

        try:
            import fal_client
        except ImportError:
            raise RuntimeError(
                "fal-client не установлен. Выполни: "
                "pip install fal-client  (в окружении ComfyUI)"
            )

        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]

        print(f"[ElevenLabsFal] Input: shape={waveform.shape}, sr={sample_rate}")

        wav_bytes = tensor_to_wav_bytes(waveform, sample_rate)
        print(f"[ElevenLabsFal] WAV size: {len(wav_bytes)/1024:.1f} KB")

        # записываем во временный файл — fal_client.upload_file() принимает путь
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav_bytes)
            tmp.close()

            # устанавливаем ключ в env — fal_client читает FAL_KEY
            os.environ["FAL_KEY"] = api_key

            print("[ElevenLabsFal] Uploading via fal_client.upload_file()…")
            audio_url = fal_client.upload_file(tmp.name)
            print(f"[ElevenLabsFal] Uploaded → {audio_url}")

        finally:
            os.unlink(tmp.name)

        # ── вызов API ───────────────────────────────────────────────────────
        arguments = {
            "audio_url":               audio_url,
            "voice":                   voice,
            "remove_background_noise": remove_background_noise,
            "output_format":           output_format,
        }
        if seed != 0:
            arguments["seed"] = seed

        print(f"[ElevenLabsFal] Calling fal-ai/elevenlabs/voice-changer  voice={voice}")

        result = fal_client.subscribe(
            "fal-ai/elevenlabs/voice-changer",
            arguments=arguments,
            with_logs=True,
            on_queue_update=lambda u: (
                [print(f"[ElevenLabsFal] {l['message']}") for l in getattr(u, 'logs', [])]
            ),
        )

        print(f"[ElevenLabsFal] Result: {result}")

        result_url = result["audio"]["url"]
        print(f"[ElevenLabsFal] Downloading from: {result_url}")

        dl = requests.get(result_url, timeout=120)
        dl.raise_for_status()

        if output_format.startswith("pcm_"):
            sr         = int(output_format.split("_")[1])
            wav_result = _pcm_to_wav_bytes(dl.content, sr)
        else:
            wav_result = mp3_bytes_to_wav_bytes(dl.content)

        out_waveform, out_sr = wav_bytes_to_tensor(wav_result)
        print(f"[ElevenLabsFal] Output: shape={out_waveform.shape}, sr={out_sr}")

        return ({"waveform": out_waveform, "sample_rate": out_sr},)


NODE_CLASS_MAPPINGS        = {"ElevenLabsFalVoiceChanger": ElevenLabsFalVoiceChangerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ElevenLabsFalVoiceChanger": "🎙️ ElevenLabs Voice Changer (fal.ai)"}
