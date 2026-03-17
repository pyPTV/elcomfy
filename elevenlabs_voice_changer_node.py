"""
ComfyUI Custom Node: ElevenLabs Voice Changer (Direct API)
AUDIO -> ElevenLabs Speech-to-Speech API -> AUDIO
"""

import io
import json
import wave
import subprocess
import requests
import torch


# ─── форматы ─────────────────────────────────────────────────────────────────

OUTPUT_FORMATS = [
    # MP3 (lossy)
    "mp3_22050_32", "mp3_24000_48",
    "mp3_44100_32", "mp3_44100_64", "mp3_44100_96", "mp3_44100_128", "mp3_44100_192",
    # PCM (lossless)
    "pcm_8000", "pcm_16000", "pcm_22050", "pcm_24000", "pcm_32000", "pcm_44100", "pcm_48000",
    # Opus (lossy, хорошее качество)
    "opus_48000_32", "opus_48000_64", "opus_48000_96", "opus_48000_128", "opus_48000_192",
    # Telephony
    "ulaw_8000", "alaw_8000",
]


# ─── audio helpers ───────────────────────────────────────────────────────────

def tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Tensor (float32) -> WAV PCM s16le bytes."""
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    samples_np = (waveform[0].clamp(-1.0, 1.0).cpu().numpy() * 32767).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_np.tobytes())
    return buf.getvalue()


def wav_bytes_to_tensor(wav_bytes: bytes):
    """WAV bytes -> (waveform tensor (1,ch,samples), sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_ch = wf.getnchannels()
        sr   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
    samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    return samples.reshape(n_ch, -1).unsqueeze(0), sr


def decode_audio_response(data: bytes, output_format: str) -> bytes:
    """
    Decode API response bytes -> WAV bytes.
    PCM formats are already raw samples — just wrap in WAV header.
    Everything else (mp3, opus, ulaw, alaw) goes through ffmpeg.
    """
    fmt = output_format.lower()

    if fmt.startswith("pcm_"):
        # raw PCM s16le — wrap into WAV
        sr = int(fmt.split("_")[1])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data)
        return buf.getvalue()

    # detect input format for ffmpeg
    if fmt.startswith("mp3"):
        in_fmt = "mp3"
    elif fmt.startswith("opus"):
        in_fmt = "ogg"   # opus in ogg container
    elif fmt.startswith("ulaw"):
        in_fmt = "mulaw"
    elif fmt.startswith("alaw"):
        in_fmt = "alaw"
    else:
        in_fmt = "mp3"   # fallback

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", in_fmt, "-i", "pipe:0",
         "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"],
        input=data, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{result.stderr.decode()}")
    return result.stdout


# ─── node ────────────────────────────────────────────────────────────────────

class ElevenLabsVoiceChangerNode:

    CATEGORY = "audio/elevenlabs"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":    ("AUDIO",),
                "api_key":  ("STRING", {"default": "", "multiline": False, "placeholder": "ElevenLabs API key"}),
                "voice_id": ("STRING", {"default": "", "multiline": False, "placeholder": "ElevenLabs voice ID"}),
            },
            "optional": {
                "model_id": (
                    ["eleven_multilingual_sts_v2", "eleven_english_sts_v2"],
                    {"default": "eleven_multilingual_sts_v2"},
                ),
                "output_format": (OUTPUT_FORMATS, {"default": "pcm_44100"}),
                "stability": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                    "tooltip": "Низкое = эмоциональнее, высокое = монотоннее.",
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                    "tooltip": "Схожесть с голосом. Слишком высокое + плохой исходник = артефакты.",
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                    "tooltip": "Усиление стиля. ElevenLabs рекомендует держать на 0.",
                }),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": "0 = случайный. Число = повторяемый результат.",
                }),
            },
        }

    def process(self, audio, api_key, voice_id,
                model_id="eleven_multilingual_sts_v2",
                output_format="pcm_44100",
                stability=0.5, similarity_boost=0.75, style=0.0,
                remove_background_noise=False, seed=0):

        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]
        print(f"[ElevenLabsVC] Input: shape={waveform.shape}, sr={sample_rate}, output_format={output_format}")

        wav_bytes = tensor_to_wav_bytes(waveform, sample_rate)
        print(f"[ElevenLabsVC] Sending {len(wav_bytes)/1024:.1f} KB WAV -> ElevenLabs…")

        data = {
            "model_id": model_id,
            "voice_settings": json.dumps({
                "stability":        stability,
                "similarity_boost": similarity_boost,
                "style":            style,
                "use_speaker_boost": True,
            }),
            "remove_background_noise": str(remove_background_noise).lower(),
        }
        if seed != 0:
            data["seed"] = seed

        response = requests.post(
            f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}",
            params={"output_format": output_format},
            headers={"xi-api-key": api_key},
            files={"audio": ("source.wav", wav_bytes, "audio/wav")},
            data=data,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs API error {response.status_code}: {response.text}")

        print(f"[ElevenLabsVC] Got {len(response.content)/1024:.1f} KB back, decoding…")

        wav_result = decode_audio_response(response.content, output_format)
        out_waveform, out_sr = wav_bytes_to_tensor(wav_result)
        print(f"[ElevenLabsVC] Output: shape={out_waveform.shape}, sr={out_sr}")

        return ({"waveform": out_waveform, "sample_rate": out_sr},)


NODE_CLASS_MAPPINGS        = {"ElevenLabsVoiceChanger": ElevenLabsVoiceChangerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ElevenLabsVoiceChanger": "🎙️ ElevenLabs Voice Changer"}
