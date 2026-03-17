"""
ComfyUI Custom Node: ElevenLabs Voice Changer via fal.ai
AUDIO → fal.ai (fal-ai/elevenlabs/voice-changer) → AUDIO
"""

import io
import wave
import base64
import requests
import torch


# ─── audio helpers (same as direct node) ─────────────────────────────────────

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


def wav_bytes_to_data_uri(wav_bytes: bytes) -> str:
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


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
                    "tooltip": "0 = случайный seed. Любое другое значение — детерминированный результат.",
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

        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]

        print(f"[ElevenLabsFal] Input: shape={waveform.shape}, sr={sample_rate}")

        # encode audio as base64 data URI (no file upload step needed)
        wav_bytes   = tensor_to_wav_bytes(waveform, sample_rate)
        audio_url   = wav_bytes_to_data_uri(wav_bytes)

        print(f"[ElevenLabsFal] Sending {len(wav_bytes)/1024:.1f} KB → fal.ai…")

        payload = {
            "audio_url":               audio_url,
            "voice":                   voice,
            "remove_background_noise": remove_background_noise,
            "output_format":           output_format,
        }
        if seed != 0:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type":  "application/json",
        }

        response = requests.post(
            "https://fal.run/fal-ai/elevenlabs/voice-changer",
            json=payload,
            headers=headers,
            timeout=300,
        )

        if response.status_code != 200:
            raise RuntimeError(f"fal.ai API error {response.status_code}: {response.text}")

        result = response.json()
        # result["audio"]["url"] — URL готового файла
        audio_result_url = result["audio"]["url"]
        print(f"[ElevenLabsFal] Result URL: {audio_result_url}")

        # download the result audio
        dl = requests.get(audio_result_url, timeout=120)
        dl.raise_for_status()

        # detect format and decode
        content_type = dl.headers.get("content-type", "")
        if output_format.startswith("pcm_"):
            # raw PCM — wrap into WAV manually
            sr = int(output_format.split("_")[1])
            wav_result = _pcm_to_wav_bytes(dl.content, sr)
        else:
            wav_result = mp3_bytes_to_wav_bytes(dl.content)

        out_waveform, out_sr = wav_bytes_to_tensor(wav_result)
        print(f"[ElevenLabsFal] Output: shape={out_waveform.shape}, sr={out_sr}")

        return ({"waveform": out_waveform, "sample_rate": out_sr},)


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw PCM s16le bytes into a proper WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


NODE_CLASS_MAPPINGS        = {"ElevenLabsFalVoiceChanger": ElevenLabsFalVoiceChangerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ElevenLabsFalVoiceChanger": "🎙️ ElevenLabs Voice Changer (fal.ai)"}
