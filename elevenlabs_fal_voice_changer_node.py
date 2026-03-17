"""
ComfyUI Custom Node: ElevenLabs Voice Changer via fal.ai
AUDIO -> fal.ai (fal-ai/elevenlabs/voice-changer) -> AUDIO
Requires: pip install fal-client
"""

import io
import os
import wave
import tempfile
import subprocess
import requests
import torch


# ─── форматы ─────────────────────────────────────────────────────────────────

OUTPUT_FORMATS = [
    "mp3_22050_32", "mp3_44100_32", "mp3_44100_64", "mp3_44100_96", "mp3_44100_128", "mp3_44100_192",
    "pcm_8000", "pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100", "pcm_48000",
    "opus_48000_32", "opus_48000_64", "opus_48000_96", "opus_48000_128", "opus_48000_192",
    "ulaw_8000", "alaw_8000",
]


# ─── audio helpers ───────────────────────────────────────────────────────────

def tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
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
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_ch = wf.getnchannels()
        sr   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
    samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    return samples.reshape(n_ch, -1).unsqueeze(0), sr


def decode_audio_response(data: bytes, output_format: str) -> bytes:
    fmt = output_format.lower()

    if fmt.startswith("pcm_"):
        sr = int(fmt.split("_")[1])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data)
        return buf.getvalue()

    if fmt.startswith("mp3"):
        in_fmt = "mp3"
    elif fmt.startswith("opus"):
        in_fmt = "ogg"
    elif fmt.startswith("ulaw"):
        in_fmt = "mulaw"
    elif fmt.startswith("alaw"):
        in_fmt = "alaw"
    else:
        in_fmt = "mp3"

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", in_fmt, "-i", "pipe:0",
         "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"],
        input=data, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{result.stderr.decode()}")
    return result.stdout


# ─── node ────────────────────────────────────────────────────────────────────

class ElevenLabsFalVoiceChangerNode:

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
                "output_format": (OUTPUT_FORMATS, {"default": "pcm_44100"}),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": "0 = случайный. Число = повторяемый результат.",
                }),
            },
        }

    def process(self, audio, api_key, voice,
                output_format="pcm_44100",
                remove_background_noise=False, seed=0):

        try:
            import fal_client
        except ImportError:
            raise RuntimeError("fal-client не установлен: .venv\\Scripts\\python.exe -m pip install fal-client")

        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]
        print(f"[ElevenLabsFal] Input: shape={waveform.shape}, sr={sample_rate}, format={output_format}")

        wav_bytes = tensor_to_wav_bytes(waveform, sample_rate)
        print(f"[ElevenLabsFal] WAV size: {len(wav_bytes)/1024:.1f} KB")

        os.environ["FAL_KEY"] = api_key

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav_bytes)
            tmp.close()
            print("[ElevenLabsFal] Uploading via fal_client.upload_file()…")
            audio_url = fal_client.upload_file(tmp.name)
            print(f"[ElevenLabsFal] Uploaded -> {audio_url}")
        finally:
            os.unlink(tmp.name)

        arguments = {
            "audio_url":               audio_url,
            "voice":                   voice,
            "remove_background_noise": remove_background_noise,
            "output_format":           output_format,
        }
        if seed != 0:
            arguments["seed"] = seed

        print(f"[ElevenLabsFal] Calling API: voice={voice}")
        result = fal_client.subscribe(
            "fal-ai/elevenlabs/voice-changer",
            arguments=arguments,
            with_logs=True,
            on_queue_update=lambda u: [
                print(f"[ElevenLabsFal] {l['message']}") for l in getattr(u, "logs", [])
            ],
        )
        print(f"[ElevenLabsFal] Result: {result}")

        dl = requests.get(result["audio"]["url"], timeout=120)
        dl.raise_for_status()
        print(f"[ElevenLabsFal] Downloaded {len(dl.content)/1024:.1f} KB, decoding…")

        wav_result   = decode_audio_response(dl.content, output_format)
        out_waveform, out_sr = wav_bytes_to_tensor(wav_result)
        print(f"[ElevenLabsFal] Output: shape={out_waveform.shape}, sr={out_sr}")

        return ({"waveform": out_waveform, "sample_rate": out_sr},)


NODE_CLASS_MAPPINGS        = {"ElevenLabsFalVoiceChanger": ElevenLabsFalVoiceChangerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ElevenLabsFalVoiceChanger": "🎙️ ElevenLabs Voice Changer (fal.ai)"}
