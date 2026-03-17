"""
ComfyUI Custom Node: ElevenLabs Voice Changer
Accepts VHS_AUDIO (or ComfyUI AUDIO) → sends WAV to ElevenLabs STS API → returns AUDIO
Plug directly between VHS_LoadVideoFFmpeg and VHS_VideoCombine.
"""

import io
import json
import wave
import requests
import torch


# ─── audio helpers ───────────────────────────────────────────────────────────

def tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """
    Convert ComfyUI/VHS audio tensor → raw WAV bytes ready for upload.
    waveform shape: (batch, channels, samples)  or  (channels, samples)
    """
    # squeeze batch dim if present
    if waveform.dim() == 3:
        waveform = waveform[0]          # (channels, samples)

    # mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    samples = waveform[0]               # (samples,)

    # clamp & convert to int16
    samples_np = (samples.clamp(-1.0, 1.0).cpu().numpy() * 32767).astype("int16")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)              # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples_np.tobytes())
    return buf.getvalue()


def wav_bytes_to_tensor(wav_bytes: bytes):
    """
    Parse raw WAV bytes → (waveform tensor, sample_rate).
    Returns waveform shape: (1, channels, samples) — VHS_VideoCombine compatible.
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)

    samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    samples = samples.reshape(n_channels, -1)
    waveform = samples.unsqueeze(0)     # (1, channels, samples)
    return waveform, sample_rate


def mp3_to_wav_bytes(mp3_bytes: bytes) -> bytes:
    """Decode mp3 bytes → wav bytes via torchaudio (no ffmpeg needed)."""
    import torchaudio
    buf_in = io.BytesIO(mp3_bytes)
    waveform, sr = torchaudio.load(buf_in, format="mp3")
    buf_out = io.BytesIO()
    torchaudio.save(buf_out, waveform, sr, format="wav")
    return buf_out.getvalue()


# ─── API call ────────────────────────────────────────────────────────────────

def elevenlabs_sts(
    wav_bytes: bytes,
    api_key: str,
    voice_id: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
    remove_background_noise: bool,
    output_format: str,
) -> bytes:
    url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key}
    voice_settings = json.dumps({
        "stability": stability,
        "similarity_boost": similarity_boost,
        "style": style,
        "use_speaker_boost": True,
    })

    response = requests.post(
        url,
        params={"output_format": output_format},
        headers=headers,
        files={"audio": ("source.wav", wav_bytes, "audio/wav")},
        data={
            "model_id": model_id,
            "voice_settings": voice_settings,
            "remove_background_noise": str(remove_background_noise).lower(),
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API error {response.status_code}: {response.text}")

    return response.content


# ─── node ────────────────────────────────────────────────────────────────────

class ElevenLabsVoiceChangerNode:
    """
    AUDIO → ElevenLabs Speech-to-Speech → AUDIO

    Connect:
      VHS_LoadVideoFFmpeg  [audio] ──► this node ──► VHS_VideoCombine [audio]
    """

    CATEGORY = "audio/elevenlabs"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "your_elevenlabs_api_key",
                }),
                "voice_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "ElevenLabs voice ID",
                }),
            },
            "optional": {
                "model_id": (
                    ["eleven_multilingual_sts_v2", "eleven_english_sts_v2"],
                    {"default": "eleven_multilingual_sts_v2"},
                ),
                "stability": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
                "output_format": (
                    ["mp3_44100_128", "mp3_44100_192", "mp3_22050_32"],
                    {"default": "mp3_44100_128"},
                ),
            },
        }

    def process(
        self,
        audio: dict,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_sts_v2",
        stability: float = 1.0,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        remove_background_noise: bool = False,
        output_format: str = "mp3_44100_128",
    ):
        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]

        print(f"[ElevenLabsVC] Input: shape={waveform.shape}, sr={sample_rate}")

        # tensor → WAV bytes
        wav_bytes = tensor_to_wav_bytes(waveform, sample_rate)
        print(f"[ElevenLabsVC] Sending {len(wav_bytes)/1024:.1f} KB WAV to ElevenLabs…")

        # call API
        result_bytes = elevenlabs_sts(
            wav_bytes=wav_bytes,
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            remove_background_noise=remove_background_noise,
            output_format=output_format,
        )
        print(f"[ElevenLabsVC] Got {len(result_bytes)/1024:.1f} KB back")

        # mp3 response → WAV → tensor
        wav_result = mp3_to_wav_bytes(result_bytes)
        out_waveform, out_sr = wav_bytes_to_tensor(wav_result)

        print(f"[ElevenLabsVC] Output: shape={out_waveform.shape}, sr={out_sr}")

        return ({"waveform": out_waveform, "sample_rate": out_sr},)


# ─── registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ElevenLabsVoiceChanger": ElevenLabsVoiceChangerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenLabsVoiceChanger": "🎙️ ElevenLabs Voice Changer",
}
