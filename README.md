# 🎙️ ElevenLabs Voice Changer — ComfyUI Node

Простая нода: **AUDIO in → ElevenLabs STS API → AUDIO out**

## Воркфлоу

```
VHS_LoadVideoFFmpeg
  ├─[images]──► VHS_VideoCombine
  └─[audio] ──► 🎙️ ElevenLabs Voice Changer ──► VHS_VideoCombine [audio]
```

## Установка

```
ComfyUI/custom_nodes/elevenlabs_voice_changer/
    __init__.py
    elevenlabs_voice_changer_node.py
    README.md
```

Зависимости (обычно уже есть в ComfyUI окружении):
- `torch`
- `torchaudio`
- `requests`

## Входы

| Поле | Обязательный | Описание |
|------|:---:|---------|
| `audio` | ✅ | AUDIO поток из VHS_LoadVideoFFmpeg |
| `api_key` | ✅ | Ключ ElevenLabs API |
| `voice_id` | ✅ | ID целевого голоса |
| `model_id` | — | `eleven_multilingual_sts_v2` (default) или `eleven_english_sts_v2` |
| `stability` | — | 0–1, рекомендуется **1.0** |
| `similarity_boost` | — | 0–1, схожесть с целевым голосом |
| `style` | — | 0–1, ставь **0.0** если исходник уже выразительный |
| `remove_background_noise` | — | Шумоподавление перед отправкой |
| `output_format` | — | Формат ответа от API |

## Как получить voice_id

На сайте https://elevenlabs.io/voice-library или через API:
```
GET https://api.elevenlabs.io/v1/voices
xi-api-key: your_key
```
