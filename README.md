# 🎙️ ElevenLabs Voice Changer — ComfyUI Nodes

Две ноды в одном пакете. Обе принимают `AUDIO` и возвращают `AUDIO`.

## Воркфлоу

```
VHS_LoadVideoFFmpeg
  ├─[images] ──────────────────────────────────────► VHS_VideoCombine
  └─[audio]  ──► 🎙️ ElevenLabs Voice Changer ──────► VHS_VideoCombine [audio]
```

---

## 🎙️ ElevenLabs Voice Changer (прямой API)

**Входы:**

| Поле | Обязательный | Описание |
|------|:---:|---------|
| `audio` | ✅ | AUDIO поток |
| `api_key` | ✅ | ElevenLabs API key |
| `voice_id` | ✅ | ID голоса из библиотеки ElevenLabs |
| `model_id` | — | `eleven_multilingual_sts_v2` (default) или `eleven_english_sts_v2` |
| `stability` | — | 0–1. Низкое = эмоциональнее, высокое = монотоннее. **Default: 0.5** |
| `similarity_boost` | — | 0–1. Схожесть с целевым голосом. Высокое + плохой исходник = артефакты. **Default: 0.75** |
| `style` | — | 0–1. Усиление стиля. ElevenLabs рекомендует держать на **0**. |
| `remove_background_noise` | — | Шумоподавление перед отправкой |
| `seed` | — | 0 = рандом. Число от 1 до 4294967295 = детерминированный результат |
| `output_format` | — | Формат ответа от API |

---

## 🎙️ ElevenLabs Voice Changer (fal.ai)

Тот же ElevenLabs STS, но через инфраструктуру fal.ai (очередь, меньше таймаутов на длинных файлах).

**Входы:**

| Поле | Обязательный | Описание |
|------|:---:|---------|
| `audio` | ✅ | AUDIO поток |
| `api_key` | ✅ | fal.ai API key (`FAL_KEY`) |
| `voice` | ✅ | Имя голоса (`Rachel`, `Aria` и т.д.) или voice_id |
| `remove_background_noise` | — | Шумоподавление |
| `seed` | — | 0 = рандом. Число = детерминированный результат |
| `output_format` | — | Формат ответа |

---

## Установка

```
ComfyUI/custom_nodes/elevenlabs_voice_changer/
    __init__.py
    elevenlabs_voice_changer_node.py
    elevenlabs_fal_voice_changer_node.py
    README.md
```

Зависимости: `torch`, `torchaudio`, `requests` — всё есть в стандартном окружении ComfyUI.
