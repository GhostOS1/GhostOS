"""Optional, offline speech-to-text for GhostOS.

GhostOS never requires voice input to start. Transcription runs only when
enabled in local settings and when a local Whisper model (faster-whisper)
is installed. No audio is ever sent anywhere over the network - the model
runs entirely on-device, unlike browser/OS "cloud speech" APIs.
"""

import importlib.util
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".webm", ".ogg", ".m4a", ".mp3"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 120

# Small, CPU-friendly default. Users with a GPU or who want higher accuracy
# can override via GHOSTOS_WHISPER_MODEL (see config.py).
DEFAULT_MODEL_SIZE = "small.en"

_model = None
_model_size_loaded = None


def get_voice_status() -> dict:
    binding = importlib.util.find_spec("faster_whisper") is not None
    return {
        "available": bool(binding),
        "python_binding": binding,
        "model_size": DEFAULT_MODEL_SIZE,
    }


def _get_model(model_size: str):
    global _model, _model_size_loaded
    if _model is not None and _model_size_loaded == model_size:
        return _model
    from faster_whisper import WhisperModel

    # int8 keeps this usable on a plain CPU without extra native deps.
    _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model_size_loaded = model_size
    return _model


def transcribe_audio(path: Path, model_size: str = DEFAULT_MODEL_SIZE) -> str:
    status = get_voice_status()
    if not status["available"]:
        return ""

    path = Path(path)
    if path.stat().st_size > MAX_AUDIO_BYTES:
        return ""

    model = _get_model(model_size)
    segments, _info = model.transcribe(
        str(path),
        beam_size=1,
        vad_filter=True,
        max_new_tokens=None,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text
