"""Speech-to-text with Whisper (faster-whisper). Install: pip install -r requirements-ai.txt"""
import importlib.util
import threading

from . import config

_model = None
_lock = threading.Lock()


class SpeechUnavailable(Exception):
    pass


def available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def transcribe(path: str, language: str | None = None, translate: bool = False) -> dict:
    global _model
    if not available():
        raise SpeechUnavailable("faster-whisper is not installed (pip install -r requirements-ai.txt)")
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = _model.transcribe(
        path,
        language=language or None,
        task="translate" if translate else "transcribe",
        vad_filter=True,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": info.language}
