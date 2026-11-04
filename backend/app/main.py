import os
import re
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ml.phrase import signs_to_sentence

from . import config, speech, storage
from .conversation import get_store
from .predictor import get_predictor
from .schemas import (
    MessageRequest,
    PhraseRequest,
    PredictRequest,
    PredictResponse,
    SentenceRequest,
)

app = FastAPI(title="SIGNBRIDGE AI API", version=config.MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
predictor = get_predictor()
store = get_store()

ALIASES = {"THANKS": "THANK YOU", "DOCTORS": "DOCTOR", "MEDICINES": "MEDICINE"}


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    msg = "; ".join(e["msg"] for e in exc.errors())
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": msg})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": config.MODEL_VERSION,
        "mock_predictor": config.USE_MOCK_PREDICTOR,
        "whisper_available": speech.available(),
    }


@app.get("/api/signs")
def list_signs():
    vocab = storage.load_vocabulary()
    return {"version": vocab["version"], "signs": vocab["signs"]}


@app.post("/api/sign/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    frames = np.asarray(req.frames, dtype=np.float32)
    if not np.isfinite(frames).all():
        raise HTTPException(422, detail="frames contain NaN or infinite values")
    if not frames.any():
        # all zeros = no hand detected in any frame
        return PredictResponse(
            label=None, confidence=0.0, unknown=True, model_version=config.MODEL_VERSION
        )
    label, conf = predictor.predict(frames, storage.labels())
    unknown = conf < config.CONFIDENCE_THRESHOLD
    return PredictResponse(
        label=None if unknown else label,
        confidence=conf,
        unknown=unknown,
        model_version=config.MODEL_VERSION,
    )


@app.post("/api/sentence")
def make_sentence(req: SentenceRequest):
    return {"sentence": signs_to_sentence(req.labels)}


@app.get("/api/reference/{label}")
def reference(label: str):
    sign = storage.find_sign(label)
    if sign is None:
        raise HTTPException(404, detail=f"No sign reference for '{label}'")
    return sign


@app.post("/api/phrase/map")
def map_phrase(req: PhraseRequest):
    """Word lookup against the controlled vocabulary. NOT a translation."""
    known = {s["label"].replace("_", " ").upper(): s for s in storage.load_vocabulary()["signs"]}
    words = [w.replace("'", "") for w in re.findall(r"[A-Za-z']+", req.text.upper())]
    matched, unmatched, i = [], [], 0
    while i < len(words):
        pair = " ".join(words[i : i + 2])
        pair = ALIASES.get(pair, pair)
        single = ALIASES.get(words[i], words[i])
        if len(words[i : i + 2]) == 2 and pair in known:
            matched.append(known[pair])
            i += 2
        elif single in known:
            matched.append(known[single])
            i += 1
        else:
            unmatched.append(words[i].lower())
            i += 1
    return {
        "matched": matched,
        "unmatched_words": unmatched,
        "note": "Word-by-word reference lookup only; not a fluent ISL translation.",
    }


@app.post("/api/speech/transcribe")
def transcribe(file: UploadFile = File(...), language: str = "", translate: bool = False):
    data = file.file.read(config.MAX_AUDIO_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"Audio larger than {config.MAX_AUDIO_MB} MB")
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        return speech.transcribe(tmp.name, language or None, translate)
    except speech.SpeechUnavailable as e:
        return JSONResponse(status_code=503, content={"error": "speech_unavailable", "detail": str(e)})
    finally:
        os.unlink(tmp.name)  # audio is never kept


@app.post("/api/conversation")
def add_message(req: MessageRequest):
    return store.add(req.speaker, req.text, req.label, req.confidence)


@app.get("/api/conversation/history")
def history(limit: int = 50):
    return {"messages": store.history(max(1, min(limit, 200)))}


@app.delete("/api/conversation")
def clear_history():
    store.clear()
    return {"cleared": True}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/app/")


app.mount(
    "/clips",
    StaticFiles(directory=config.REPO_ROOT / "sign_library", check_dir=False),
    name="clips",
)
app.mount("/app", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="web")
