import re

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config, storage
from .predictor import get_predictor
from .schemas import PhraseRequest, PredictRequest, PredictResponse

app = FastAPI(title="SIGNBRIDGE AI API", version=config.MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
predictor = get_predictor()


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    msg = "; ".join(e["msg"] for e in exc.errors())
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": msg})


@app.get("/health")
def health():
    return {"status": "ok", "model_version": config.MODEL_VERSION,
            "mock_predictor": config.USE_MOCK_PREDICTOR}


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
        return PredictResponse(label=None, confidence=0.0, unknown=True,
                               model_version=config.MODEL_VERSION)
    label, conf = predictor.predict(frames, storage.labels())
    unknown = conf < config.CONFIDENCE_THRESHOLD
    return PredictResponse(label=None if unknown else label, confidence=conf,
                           unknown=unknown, model_version=config.MODEL_VERSION)


@app.get("/api/reference/{label}")
def reference(label: str):
    sign = storage.find_sign(label)
    if sign is None:
        raise HTTPException(404, detail=f"No sign reference for '{label}'")
    return sign


@app.post("/api/phrase/map")
def map_phrase(req: PhraseRequest):
    """Simple word lookup against the controlled vocabulary. NOT a translation."""
    words = re.findall(r"[a-zA-Z']+", req.text.upper())
    known = {s["label"]: s for s in storage.load_vocabulary()["signs"]}
    matched, unmatched = [], []
    for w in words:
        key = w.replace("'", "")
        if key in known:
            matched.append(known[key])
        else:
            unmatched.append(w.lower())
    return {
        "matched": matched,
        "unmatched_words": unmatched,
        "note": "Word-by-word reference lookup only; not a fluent ISL translation.",
    }
