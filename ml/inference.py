"""Loads the trained model. Matches the backend Predictor interface."""
from pathlib import Path

import joblib
import numpy as np

from .features import seq_to_features

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "model.joblib"


class SignClassifier:
    def __init__(self, path=DEFAULT_MODEL):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train first: python -m ml.train")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = bundle["classes"]
        self.meta = bundle.get("meta", {})

    def predict(self, frames, labels=None) -> tuple[str, float]:
        x = seq_to_features(frames)[None, :]
        proba = self.model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return str(self.classes[i]), round(float(proba[i]), 4)
