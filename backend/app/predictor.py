"""Predictor interface. Member 1's real model must implement `predict`.

Swap the mock for the real model by setting USE_MOCK_PREDICTOR=false and
implementing RealPredictor.load()/predict() in ml/inference.
"""
import hashlib
from typing import Protocol

import numpy as np

from . import config


class Predictor(Protocol):
    def predict(self, frames: np.ndarray, labels: list[str]) -> tuple[str, float]:
        """frames: (30, 126). Returns (label, confidence in 0..1)."""


class MockPredictor:
    """Deterministic fake so the frontend can be built before the model exists."""

    def predict(self, frames: np.ndarray, labels: list[str]) -> tuple[str, float]:
        digest = hashlib.sha256(frames.tobytes()).digest()
        idx = digest[0] % len(labels)
        confidence = 0.5 + (digest[1] / 255) * 0.49
        return labels[idx], round(confidence, 3)


class RealPredictor:
    def __init__(self, model_path: str):
        raise NotImplementedError("Plug in Member 1's model here (e.g. torch.load).")


def get_predictor() -> Predictor:
    if config.USE_MOCK_PREDICTOR:
        return MockPredictor()
    return RealPredictor(str(config.REPO_ROOT / "ml" / "models" / "model.pt"))
