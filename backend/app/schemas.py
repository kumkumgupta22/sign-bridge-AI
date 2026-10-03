from pydantic import BaseModel, field_validator

from .config import FEATURES_PER_FRAME, NUM_FRAMES


class PredictRequest(BaseModel):
    frames: list[list[float]]

    @field_validator("frames")
    @classmethod
    def check_shape(cls, v):
        if len(v) != NUM_FRAMES:
            raise ValueError(f"frames must contain exactly {NUM_FRAMES} frames, got {len(v)}")
        for i, f in enumerate(v):
            if len(f) != FEATURES_PER_FRAME:
                raise ValueError(
                    f"frame {i} must have {FEATURES_PER_FRAME} values, got {len(f)}"
                )
        return v


class PredictResponse(BaseModel):
    label: str | None
    confidence: float
    unknown: bool
    model_version: str


class PhraseRequest(BaseModel):
    text: str
