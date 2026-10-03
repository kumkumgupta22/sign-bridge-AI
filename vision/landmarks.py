"""MediaPipe Hand Landmarker wrapper (Tasks API).

Download the model once (needs internet) and put it at models/hand_landmarker.task:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .preprocessing import build_frame

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


class HandExtractor:
    def __init__(self, model_path: Path = DEFAULT_MODEL, num_hands: int = 2):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. See the download link in vision/landmarks.py."
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def extract(self, bgr_frame: np.ndarray, timestamp_ms: int):
        """Returns (frame_vector (126,), raw_hands list for drawing)."""
        rgb = bgr_frame[:, :, ::-1].copy()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)  # must increase
        self._last_ts = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        left = right = None
        raw = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            raw.append(pts)
            # Note: MediaPipe assumes a mirrored image. If you flip the preview,
            # keep the SAME convention at collection and at inference.
            if handed[0].category_name == "Left":
                left = pts
            else:
                right = pts
        return build_frame(left, right), raw

    def close(self):
        self._landmarker.close()
