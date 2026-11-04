import os
from pathlib import Path

NUM_FRAMES = 30
FEATURES_PER_FRAME = 126  # 2 hands x 21 landmarks x (x, y, z)

MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
USE_MOCK_PREDICTOR = os.getenv("USE_MOCK_PREDICTOR", "true").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MONGO_URI = os.getenv("MONGO_URI", "")
MAX_AUDIO_MB = 10

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = Path(os.getenv("VOCAB_PATH", REPO_ROOT / "dataset" / "vocabulary.json"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", REPO_ROOT / "ml" / "models" / "model.joblib"))
DB_PATH = os.getenv("DB_PATH", str(REPO_ROOT / "data" / "signbridge.db"))
