import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["USE_MOCK_PREDICTOR"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def good_frames(seed=1):
    import random
    random.seed(seed)
    return [[random.random() for _ in range(126)] for _ in range(30)]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_signs():
    r = client.get("/api/signs")
    assert r.status_code == 200 and len(r.json()["signs"]) > 0


def test_predict_ok():
    r = client.post("/api/sign/predict", json={"frames": good_frames()})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "confidence", "unknown", "model_version"}


def test_predict_wrong_shape():
    r = client.post("/api/sign/predict", json={"frames": [[0.0] * 126] * 10})
    assert r.status_code == 422 and r.json()["error"] == "invalid_input"


def test_predict_no_hand():
    r = client.post("/api/sign/predict", json={"frames": [[0.0] * 126] * 30})
    assert r.json()["unknown"] is True


def test_reference_found_and_missing():
    assert client.get("/api/reference/help").status_code == 200
    assert client.get("/api/reference/zzz").status_code == 404


def test_phrase_map():
    r = client.post("/api/phrase/map", json={"text": "Do you need water, doctor?"})
    labels = [m["label"] for m in r.json()["matched"]]
    assert "WATER" in labels and "DOCTOR" in labels
