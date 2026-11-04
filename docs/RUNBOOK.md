# Runbook (solo workflow, in order)

Activate the venv first: `.venv\Scripts\activate`

## One-time setup
```
pip install -r backend/requirements.txt
pip install -r requirements-demo.txt
pip install -r requirements-ai.txt        # optional: Whisper (large download on first use)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

## 1. Check the camera
`python -m vision.camera`

## 2. Freeze the vocabulary
Edit `dataset/vocabulary.json` (20 starter signs). Verify against the ISLRTC dictionary, fill definitions.

## 3. Record (repeat per signer and session)
Add the signer to `dataset/metadata/consent_register.csv` first.
`python -m vision.collect --signer S01 --session 1`
Target: 30 reps per sign, 3 signers, 2 sessions (different lighting/background).

## 4. Check, split, train
```
python -m vision.quality_report
python -m vision.split_dataset
python -m ml.train
```
`ml/models/metrics.json` and `confusion_matrix.png` are the numbers for your slides (held-out test signers only).

## 5. Run the live demo (sign -> phrase -> voice)
`python -m vision.demo`

## 6. Run the web app (speech -> sign reference) with the real model
Set `USE_MOCK_PREDICTOR=false` in `.env`, then:
`uvicorn backend.app.main:app --reload`  ->  http://localhost:8000/app/

## 7. Before every push
`ruff check backend vision ml --fix` and `pytest -q`
