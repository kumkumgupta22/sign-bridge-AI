# SIGNBRIDGE AI
Two-way communication aid for a controlled vocabulary of ISL signs (MVP). Not a universal ISL translator.

## Run the API (mock predictor)
```
pip install -r backend/requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --reload
pytest backend/tests -q
```
Docs at http://localhost:8000/docs

## Data pipeline (Member 2)
1. Download `hand_landmarker.task` (link in `vision/landmarks.py`) to `models/`.
2. `pip install mediapipe opencv-python numpy`
3. `python -m vision.camera` - check detection
4. `python -m vision.collect --label HELP --signer S01 --session 1`
5. `python -m vision.quality_report`, then `python -m vision.split_dataset`

## Branching
`main` (stable) <- `develop` <- `feature/<member>-<task>`. PRs need one review and green CI.
