# SIGNBRIDGE AI
Two-way communication aid for a controlled vocabulary of ISL signs (MVP). Not a universal ISL translator and not a replacement for an interpreter.

- Sign -> speech: camera -> MediaPipe hand landmarks -> classifier -> confirm -> phrase -> voice (`python -m vision.demo`)
- Speech -> sign reference: Whisper or browser speech -> vocabulary lookup -> reference clips (`/app/`)

See `docs/RUNBOOK.md` for the step-by-step commands, `docs/API.md` for the API, `docs/FEATURE_FORMAT.md` for the data contract.

## Branching
`main` (stable) <- `develop` <- `feature/<task>`. Run `ruff check backend vision ml --fix` and `pytest -q` before pushing.
