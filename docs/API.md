# API v0.2

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness, model version, whether Whisper is installed |
| GET | /api/signs | Vocabulary |
| POST | /api/sign/predict | Predict a sign from (30,126) frames |
| POST | /api/sentence | `{"labels": ["I","NEED","WATER"]}` -> `{"sentence": "I need water."}` |
| GET | /api/reference/{label} | Reference clip metadata |
| POST | /api/phrase/map | Word lookup against the vocabulary (not a translation) |
| POST | /api/speech/transcribe | multipart `file` (+ `?language=&translate=`) -> `{"text","language"}`; 503 if Whisper missing |
| POST | /api/conversation | `{"speaker": "sign"|"speech", "text": "...", "label"?, "confidence"?}` |
| GET | /api/conversation/history?limit=50 | Recent messages |
| DELETE | /api/conversation | Delete all stored messages |
| GET | /app/ | Speech -> sign reference web page |

## POST /api/sign/predict
Request: `{"frames": [[126 floats] x 30]}`
Response: `{"label": "HELP" | null, "confidence": 0.94, "unknown": false, "model_version": "0.1.0"}`
`unknown: true` and `label: null` when confidence < CONFIDENCE_THRESHOLD or no hand in any frame.

Errors: `422 {"error": "invalid_input", "detail": "..."}`, `404` unknown reference, `413` audio too large, `503` Whisper not installed.

Privacy: audio is deleted right after transcription; only text is stored in the history.
