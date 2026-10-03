# API v0.1

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness + model version |
| GET | /api/signs | Vocabulary |
| POST | /api/sign/predict | Predict sign from (30,126) frames |
| GET | /api/reference/{label} | Reference clip metadata |
| POST | /api/phrase/map | Word lookup against vocabulary (not translation) |

## POST /api/sign/predict
Request: `{"frames": [[126 floats] x 30]}`

Response: `{"label": "HELP" | null, "confidence": 0.94, "unknown": false, "model_version": "0.1.0"}`

- `unknown: true` and `label: null` when confidence < CONFIDENCE_THRESHOLD or no hand in any frame.
- Errors: `422 {"error": "invalid_input", "detail": "..."}`; `404` for unknown reference label.
