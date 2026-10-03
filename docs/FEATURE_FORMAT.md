# Feature Format v0.1 (Member 2 -> Member 1)

- Detector: MediaPipe Hand Landmarker (Tasks API), up to 2 hands.
- Per frame: `[Left hand (63) | Right hand (63)]` = 126 floats.
- Per hand: 21 landmarks x (x, y, z). Subtract wrist (landmark 0), divide by
  wrist -> middle-finger-MCP (landmark 9) distance.
- Missing hand: all zeros for that hand (a real normalised hand is never all zeros).
- Per sample: 30 frames, resampled evenly from ~2 s of recording -> shape `(30, 126)`.
- Storage: `dataset/raw/<LABEL>/<signer>_<session>_<n>.npy`; index in `dataset/metadata/labels.csv`.
- Handedness: camera preview is mirrored at collection. Inference MUST use the same convention.
- Splits: by signer (>=3 signers) else by session. Never random per clip.

Any change to this file = bump version and tell Member 1 and Member 5.
