"""Turns a (30, 126) landmark sequence into one feature vector for the classifier."""
import numpy as np

FEATURE_VERSION = "0.1"
NUM_FRAMES = 30
FRAME_DIM = 126


def seq_to_features(seq) -> np.ndarray:
    a = np.asarray(seq, dtype=np.float32)
    if a.shape != (NUM_FRAMES, FRAME_DIM):
        raise ValueError(f"expected shape ({NUM_FRAMES}, {FRAME_DIM}), got {a.shape}")
    subsampled = a[::3].reshape(-1)  # 10 frames: coarse trajectory
    mean = a.mean(axis=0)  # overall hand shape
    std = a.std(axis=0)  # how much each coordinate moves
    speed = np.abs(np.diff(a, axis=0)).mean(axis=0)  # movement speed
    return np.concatenate([subsampled, mean, std, speed])
