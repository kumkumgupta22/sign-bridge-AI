"""Feature contract (Member 2 -> Member 1). See docs/FEATURE_FORMAT.md.

Per frame: [Left hand 63 | Right hand 63] = 126 floats.
Per hand: 21 landmarks x (x, y, z), wrist-centred and scaled by the
wrist -> middle-finger-MCP distance. A missing hand is all zeros.
(A real normalised hand is never all zeros: landmark 9 sits at distance 1.)
Per sample: 30 frames -> shape (30, 126).
"""
import numpy as np

NUM_FRAMES = 30
NUM_LANDMARKS = 21
HAND_DIM = NUM_LANDMARKS * 3
FRAME_DIM = HAND_DIM * 2
WRIST, MIDDLE_MCP = 0, 9


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (21, 3) raw MediaPipe coords -> flat (63,) normalised."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    pts = pts - pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        return np.zeros(HAND_DIM, dtype=np.float32)
    return (pts / scale).reshape(-1)


def build_frame(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    """Each argument is a (21, 3) array or None if that hand was not seen."""
    l_vec = normalize_hand(left) if left is not None else np.zeros(HAND_DIM, np.float32)
    r_vec = normalize_hand(right) if right is not None else np.zeros(HAND_DIM, np.float32)
    return np.concatenate([l_vec, r_vec])


def sample_sequence(frames: list[np.ndarray], n: int = NUM_FRAMES) -> np.ndarray:
    """Resample a variable-length list of frame vectors to exactly n frames."""
    if len(frames) == 0:
        return np.zeros((n, FRAME_DIM), dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([frames[i] for i in idx]).astype(np.float32)


def hand_presence(seq: np.ndarray) -> float:
    """Fraction of frames with at least one hand. Use to reject bad recordings."""
    return float((np.abs(seq).sum(axis=1) > 0).mean())
