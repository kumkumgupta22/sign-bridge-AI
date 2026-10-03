import numpy as np

from vision.preprocessing import (FRAME_DIM, build_frame, hand_presence,
                                  normalize_hand, sample_sequence)


def fake_hand(scale=1.0, offset=(0.3, 0.4, 0.0)):
    rng = np.random.default_rng(0)
    pts = rng.random((21, 3)).astype(np.float32) * scale
    return pts + np.array(offset, dtype=np.float32)


def test_translation_and_scale_invariant():
    a = normalize_hand(fake_hand(1.0, (0.1, 0.1, 0)))
    b = normalize_hand(fake_hand(2.0, (0.5, 0.2, 0)))
    assert np.allclose(a, b, atol=1e-5)


def test_frame_shape_and_missing_hand():
    f = build_frame(fake_hand(), None)
    assert f.shape == (FRAME_DIM,)
    assert not f[63:].any() and f[:63].any()


def test_sample_sequence():
    frames = [build_frame(fake_hand(), None) for _ in range(47)]
    assert sample_sequence(frames).shape == (30, 126)
    assert sample_sequence([]).shape == (30, 126)
    assert hand_presence(sample_sequence(frames)) == 1.0
