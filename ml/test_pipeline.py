import csv

import numpy as np

from ml import train
from ml.inference import SignClassifier
from ml.phrase import signs_to_sentence

LABELS = ["HELP", "WATER", "DOCTOR"]


def make_dataset(root):
    rng = np.random.default_rng(0)
    base = {lab: rng.normal(size=(30, 126)).astype(np.float32) for lab in LABELS}
    raw, meta = root / "raw", root / "meta"
    meta.mkdir()
    rows = []
    for lab in LABELS:
        (raw / lab).mkdir(parents=True)
        for signer in range(1, 7):
            for k in range(8):
                name = f"S{signer:02d}_1_{k:03d}.npy"
                seq = base[lab] + rng.normal(scale=0.1, size=(30, 126)).astype(np.float32)
                np.save(raw / lab / name, seq)
                split = "test" if signer == 6 else "val" if signer == 5 else "train"
                rows.append([name, lab, f"S{signer:02d}", "1", split])
    with open(meta / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "signer_id", "session_id", "split"])
        w.writerows(rows)
    return raw, meta


def test_train_and_infer(tmp_path):
    raw, meta = make_dataset(tmp_path)
    out = tmp_path / "models"
    metrics = train.run(meta, raw, out, make_plot=False, verbose=False)
    assert metrics["test_accuracy"] > 0.9
    clf = SignClassifier(out / "model.joblib")
    seq = np.load(raw / "WATER" / "S06_1_000.npy")
    label, conf = clf.predict(seq)
    assert label == "WATER" and 0 < conf <= 1


def test_sentences():
    assert signs_to_sentence(["I", "NEED", "WATER"]) == "I need water."
    assert signs_to_sentence(["I", "NEED", "DOCTOR"]) == "I need a doctor."
    assert signs_to_sentence(["WHERE", "HOSPITAL"]) == "Where hospital?"
    assert signs_to_sentence(["THANK_YOU"]) == "Thank you."
    assert signs_to_sentence([]) == ""
