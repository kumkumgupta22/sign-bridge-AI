"""Signer-based train/val/test split. Run: python -m vision.split_dataset
Falls back to session-based split if there are fewer than 3 signers.
Writes dataset/metadata/split.csv (file,label,signer_id,session_id,split).
"""
import csv
import random
from pathlib import Path

META = Path(__file__).resolve().parents[1] / "dataset" / "metadata"


def main(seed: int = 42):
    rows = list(csv.DictReader(open(META / "labels.csv")))
    key = "signer_id" if len({r["signer_id"] for r in rows}) >= 3 else None
    groups = sorted({r[key] if key else f"{r['signer_id']}_{r['session_id']}" for r in rows})
    if len(groups) < 3:
        raise SystemExit("Need at least 3 signers or sessions for a proper split.")
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * 0.2))
    n_val = max(1, round(len(groups) * 0.2))
    test, val = set(groups[:n_test]), set(groups[n_test:n_test + n_val])

    out = META / "split.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "signer_id", "session_id", "split"])
        for r in rows:
            g = r[key] if key else f"{r['signer_id']}_{r['session_id']}"
            s = "test" if g in test else "val" if g in val else "train"
            w.writerow([r["file"], r["label"], r["signer_id"], r["session_id"], s])
    print(f"Wrote {out}  (groups: train={len(groups)-n_test-n_val} val={n_val} test={n_test})")


if __name__ == "__main__":
    main()
