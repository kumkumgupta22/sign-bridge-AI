"""Data quality report. Run: python -m vision.quality_report"""
from collections import Counter
from pathlib import Path

import numpy as np

from .preprocessing import hand_presence

RAW = Path(__file__).resolve().parents[1] / "dataset" / "raw"


def main():
    print(f"{'LABEL':<14}{'SAMPLES':>8}{'AVG HAND PRESENCE':>20}")
    totals = Counter()
    for d in sorted(p for p in RAW.glob("*") if p.is_dir()):
        files = list(d.glob("*.npy"))
        pres = [hand_presence(np.load(f)) for f in files]
        totals[d.name] = len(files)
        avg = sum(pres) / len(pres) if pres else 0
        flag = "  <-- need more samples" if len(files) < 30 else ""
        print(f"{d.name:<14}{len(files):>8}{avg:>20.2f}{flag}")
    print(f"\nTotal samples: {sum(totals.values())} across {len(totals)} classes")


if __name__ == "__main__":
    main()
