"""Guided dataset recorder. Walks through the vocabulary one sign at a time.

  python -m vision.collect --signer S01 --session 1                # all signs, 30 reps each
  python -m vision.collect --signer S01 --session 1 --labels HELP WATER --reps 20

Keys:  SPACE record (1 s get-ready, then 2 s)   u undo last   n next sign   p previous   q quit
Saves dataset/raw/<LABEL>/<signer>_<session>_<n>.npy (landmarks only - never video).
Check the signer has a row in dataset/metadata/consent_register.csv first.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import draw_hands
from .landmarks import HandExtractor
from .preprocessing import hand_presence, sample_sequence

ROOT = Path(__file__).resolve().parents[1] / "dataset"
LABELS_CSV = ROOT / "metadata" / "labels.csv"
READY_SECONDS = 1.0
RECORD_SECONDS = 2.0
MIN_PRESENCE = 0.5  # reject clips where a hand is visible in <50% of frames


def load_labels() -> list[str]:
    with open(ROOT / "vocabulary.json", encoding="utf-8") as f:
        return [s["label"] for s in json.load(f)["signs"]]


def count_for(out_dir: Path, signer: str, session: str) -> int:
    return len(list(out_dir.glob(f"{signer}_{session}_*.npy")))


def next_index(out_dir: Path, signer: str, session: str) -> int:
    nums = [int(p.stem.rsplit("_", 1)[1]) for p in out_dir.glob(f"{signer}_{session}_*.npy")]
    return max(nums, default=0) + 1


def append_label(file: str, label: str, signer: str, session: str):
    new = not LABELS_CSV.exists()
    with open(LABELS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "label", "signer_id", "session_id"])
        w.writerow([file, label, signer, session])


def remove_label(file: str):
    if not LABELS_CSV.exists():
        return
    with open(LABELS_CSV, newline="") as f:
        rows = [r for r in csv.reader(f) if r and r[0] != file]
    with open(LABELS_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def put_lines(frame, lines, color):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    labels = [x.upper() for x in args.labels] if args.labels else load_labels()
    idx = 0
    while idx < len(labels) and count_for(
        ROOT / "raw" / labels[idx], args.signer, args.session
    ) >= args.reps:
        idx += 1
    if idx >= len(labels):
        print("All signs already have enough samples for this signer/session.")
        return

    ROOT.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    extractor = HandExtractor()
    t0 = time.time()
    phase, phase_start, buf = "idle", 0.0, []
    saved: list[tuple[str, str]] = []  # (label, filename) for undo
    message = ""

    while idx < len(labels):
        label = labels[idx]
        out_dir = ROOT / "raw" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        count = count_for(out_dir, args.signer, args.session)

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        vec, raw = extractor.extract(frame, int((time.time() - t0) * 1000))
        draw_hands(frame, raw)
        now = time.time()

        if phase == "ready" and now - phase_start >= READY_SECONDS:
            phase, phase_start, buf = "recording", now, []
        elif phase == "recording":
            buf.append(vec)
            if now - phase_start >= RECORD_SECONDS:
                phase = "idle"
                seq = sample_sequence(buf)
                if hand_presence(seq) < MIN_PRESENCE:
                    message = "Rejected: hand not visible enough. Try again."
                else:
                    name = f"{args.signer}_{args.session}_{next_index(out_dir, args.signer, args.session):03d}.npy"
                    np.save(out_dir / name, seq)
                    append_label(name, label, args.signer, args.session)
                    saved.append((label, name))
                    count += 1
                    message = f"Saved {name}"
                    if count >= args.reps:
                        idx += 1
                        message += "  -> sign complete, moving on"
                        continue

        status = {"idle": "SPACE to record", "ready": "GET READY...", "recording": "RECORDING"}[phase]
        put_lines(
            frame,
            [f"{label}  ({idx + 1}/{len(labels)})", f"saved {count}/{args.reps}  |  {status}", message],
            (0, 0, 255) if phase == "recording" else (0, 255, 0),
        )
        cv2.imshow("SIGNBRIDGE collect (q quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and phase == "idle":
            phase, phase_start, message = "ready", now, ""
        elif key == ord("n"):
            idx = min(idx + 1, len(labels))
        elif key == ord("p"):
            idx = max(idx - 1, 0)
        elif key == ord("u") and saved:
            last_label, last_file = saved.pop()
            (ROOT / "raw" / last_label / last_file).unlink(missing_ok=True)
            remove_label(last_file)
            message = f"Undid {last_file}"

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done. Next: python -m vision.quality_report")


if __name__ == "__main__":
    main()
