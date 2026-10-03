"""Dataset collection. Run:
  python -m vision.collect --label HELP --signer S01 --session 1

Keys:  SPACE = record ~2 s   |   q = quit
Saves dataset/raw/<LABEL>/<signer>_<session>_<n>.npy with shape (30, 126).
Only landmarks are saved, never video. Check the signer has a row in
dataset/metadata/consent_register.csv first.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import draw_hands
from .landmarks import HandExtractor
from .preprocessing import hand_presence, sample_sequence

ROOT = Path(__file__).resolve().parents[1] / "dataset"
RECORD_SECONDS = 2.0
MIN_PRESENCE = 0.5  # reject clips where a hand is visible in <50% of frames


def append_label(file: str, label: str, signer: str, session: str):
    csv_path = ROOT / "metadata" / "labels.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "label", "signer_id", "session_id"])
        w.writerow([file, label, signer, session])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    label = args.label.upper()
    out_dir = ROOT / "raw" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("*.npy")))

    cap = cv2.VideoCapture(args.cam)
    extractor = HandExtractor()
    t0 = time.time()
    recording, buf, rec_start = False, [], 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        vec, raw = extractor.extract(frame, int((time.time() - t0) * 1000))
        draw_hands(frame, raw)

        if recording:
            buf.append(vec)
            if time.time() - rec_start >= RECORD_SECONDS:
                recording = False
                seq = sample_sequence(buf)
                if hand_presence(seq) < MIN_PRESENCE:
                    print("Rejected: hand not visible enough. Try again.")
                else:
                    count += 1
                    name = f"{args.signer}_{args.session}_{count:03d}.npy"
                    np.save(out_dir / name, seq)
                    append_label(name, label, args.signer, args.session)
                    print(f"Saved {name}  (total {count})")

        color = (0, 0, 255) if recording else (0, 255, 0)
        txt = f"{label} | saved: {count} | " + ("RECORDING" if recording else "SPACE to record")
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and not recording:
            recording, buf, rec_start = True, [], time.time()

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
