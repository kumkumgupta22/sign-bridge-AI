"""Live demo: sign -> confirm -> phrase -> speech.   Run: python -m vision.demo

Keys: SPACE capture a sign | y accept the suggestion | n reject | b remove last word
      s speak the phrase   | c clear phrase          | q quit
The model only SUGGESTS a sign; you confirm it. Low confidence shows "Not sure".
"""
import os
import threading
import time

import cv2

from ml.inference import SignClassifier
from ml.phrase import signs_to_sentence

from .camera import draw_hands
from .landmarks import HandExtractor
from .preprocessing import hand_presence, sample_sequence

THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
READY_SECONDS, RECORD_SECONDS = 1.0, 2.0


def speak(text: str):
    def _run():
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[speech unavailable] {text}  ({e})")

    threading.Thread(target=_run, daemon=True).start()


def put(frame, text, y, color=(0, 255, 0), scale=0.8):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main(cam_index: int = 0):
    clf = SignClassifier()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    extractor = HandExtractor()
    t0 = time.time()
    phase, phase_start, buf = "idle", 0.0, []
    candidate, words, info = None, [], "Press SPACE and perform a sign"

    while True:
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
                if hand_presence(seq) < 0.5:
                    candidate, info = None, "No hand detected - try again"
                else:
                    label, conf = clf.predict(seq)
                    if conf < THRESHOLD:
                        candidate, info = None, f"Not sure ({label}? {conf:.0%}) - repeat the sign"
                    else:
                        candidate, info = label, f"{label} {conf:.0%}   y=accept  n=reject"

        status = {"idle": "SPACE = capture", "ready": "GET READY...", "recording": "RECORDING"}[phase]
        put(frame, status, 30, (0, 0, 255) if phase == "recording" else (0, 255, 0))
        put(frame, info, 65, (255, 255, 0))
        sentence = signs_to_sentence(words)
        put(frame, f"Phrase: {sentence}", frame.shape[0] - 20, (255, 255, 255))
        cv2.imshow("SIGNBRIDGE demo (q quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and phase == "idle":
            phase, phase_start = "ready", now
        elif key == ord("y") and candidate:
            words.append(candidate)
            candidate, info = None, f"Added {words[-1]}. SPACE for next sign, s to speak"
        elif key == ord("n"):
            candidate, info = None, "Rejected. SPACE to try again"
        elif key == ord("b") and words:
            words.pop()
        elif key == ord("c"):
            words = []
        elif key == ord("s") and words:
            speak(sentence)
            info = f"Speaking: {sentence}"

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
