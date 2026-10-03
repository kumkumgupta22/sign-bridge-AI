"""Live camera preview with hand-detection overlay. Run: python -m vision.camera"""
import time

import cv2

from .landmarks import HandExtractor


def draw_hands(img, raw_hands):
    h, w = img.shape[:2]
    for pts in raw_hands:
        for x, y, _ in pts:
            cv2.circle(img, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)


def main(cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    extractor = HandExtractor()
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        _, raw = extractor.extract(frame, int((time.time() - t0) * 1000))
        draw_hands(frame, raw)
        status = f"Hands detected: {len(raw)}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if raw else (0, 0, 255), 2)
        cv2.imshow("SIGNBRIDGE camera (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    extractor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
