"""Live view of what the robot's Kinect sees: RGB | depth, plus the runner's status.

    pythonw kinect_live.py

Whatever process currently owns the Kinect (run_pick_place, calibration, ...) writes
control/live.jpg a few times a second; this window just shows it. Esc or q closes it.
"""
import os
import time

import cv2
import numpy as np

import control

LIVE = control.DIR / "live.jpg"
WIN = "Kinect: what the robot sees"


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 520)
    last = np.zeros((480, 1280, 3), np.uint8)
    while True:
        try:
            img = cv2.imread(str(LIVE))
            if img is not None:
                last = img
            age = time.time() - os.path.getmtime(LIVE)
        except OSError:
            age = float("inf")
        frame = last.copy()
        try:
            status = control.STATUS.read_text().strip()
        except OSError:
            status = ""
        bar = np.zeros((40, frame.shape[1], 3), np.uint8)
        cv2.putText(bar, status[:110], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if age > 2:
            cv2.putText(frame, "Kinect idle (no process is streaming)" if age < 1e9 else "waiting for frames...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.imshow(WIN, np.vstack([frame, bar]))
        if cv2.waitKey(100) in (27, ord("q")) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
