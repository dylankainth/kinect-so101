"""Live view of what the robot's Kinect sees: RGB | depth, plus the runner's status.

    pythonw -m core.tools.kinect_live

Whatever process currently owns the Kinect (a project runner, calibration, ...) writes
control/live.jpg a few times a second; this window just shows it. Esc or q closes it.
"""
import json
import os
import time

import cv2
import numpy as np

from core import control
from core.paths import CALIB_PATH
from core.perception.kinect import RGB_INTR

LIVE = control.DIR / "live.jpg"
WIN = "Kinect: what the robot sees"
OVERLAY = control.DIR / "overlay.json"   # targets from the runner: [{label, xyz (robot frame), color (BGR)}]


def draw_targets(frame, T):
    """Cross-hairs at the planner's pick and place points, on the RGB half (and the depth half beside it)."""
    try:
        if time.time() - os.path.getmtime(OVERLAY) > 900:
            return
        pts = json.loads(OVERLAY.read_text())
    except (OSError, ValueError):
        return
    R, t = T[:3, :3], T[:3, 3]
    for p in pts:
        c = R.T @ (np.array(p["xyz"]) - t)
        if c[2] <= 0:
            continue
        u, v = int(RGB_INTR["fx"] * c[0] / c[2] + RGB_INTR["cx"]), int(RGB_INTR["fy"] * c[1] / c[2] + RGB_INTR["cy"])
        for dx in (0, 640):
            if dx and frame.shape[1] < 1280:
                continue
            cv2.drawMarker(frame, (u + dx, v), tuple(p["color"]), cv2.MARKER_CROSS, 28, 2)
            cv2.putText(frame, p["label"], (u + dx + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(p["color"]), 2)


def main():
    T = np.array(json.loads(CALIB_PATH.read_text())["T_robot_cam"])
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
        draw_targets(frame, T)
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
