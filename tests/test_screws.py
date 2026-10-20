"""projects.put_screws_away: screw detection on a synthetic image, and the pick / dropped-screw bookkeeping.

    .venv\Scripts\python -m tests.test_screws
"""
import cv2
import numpy as np

from core.perception.calibrate import load_calib
from projects.put_screws_away import detect, main as sa

INTR = dict(fx=529.2, fy=525.6, cx=328.9, cy=267.5)


def mat():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (20, 60, 40)                       # dark green, RGB
    for x in range(160, 520, 40):               # thin, fainter grid lines like the real mat
        cv2.line(img, (x, 150), (x, 445), (32, 78, 54), 1)
    return img


def screw(img, c, deg, length=18):
    d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))]) * length / 2
    cv2.line(img, tuple(np.round(c - d).astype(int)), tuple(np.round(c + d).astype(int)), (170, 175, 170), 2)


def main():
    T = load_calib()
    img = mat()
    screw(img, np.array([250.0, 300.0]), 0)     # lies along the image's horizontal = robot y
    screw(img, np.array([330.0, 330.0]), 90)    # along the image's vertical = robot x
    screw(img, np.array([200.0, 380.0]), 40)
    found = detect.find_screws(img, INTR, T)
    assert len(found) == 3, [s.px for s in found]
    by_px = {round(s.px[0] / 10): s for s in found}
    assert abs(abs(np.degrees(by_px[25].yaw)) - 90) < 8, np.degrees(by_px[25].yaw)
    assert min(abs(np.degrees(by_px[33].yaw)), abs(abs(np.degrees(by_px[33].yaw)) - 180)) < 8
    print("detector: 3 synthetic screws found, directions right")

    assert detect.find_screws(mat(), INTR, T) == []
    print("detector: empty mat -> nothing (grid lines are not screws)")

    screw(img := mat(), np.array([420.0, 250.0]), 90)   # inside the tray's exclusion box
    assert detect.find_screws(img, INTR, T) == []
    print("detector: something in the tray is ignored")

    mk = lambda x, y: detect.Screw(np.array([x, y]), 0.0, 18.0, (0, 0))
    a, b, c = mk(0.25, 0.0), mk(0.30, 0.05), mk(0.30, -0.08)
    assert sa.isolation(c, [a, b, c]) > sa.isolation(a, [a, b, c]), "the far screw is the more isolated one"
    gone, new = sa.mat_changed([a, b, c], [b, c], a)
    assert gone and not new
    gone, new = sa.mat_changed([a, b, c], [a, b, c], a)
    assert not gone and not new
    gone, new = sa.mat_changed([a, b, c], [b, c, mk(0.40, 0.0)], a)
    assert gone and len(new) == 1
    print("bookkeeping: picked / not picked / dropped screw told apart")
    print("ALL OK")


if __name__ == "__main__":
    main()
