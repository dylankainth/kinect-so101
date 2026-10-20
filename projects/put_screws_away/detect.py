"""Find the screws lying on the mat from the colour image (they are only a few mm tall, so depth cannot see them).

A screw is a small, thin, bright, grey-ish blob on the dark green mat. Its centre and direction come from the blob's
minimum-area rectangle; the pixel is turned into a robot-frame point by intersecting the camera ray with the table.

    .venv\Scripts\python -m projects.put_screws_away.detect            # look, print, save control/screws.jpg
"""
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core import control
from core.perception.kinect import Kinect
from core.perception.util import kinect_retry, xy_from_pixel

log = logging.getLogger("screws")

ROI = (150, 150, 535, 445)         # u0, v0, u1, v1: the mat, without the table edge
EXCLUDE = ((295, 0, 400, 196),     # the arm (u0, v0, u1, v1)
           (333, 196, 378, 238),   # the calibration marker
           (400, 150, 535, 380))   # the tray
CONTRAST = 25                      # V brighter than the local background by this much (colour fringes make S useless)
MIN_AREA, MAX_AREA = 10, 400       # px
MIN_LEN_PX = 10                    # a screw is about 18 px long in the image
MIN_ASPECT = 3.0                   # length / width of the min-area rectangle
TABLE_Z = 0.002                    # m, the screw's axis above the mat


@dataclass(eq=False)
class Screw:
    xy: np.ndarray    # robot frame, m
    yaw: float        # rad: direction of the screw's length in the robot frame (x forward, y left)
    length_px: float
    px: tuple


def find_screws(rgb, intr, T, debug=None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32)
    bg = cv2.medianBlur(hsv[..., 2], 25).astype(np.float32)   # the mat's own brightness (grid lines are thin)
    mask = cv2.morphologyEx((v - bg > CONTRAST).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    keep = np.zeros_like(mask)
    u0, v0, u1, v1 = ROI
    keep[v0:v1, u0:u1] = 1
    for a, b, c, d in EXCLUDE:
        keep[b:d, a:c] = 0
    mask *= keep
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        L, W = max(w, h), max(min(w, h), 1.0)
        if L < MIN_LEN_PX or L / W < MIN_ASPECT:
            continue
        # direction of the long side in the image, then through the camera ray at both ends
        th = np.radians(ang if w >= h else ang + 90)
        d = np.array([np.cos(th), np.sin(th)]) * L / 2
        a = xy_from_pixel(intr, T, (cx - d[0], cy - d[1]), TABLE_Z)
        b = xy_from_pixel(intr, T, (cx + d[0], cy + d[1]), TABLE_Z)
        c = xy_from_pixel(intr, T, (cx, cy), TABLE_Z)
        dv = b - a
        out.append(Screw(c, float(np.arctan2(dv[1], dv[0])), float(L), (cx, cy)))
    if debug is not None:
        debug["mask"] = mask
    return sorted(out, key=lambda s: s.px[1])


def perceive_screws(n_frames=5):
    from core.perception.calibrate import load_calib

    def grab():
        with Kinect() as kin:
            kin.wait_for_frames(8)
            time.sleep(1.0)
            frames = []
            while len(frames) < n_frames:   # the mean of a few frames: less sensor noise on 3 mm features
                time.sleep(0.15)
                f, _ = kin.get_rgb()
                frames.append(f.astype(np.float32))
            return np.mean(frames, axis=0).astype(np.uint8), kin.intr
    rgb, intr = kinect_retry(grab, "screw detection")
    return find_screws(rgb, intr, load_calib()), rgb


def draw(rgb, screws, path):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for k, s in enumerate(screws, 1):
        u, v = map(int, s.px)
        cv2.circle(img, (u, v), 9, (0, 0, 255), 1)
        cv2.putText(img, str(k), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    u0, v0, u1, v1 = ROI
    cv2.rectangle(img, (u0, v0), (u1, v1), (255, 255, 0), 1)
    for a, b, c, d in EXCLUDE:
        cv2.rectangle(img, (a, b), (c, d), (255, 0, 255), 1)
    cv2.imwrite(str(path), img)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    screws, rgb = perceive_screws()
    for k, s in enumerate(screws, 1):
        print(f"{k:2d}: robot ({s.xy[0]:+.3f}, {s.xy[1]:+.3f}) m, yaw {np.degrees(s.yaw):+6.1f} deg, {s.length_px:.0f} px long")
    draw(rgb, screws, control.DIR / "screws.jpg")
    print(f"{len(screws)} screw(s); picture in control/screws.jpg")


if __name__ == "__main__":
    main()
