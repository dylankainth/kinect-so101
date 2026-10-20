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
EXCLUDE = ((295, 0, 382, 196),     # the arm (u0, v0, u1, v1)
           (333, 196, 378, 238))   # the calibration marker; the tray is excluded where the camera finds it
CONTRAST = 25                      # V brighter than the local background by this much (colour fringes make S useless)
MIN_AREA, MAX_AREA = 10, 700       # px
MIN_LEN_PX = 10                    # a screw is about 18 px long in the image, a bolt about 22
MAX_LEN_PX = 34                    # longer than this is two touching (a longer bolt is 30 px, a touching pair 38)
MAX_WIDTH_PX = 9                   # wider than this likewise (a touching pair measured 15)
MIN_ASPECT = 2.5                   # length / width of the min-area rectangle
TABLE_Z = 0.002                    # m, the screw's axis above the mat


@dataclass(eq=False)
class Screw:
    xy: np.ndarray    # robot frame, m
    yaw: float        # rad: direction of the screw's length in the robot frame (x forward, y left)
    length_px: float
    px: tuple


# nuts lie flat and look like small compact blobs: (min, max) of the longer side in px, widest the shorter side, max aspect
NUT = dict(len=(7, 15), max_w=12, max_aspect=1.8, area=(30, 160))


def find_screws(rgb, intr, T, debug=None, extra_exclude=(), kind="bolt"):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32)
    bg = cv2.medianBlur(hsv[..., 2], 25).astype(np.float32)   # the mat's own brightness (grid lines are thin)
    mask = cv2.morphologyEx((v - bg > CONTRAST).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    keep = np.zeros_like(mask)
    u0, v0, u1, v1 = ROI
    keep[v0:v1, u0:u1] = 1
    for a, b, c, d in EXCLUDE + tuple(extra_exclude):
        keep[b:d, a:c] = 0
    mask *= keep
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (NUT["area"] if kind == "nut" else (MIN_AREA, MAX_AREA))[0] <= area <=                 (NUT["area"] if kind == "nut" else (MIN_AREA, MAX_AREA))[1]:
            continue
        pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        L, W = max(w, h), max(min(w, h), 1.0)
        if kind == "nut":
            if not NUT["len"][0] <= L <= NUT["len"][1] or W > NUT["max_w"] or L / W > NUT["max_aspect"]:
                continue
        elif not MIN_LEN_PX <= L <= MAX_LEN_PX or W > MAX_WIDTH_PX or L / W < MIN_ASPECT:
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


def perceive_screws(n_frames=5, kind="bolt"):
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
            ds, last = [], 0
            while len(ds) < 4:
                _, _, d, dt = kin.get_frames()
                if d is not None and dt != last:
                    last = dt
                    ds.append(kin.depth_to_meters(d))
            grab.depth = np.nanmedian(np.stack(ds), axis=0)
            return np.mean(frames, axis=0).astype(np.uint8), kin.intr
    rgb, intr = kinect_retry(grab, "screw detection")
    T = load_calib()
    tray = find_tray(rgb, grab.depth, intr, T)
    return find_screws(rgb, intr, T, extra_exclude=(tray[2],) if tray else (), kind=kind), rgb, tray


TRAY_ROI = (420, 150, 533, 365)    # u0, v0, u1, v1: where the tray stands on the mat (it is the only white thing there)


def find_tray(rgb, depth, intr, T):
    """The tray from the colour and depth images -> (floor points xy (N x 2), rim height z, pixel box) in the robot frame, or None.
    The tray is the biggest bright, unsaturated blob in TRAY_ROI; its depth points give the floor and the rim."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8)
    u0, v0, u1, v1 = TRAY_ROI
    roi = np.zeros_like(m)
    roi[v0:v1, u0:u1] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(m * roi)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 4000:
        return None
    vs, us = np.nonzero(lab == i)
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0)
    P = np.column_stack([(us[ok] - intr["cx"]) / intr["fx"] * z[ok], (vs[ok] - intr["cy"]) / intr["fy"] * z[ok], z[ok]])
    W = P @ T[:3, :3].T + T[:3, 3]
    rim = float(np.percentile(W[:, 2], 98))
    floor = W[W[:, 2] < np.percentile(W[:, 2], 50)]
    m_ = 30   # px of margin: the tray's walls and anything lying on its rim are not screws on the mat
    box = (int(st[i, cv2.CC_STAT_LEFT]) - m_, int(st[i, cv2.CC_STAT_TOP]) - m_,
           int(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH]) + m_, int(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT]) + m_)
    return floor[:, :2], rim, box


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
