"""HSV color block detector. Config in perception/config.json (key "block").

Hue is OpenCV scale 0-179. If h_lo > h_hi the hue range wraps (e.g. red: 170..10).
Tune interactively:  python -m perception.block_detect --tune
"""
import argparse
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

DEFAULT_BLOCK = dict(
    h_lo=170, h_hi=12,        # wraps: bright red/orange
    s_lo=90, s_hi=255,
    v_lo=80, v_hi=255,
    min_area_px=40,           # ~3 cm block at 1 m is ~ 50-90 px across 15-30 px side
    max_area_px=4000,
    min_fill=0.45,            # contour area / bbox area (reject thin streaks)
    min_depth_m=0.4, max_depth_m=3.0,
    min_valid_depth_frac=0.3,  # fraction of blob pixels with valid depth
    roi=None,                 # optional [x0, y0, x1, y1] pixel search window
)


def load_config(path=CONFIG_PATH):
    """Detector settings: "block" from config.json, overlaid with objects[$KINECT_OBJECT] if set."""
    cfg = dict(DEFAULT_BLOCK)
    try:
        with open(path) as f:
            full = json.load(f)
        cfg.update(full.get("block", {}))
        obj = os.environ.get("KINECT_OBJECT")
        if obj:
            cfg.update(full["objects"][obj].get("detect", {}))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(block_cfg, path=CONFIG_PATH):
    try:
        with open(path) as f:
            full = json.load(f)
    except (OSError, ValueError):
        full = {}
    full["block"] = block_cfg
    with open(path, "w") as f:
        json.dump(full, f, indent=2)


@dataclass
class Detection:
    valid: bool
    reason: str = ""
    xyz_cam: np.ndarray = None      # metres, camera optical frame
    centroid_px: tuple = None       # (u, v)
    bbox: tuple = None              # (x, y, w, h)
    depth_m: float = float("nan")
    confidence: float = 0.0
    area_px: int = 0
    mask: np.ndarray = field(default=None, repr=False)


def hsv_mask(rgb, cfg):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = (cfg["s_lo"], cfg["v_lo"])
    lo_sv, hi_sv = (cfg["s_lo"], cfg["v_lo"]), (cfg["s_hi"], cfg["v_hi"])
    if cfg["h_lo"] <= cfg["h_hi"]:
        m = cv2.inRange(hsv, (cfg["h_lo"],) + lo_sv, (cfg["h_hi"],) + hi_sv)
    else:
        m = cv2.inRange(hsv, (cfg["h_lo"],) + lo_sv, (179,) + hi_sv) | \
            cv2.inRange(hsv, (0,) + lo_sv, (cfg["h_hi"],) + hi_sv)
    if cfg.get("roi"):
        x0, y0, x1, y1 = cfg["roi"]
        r = np.zeros_like(m)
        r[y0:y1, x0:x1] = 255
        m &= r
    for x0, y0, x1, y1 in cfg.get("exclude") or []:  # e.g. the parked arm's own yellow-green parts
        m[y0:y1, x0:x1] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def detect_block(rgb, depth_m, cfg=None, intr=None):
    """rgb: HxWx3 uint8 RGB. depth_m: HxW float metres (nan invalid), aligned to rgb
    (use DEPTH_REGISTERED). Returns Detection (valid=False with reason on failure)."""
    from .kinect import RGB_INTR, deproject
    cfg = cfg or load_config()
    intr = intr or RGB_INTR
    mask = hsv_mask(rgb, cfg)
    # close small gaps (labels, glare) so one block is one contour
    c = int(cfg.get("close_px", 5))  # wrapped/shiny objects break into pieces: join them
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((c, c), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if not cfg["min_area_px"] <= a <= cfg["max_area_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)  # rotation-invariant: a turned block still fills its rect
        fill = a / max(rw * rh, 1.0)
        if fill < cfg["min_fill"]:
            continue
        if best is None or a > best[0]:
            best = (a, c, (x, y, w, h), fill)
    if best is None:
        return Detection(False, "no blob in size range", mask=mask)
    a, c, bbox, fill = best
    blob = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(blob, [c], -1, 255, -1)
    inner = cv2.erode(blob, np.ones((3, 3), np.uint8))
    if inner.sum() == 0:
        inner = blob
    vals = depth_m[inner > 0]
    vals = vals[np.isfinite(vals)]
    n_blob = int((inner > 0).sum())
    valid_frac = len(vals) / max(n_blob, 1)
    m = cv2.moments(c)
    cu, cv_ = m["m10"] / m["m00"], m["m01"] / m["m00"]
    if valid_frac < cfg["min_valid_depth_frac"] or len(vals) < 5:
        return Detection(False, f"depth invalid in blob ({valid_frac:.0%} valid)",
                         centroid_px=(cu, cv_), bbox=bbox, area_px=int(a), mask=blob)
    z = float(np.median(vals))
    if not cfg["min_depth_m"] <= z <= cfg["max_depth_m"]:
        return Detection(False, f"depth {z:.2f} m out of range", centroid_px=(cu, cv_),
                         bbox=bbox, depth_m=z, area_px=int(a), mask=blob)
    # block top surface is closest to camera; median of a block blob is its top face.
    xyz = deproject(cu, cv_, z, intr)
    spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))
    conf = min(1.0, valid_frac) * min(1.0, fill / 0.8) * (1.0 if spread < 0.02 else 0.6)
    return Detection(True, "ok", xyz, (cu, cv_), bbox, z, float(conf), int(a), blob)


def draw(rgb, det):
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if det.bbox:
        x, y, w, h = det.bbox
        col = (0, 255, 0) if det.valid else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 1)
        txt = (f"{det.xyz_cam[0]:.3f},{det.xyz_cam[1]:.3f},{det.xyz_cam[2]:.3f} c={det.confidence:.2f}"
               if det.valid else det.reason)
        cv2.putText(out, txt, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return out


def tune():
    from .kinect import Kinect
    cfg = load_config()
    win = "tune (s=save, q=quit, click=print HSV)"
    cv2.namedWindow(win)
    for k, mx in (("h_lo", 179), ("h_hi", 179), ("s_lo", 255), ("s_hi", 255),
                  ("v_lo", 255), ("v_hi", 255), ("min_area_px", 2000)):
        cv2.createTrackbar(k, win, int(cfg[k]), mx, lambda v: None)
    last = {}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and "rgb" in last:
            print("HSV at", (x, y), cv2.cvtColor(last["rgb"][y:y+1, x:x+1], cv2.COLOR_RGB2HSV)[0, 0])
    cv2.setMouseCallback(win, on_mouse)
    with Kinect() as kin:
        kin.wait_for_frames()
        while True:
            rgb, _, d, _ = kin.get_frames()
            last["rgb"] = rgb
            for k in ("h_lo", "h_hi", "s_lo", "s_hi", "v_lo", "v_hi", "min_area_px"):
                cfg[k] = cv2.getTrackbarPos(k, win)
            det = detect_block(rgb, kin.depth_to_meters(d), cfg, kin.intr)
            vis = draw(rgb, det)
            mask = cv2.cvtColor(hsv_mask(rgb, cfg), cv2.COLOR_GRAY2BGR)
            cv2.imshow(win, np.hstack([vis, mask]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_config(cfg)
                print("saved", CONFIG_PATH)
            elif key in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--once", action="store_true", help="detect on live frames and print")
    a = ap.parse_args()
    if a.tune:
        tune()
    else:
        import time
        from .kinect import Kinect
        with Kinect() as kin:
            kin.wait_for_frames()
            for _ in range(10):
                rgb, _, d, _ = kin.get_frames()
                print(detect_block(rgb, kin.depth_to_meters(d), intr=kin.intr))
                time.sleep(0.2)
