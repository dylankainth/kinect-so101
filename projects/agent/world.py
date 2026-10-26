"""What the camera sees, in robot coordinates: one captured scene and the 3D facts about a named object in it.

A grounder (grounding.py) says where an object is in the colour image (a pixel box). Here the Kinect depth under that
box, turned into the robot frame through the calibration, gives the object's position, height, footprint and direction.
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core.perception.calibrate import load_calib
from core.perception.kinect import Kinect
from core.perception.util import kinect_retry, xy_from_pixel

MIN_HEIGHT = 0.008      # m above the table: lower than this is the mat, not the object
MAX_HEIGHT = 0.30
MIN_POINTS = 40
TOP_BAND = 0.006       # m below the highest pixels that still belong to the top face (side faces seen at an angle are lower)


@dataclass
class Scene:
    rgb: np.ndarray       # HxWx3 uint8, RGB
    depth: np.ndarray     # HxW metres, nan where the Kinect has no reading
    intr: dict
    T: np.ndarray         # camera -> robot

    def points(self):
        """HxWx3 robot-frame points (nan where there is no depth)."""
        h, w = self.depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = self.depth
        cam = np.stack([(u - self.intr["cx"]) / self.intr["fx"] * z, (v - self.intr["cy"]) / self.intr["fy"] * z, z], -1)
        return cam @ self.T[:3, :3].T + self.T[:3, 3]


@dataclass
class Obj:
    id: str
    label: str
    box: tuple            # u0, v0, u1, v1
    xy: np.ndarray        # centre of the footprint, robot frame, m
    top_z: float          # highest point, m above the table
    length: float         # footprint, m (long side)
    width: float          # footprint, m (short side)
    yaw: float            # direction of the long side, rad, robot frame
    flat: bool = False    # too little depth to measure (a thin object): position only, from the table plane
    base_z: float = 0.0   # height of what it rests on (0 = the table); top_z - base_z is its own thickness

    def describe(self):
        return (f"{self.id} ({self.label}): at x={self.xy[0]:.3f} y={self.xy[1]:+.3f} m, top face {self.length * 100:.1f} x "
                f"{self.width * 100:.1f} cm, top {self.top_z * 100:.1f} cm above the table"
                + (f", resting on something {self.base_z * 100:.1f} cm high" if self.base_z else "")
                + (", very thin" if self.flat else ""))


def capture(n_frames=5):
    def grab():
        with Kinect() as kin:
            kin.wait_for_frames(8)
            time.sleep(1.0)
            frames, ds, last = [], [], 0
            while len(frames) < n_frames:
                time.sleep(0.15)
                f, _ = kin.get_rgb()
                frames.append(f.astype(np.float32))
            while len(ds) < 4:
                _, _, d, dt = kin.get_frames()
                if d is not None and dt != last:
                    last = dt
                    ds.append(kin.depth_to_meters(d))
            return np.mean(frames, axis=0).astype(np.uint8), np.nanmedian(np.stack(ds), axis=0), kin.intr
    rgb, depth, intr = kinect_retry(grab, "scene capture")
    return Scene(rgb, depth, intr, load_calib())


def measure(scene, label, box, obj_id=None, pts=None, mask=None):
    """3D facts about the object inside the pixel box, or None if nothing stands there. mask: HxW bool, only these pixels.
    Position, size and direction come from the top face alone."""
    u0, v0, u1, v1 = (int(round(x)) for x in box)
    h, w = scene.depth.shape
    u0, u1, v0, v1 = max(u0, 0), min(u1, w), max(v0, 0), min(v1, h)
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    pts = scene.points() if pts is None else pts
    P = pts[v0:v1, u0:u1]
    ok = np.isfinite(P).all(-1) & (P[..., 2] > MIN_HEIGHT) & (P[..., 2] < MAX_HEIGHT)
    if mask is not None:
        ok &= mask[v0:v1, u0:u1]
    P = P[ok]
    oid = obj_id or label
    if len(P) < MIN_POINTS:   # thin: take the box centre on the table plane
        c = xy_from_pixel(scene.intr, scene.T, ((u0 + u1) / 2, (v0 + v1) / 2), 0.002)
        return Obj(oid, label, (u0, v0, u1, v1), c, 0.004, 0.0, 0.0, 0.0, flat=True)
    ztop = float(np.percentile(P[:, 2], 95))
    top = P[P[:, 2] >= ztop - TOP_BAND]
    mm = (top[:, :2] * 1000).astype(np.float32)
    keep = np.linalg.norm(mm - np.median(mm, axis=0), axis=1) < 150
    mm = mm[keep]
    if len(mm) < MIN_POINTS // 2:
        return None
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(mm)
    L, W = max(rw, rh) / 1000, min(rw, rh) / 1000
    th = np.radians(ang if rw >= rh else ang + 90)
    return Obj(oid, label, (u0, v0, u1, v1), np.array([cx, cy]) / 1000, float(np.median(top[:, 2])), float(L), float(W),
               float(th))


def annotate(rgb, objs, path=None):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for o in objs:
        u0, v0, u1, v1 = map(int, o.box)
        cv2.rectangle(img, (u0, v0), (u1, v1), (0, 0, 255), 1)
        cv2.putText(img, o.id, (u0, max(v0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if path:
        cv2.imwrite(str(path), img)
    return img
