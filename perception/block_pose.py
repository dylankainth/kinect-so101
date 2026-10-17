"""Block position in the robot base frame (x forward, y left, z up, origin at arm base).

    from perception.block_pose import get_block_xyz_robot
    r = get_block_xyz_robot()
    if r.valid: print(r.xyz)         # np.array([x, y, z]) metres, z = block CENTRE height

The depth camera sees the block's top face, so the measured point is shifted down by
block_size/2 to give the cube centre (assumes a roughly top-down camera). Settings in
config.json under "block_pose": {"block_size_m": 0.03, "n_frames": 10, "min_detections": 4,
"max_spread_m": 0.012}. Requires calib.json (python -m perception.calibrate ...).
Opens the Kinect on first call and keeps it open (closed at interpreter exit).
"""
import atexit
import json
import os
import time
from dataclasses import dataclass

import cv2
from perception.kinect import deproject
import numpy as np

from .block_detect import CONFIG_PATH, detect_block, load_config
from .calibrate import load_calib
from .kinect import Kinect

_kin = None
_T = None


@dataclass
class BlockPose:
    valid: bool
    xyz: np.ndarray = None       # robot frame, block centre (m)
    confidence: float = 0.0
    n_detections: int = 0
    n_frames: int = 0
    spread_m: float = float("nan")   # max std-dev across detections (stationary => ~mm)
    reason: str = ""
    pixel: tuple = None          # median centroid pixel (u, v)
    yaw: float = 0.0             # rad, block face orientation about robot z, in (-pi/4, pi/4]

    def __iter__(self):          # xyz, valid = get_block_xyz_robot()
        return iter((self.xyz, self.valid))


def _settings():
    s = dict(block_size_m=0.03, n_frames=10, min_detections=4, max_spread_m=0.012)
    try:
        with open(CONFIG_PATH) as f:
            full = json.load(f)
        s.update(full.get("block_pose", {}))
        obj = os.environ.get("KINECT_OBJECT")
        if obj:
            s.update(full["objects"][obj].get("pose", {}))
    except (OSError, ValueError):
        pass
    return s


def _get_kinect():
    global _kin
    if _kin is None:
        _kin = Kinect().start()
        atexit.register(close)
    return _kin


def close():
    global _kin
    if _kin is not None:
        _kin.stop()
        _kin = None


def reload_calibration():
    global _T
    _T = load_calib()


def get_block_xyz_robot(n_frames=None, kinect=None, T_robot_cam=None, timeout=5.0):
    s = _settings()
    n_frames = n_frames or s["n_frames"]
    kin = kinect or _get_kinect()
    if T_robot_cam is None:
        if _T is None:
            reload_calibration()
        T_robot_cam = _T
    cfg = load_config()
    kin.wait_for_frames(timeout)

    pts, confs, pix, reasons, yaws = [], [], [], [], []
    last_t, seen, t0 = 0.0, 0, time.time()
    while seen < n_frames and time.time() - t0 < timeout:
        rgb, rt, d, dt = kin.get_frames()
        if rgb is None or d is None or dt == last_t:
            time.sleep(0.005)
            continue
        last_t = dt
        seen += 1
        det = detect_block(rgb, kin.depth_to_meters(d), cfg, kin.intr)
        if not det.valid:
            reasons.append(det.reason)
            continue
        p = (T_robot_cam @ np.r_[det.xyz_cam, 1.0])[:3]
        p[2] -= s["block_size_m"] / 2.0     # top-face point -> cube centre
        pts.append(p)
        y = _top_face_yaw(det.mask, kin.depth_to_meters(d), kin.intr, T_robot_cam, s.get("yaw_period", np.pi / 2))
        if y is not None:
            yaws.append(y)
        confs.append(det.confidence)
        pix.append(det.centroid_px)
    if len(pts) < s["min_detections"]:
        why = reasons[-1] if reasons else "too few detections"
        return BlockPose(False, None, 0.0, len(pts), seen, reason=why)
    pts = np.array(pts)
    # drop outlier frames (e.g. a stale frame right after the Kinect starts) before judging stability
    keep = np.linalg.norm(pts - np.median(pts, axis=0), axis=1) < 0.02
    if keep.sum() >= s["min_detections"]:
        pts = pts[keep]
        yaws = [y for y, k in zip(yaws, keep) if k] if len(yaws) == len(keep) else yaws
    xyz = np.median(pts, axis=0)
    spread = float(pts.std(axis=0).max())
    ok = spread <= s["max_spread_m"]
    # a cube's yaw repeats every 90 deg: average on the 4x circle
    m = 2 * np.pi / s.get("yaw_period", np.pi / 2)  # average on the circle of the object's symmetry
    yaw = float(np.angle(np.mean(np.exp(1j * m * np.array(yaws)))) / m) if yaws else 0.0
    return BlockPose(ok, xyz, float(np.mean(confs)) * len(pts) / seen,
                     len(pts), seen, spread, "ok" if ok else "block moving / unstable",
                     tuple(np.median(pix, axis=0)), yaw)


def _top_face_yaw(blob, depth_m, intr, T, period=np.pi / 2):
    """Orientation of the block's top face in the robot frame from its blob pixels."""
    if blob is None:
        return None
    v, u = np.nonzero(blob)
    z = depth_m[v, u]
    ok = np.isfinite(z) & (z > 0)
    if ok.sum() < 20:
        return None
    P = np.asarray(deproject(u[ok].astype(float), v[ok].astype(float), z[ok], intr))
    P = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P
    Q = (T[:3, :3] @ P.T).T + T[:3, 3]
    top = Q[Q[:, 2] > np.percentile(Q[:, 2], 40)]  # drop side faces seen at an angle
    (_, _), (w, h), ang = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
    a = np.radians(ang)
    if period > np.pi / 2 + 1e-6 and w > h:  # elongated: report the SHORT axis (the jaws close across it)
        a += np.pi / 2
    return float((a + period / 2) % period - period / 2)


if __name__ == "__main__":
    for _ in range(5):
        r = get_block_xyz_robot()
        print(r.valid, None if r.xyz is None else np.round(r.xyz, 3), r.reason, r.n_detections, "/", r.n_frames)
