"""Find every block on the table from depth alone (any colour), and read how high each one's top is.

    from core.perception.multi_block import perceive_blocks
    for b in perceive_blocks(block_size=0.03): print(b.xy, b.level)

Depth pixels are moved into the robot frame (z = 0 on the table) and the ones standing more than a
centimetre above the table are grouped. Each group's top face (its highest pixels) gives the block's x, y,
yaw and top height; top height / block size is the stack level (1 = on the table, 2 = on one block, ...).
Groups whose top face is not about one block big (two blocks touching, the arm, a hand) are rejected and
reported through `rejected`, never returned as blocks.
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .kinect import Kinect
from .util import kinect_retry

log = logging.getLogger("multi_block")

MIN_HEIGHT = 0.015        # m above the table for a pixel to count as "on a block"
MAX_HEIGHT = 0.15
WORKSPACE = (0.10, 0.45, -0.30, 0.30)   # x0, x1, y0, y1 in the robot frame; generous so a block is never clipped
                                        # (whether it can be reached is the planner's call)
TOP_BAND = 0.006          # m below the highest pixels that still belong to the top face
DEFAULT_ROI = (40, 155, 490, 445)       # pixel window (u0, v0, u1, v1) covering the mat


@dataclass(eq=False)   # identity equality: the fields are arrays
class BlockObs:
    xy: np.ndarray          # top-face centre, robot frame (m)
    top_z: float            # height of the top face above the table (m)
    yaw: float              # rad: world direction the jaws should close along (cube: any face axis, wrapped to
                            # (-pi/4, pi/4]; elongated block: its short axis, wrapped to (-pi/2, pi/2])
    level: int              # 1 = on the table, 2 = second in a stack, ...
    n_px: int
    size: tuple = (0.0, 0.0)  # top-face footprint (m), long side first
    period: float = np.pi / 2  # symmetry of yaw: pi/2 for a cube, pi for an elongated block

    def centre_z(self, block_size):
        return self.top_z - block_size / 2


def find_blocks(depth_m, intr, T, block_size=0.03, roi=DEFAULT_ROI, workspace=WORKSPACE, exclude=(),
                size_tol=None, rejected=None, footprint=None):
    """depth_m: HxW metres (nan invalid), aligned to the camera `intr`. T: 4x4 robot <- camera.
    exclude: iterable of (x, y, radius) discs in the robot frame to ignore (e.g. where the parked arm sits).
    block_size: height of one block (a layer of the stack). footprint: (long, short) top-face size in m;
    default is a cube of block_size. rejected: optional list that receives (xy, reason) for groups that
    were not blocks."""
    fl, fs = footprint or (block_size, block_size)
    elongated = fl > 1.25 * fs
    tol = size_tol or ((0.75, 1.3) if elongated else (0.6, 1.45))
    h, w = depth_m.shape
    u0, v0, u1, v1 = roi
    uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
    z = depth_m[v0:v1, u0:u1].astype(np.float64)
    ok = np.isfinite(z) & (z > 0)
    z = np.where(ok, z, 0.0)
    Pc = np.stack([(uu - intr["cx"]) * z / intr["fx"], (vv - intr["cy"]) * z / intr["fy"], z], axis=-1)
    P = Pc @ T[:3, :3].T + T[:3, 3]
    x0, x1, y0, y1 = workspace
    m = ok & (P[..., 2] > MIN_HEIGHT) & (P[..., 2] < MAX_HEIGHT) & \
        (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    for ex, ey, er in exclude:
        m &= np.hypot(P[..., 0] - ex, P[..., 1] - ey) > er
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(m, connectivity=8)
    out = []
    for i in range(1, n):
        pts = P[lab == i]
        if len(pts) < 25:
            continue
        ztop = float(np.percentile(pts[:, 2], 95))
        top = pts[pts[:, 2] >= ztop - TOP_BAND]       # side faces seen at an angle are lower: drop them
        if len(top) < 15:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
        size = (max(rw, rh) / 1000, min(rw, rh) / 1000)
        centre = np.array([cx, cy]) / 1000
        if not (tol[0] * fs <= size[1] <= tol[1] * fs and tol[0] * fl <= size[0] <= tol[1] * fl):
            if rejected is not None:
                rejected.append((centre, f"top face {size[0]*100:.1f} x {size[1]*100:.1f} cm is not one "
                                         f"{fl*100:.0f} x {fs*100:.0f} cm block"))
            continue
        a = np.radians(ang)
        if elongated:
            if rw > rh:   # OpenCV's angle is that of the first side: the closing axis is the other (short) one
                a += np.pi / 2
            period = np.pi
        else:
            period = np.pi / 2
        yaw = float((a + period / 2) % period - period / 2)
        z_top = float(np.median(top[:, 2]))
        out.append(BlockObs(centre, z_top, yaw, max(1, int(round(z_top / block_size))), len(pts), size, period))
    return sorted(out, key=lambda b: (b.xy[0], b.xy[1]))


def perceive_blocks(block_size=0.03, n_frames=6, footprint=None, **kw):
    """Grab a few depth frames from the Kinect (opened and closed here), take the per-pixel median, and
    return the BlockObs list. Extra keyword arguments go to find_blocks."""
    from .calibrate import load_calib

    def grab():
        frames = []
        with Kinect() as kin:
            kin.wait_for_frames(8)
            last = 0.0
            while len(frames) < n_frames:
                _, _, d, dt = kin.get_frames()
                if d is not None and dt != last:
                    last = dt
                    frames.append(kin.depth_to_meters(d))
            return np.nanmedian(np.stack(frames), axis=0), kin.intr

    depth, intr = kinect_retry(grab, "block detection")
    return find_blocks(depth, intr, load_calib(), block_size, footprint=footprint, **kw)
