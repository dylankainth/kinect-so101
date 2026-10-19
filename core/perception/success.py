"""Success check: did the block end up in the target region, resting on the table?

Regions are in the robot frame (m). Rect: (x_min, x_max, y_min, y_max). Circle: (cx, cy, r).
The block CENTRE must be inside the region shrunk by `margin` (so the block is mostly
inside), and its centre height within z_tol of block_size/2 above the table (table is
z = table_z, default 0 = robot base plane).

    from core.perception.success import Region, check_success
    ok, info = check_success(Region.rect(0.15, 0.30, -0.20, -0.05))
"""
from dataclasses import dataclass

import numpy as np

from .block_pose import _settings, get_block_xyz_robot


@dataclass
class Region:
    kind: str
    params: tuple

    @classmethod
    def rect(cls, x_min, x_max, y_min, y_max):
        return cls("rect", (x_min, x_max, y_min, y_max))

    @classmethod
    def circle(cls, cx, cy, r):
        return cls("circle", (cx, cy, r))

    def signed_margin(self, x, y):
        """Distance (m) the point lies inside the region; negative = outside."""
        if self.kind == "rect":
            x0, x1, y0, y1 = self.params
            return min(x - x0, x1 - x, y - y0, y1 - y)
        cx, cy, r = self.params
        return r - float(np.hypot(x - cx, y - cy))


def evaluate(xyz, region, block_size=0.03, table_z=0.0, z_tol=0.015, margin=0.0):
    """Pure function: block centre xyz (robot frame) -> (success, info dict)."""
    x, y, z = map(float, xyz)
    inside = region.signed_margin(x, y)
    z_err = z - (table_z + block_size / 2.0)
    in_region = inside >= margin
    on_table = abs(z_err) <= z_tol
    return bool(in_region and on_table), dict(
        in_region=bool(in_region), on_table=bool(on_table),
        margin_m=inside, z_err_m=z_err, xyz=(x, y, z))


def check_success(region, pose=None, margin=0.0, z_tol=0.015, table_z=0.0, **pose_kw):
    """Fetch the block pose (unless given) and evaluate. Invalid detection => (False, info
    with detected=False): a block that is not seen (e.g. still in the gripper or hidden)
    never counts as success."""
    pose = pose if pose is not None else get_block_xyz_robot(**pose_kw)
    if not pose.valid:
        return False, dict(detected=False, reason=pose.reason)
    ok, info = evaluate(pose.xyz, region, _settings()["block_size_m"], table_z, z_tol, margin)
    info.update(detected=True, confidence=pose.confidence)
    return ok, info
