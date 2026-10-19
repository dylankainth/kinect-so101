"""Kinect helpers shared by every project: restart-on-stall wrapper, one-shot block perception, pixel->table
projection, and saving a frame for the record."""
import logging
import time

import numpy as np

from .. import control

log = logging.getLogger("perception")


def kinect_retry(fn, what, tries=4):
    """Run a camera step; if the Kinect stalls or fails to open, restart it and try again."""
    from . import block_pose as bp
    last = None
    for k in range(tries):
        try:
            return fn()
        except (TimeoutError, RuntimeError) as e:
            last = e
            log.warning("Kinect trouble during %s (%s); restarting it (%d/%d)", what, e, k + 1, tries)
            try:
                bp.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3.0)
    raise last


def perceive_block():
    from .block_pose import get_block_xyz_robot, close

    def once():
        try:
            return get_block_xyz_robot(n_frames=15)
        finally:
            close()

    pose = kinect_retry(once, "block detection")
    if not pose.valid:
        raise SystemExit(f"block not found by the Kinect: {pose.reason}")
    log.info("block at %s, yaw %.0f deg (confidence %.2f, spread %.1f mm)", np.round(pose.xyz, 3),
             np.degrees(pose.yaw), pose.confidence, pose.spread_m * 1000)
    return np.asarray(pose.xyz, float), pose.yaw


def xy_from_pixel(intr, T, px, z_plane):
    """Where the camera ray through pixel px meets the horizontal plane z = z_plane (robot frame)."""
    ray_c = np.array([(px[0] - intr["cx"]) / intr["fx"], (px[1] - intr["cy"]) / intr["fy"], 1.0])
    ray = T[:3, :3] @ ray_c
    origin = T[:3, 3]
    return (origin + ray * (z_plane - origin[2]) / ray[2])[:2]


def snapshot(cam, name):
    """Save the current Kinect RGB frame to control/<name>.jpg (best effort)."""
    try:
        import cv2
        rgb = cam.wait_for_frames(2)[0]
        cv2.imwrite(str(control.DIR / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:  # noqa: BLE001
        log.warning("snapshot %s failed: %r", name, e)
