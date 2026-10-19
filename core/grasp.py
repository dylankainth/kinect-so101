"""Run one planned pick-and-place on the real arm, with the grasp checks that make it reliable.

run_arm() streams the keyframes of a plan through SafeArm: descend, close slowly, wait for the jaws to stop
and read the gripper (about 7 = nothing, 20+ = holding), lift a little and check the block really left the
table, carry, release, fold back to rest and drop torque. STOP/PARK/LIMP from core.control are honoured.
It returns 'done', 'slipped', 'missed' or 'aborted'. Without real=True it only checks the trajectory.

Per-object tunables travel in GraspParams so any project can set them without touching this module.
"""
import json
import logging
import time
from dataclasses import dataclass

import numpy as np

from . import control
from .motion import (GRIP_CLOSED_CMD, REST_POSE, STREAM_HZ, hold_until_user, segmented_move, status)
from .paths import CALIB_PATH
from .perception.kinect import Kinect
from .perception.util import snapshot
from .robot.arm import SafeArm, JOINTS

log = logging.getLogger("grasp")


@dataclass
class GraspParams:
    close_time_s: float = 1.5      # s for the slow closing sweep
    grasp_min: float = 12.0        # jaws stopped above this (0-100) => something is between them
    tactile: bool = True           # False: decide grasp success by camera after a short lift (small objects)
    grip_squeeze: float = 8.0      # 0-100 units commanded inside the contact point while carrying
    lift_check_steps: int = 40     # carry steps (25 Hz) before pausing to check the block left the table


def split_at_lift(frames):
    """Index where the arm starts lifting after the gripper has closed on the block."""
    k = next(i for i, f in enumerate(frames) if f["gripper"] <= GRIP_CLOSED_CMD + 1e-6)
    return next(i for i in range(k + 1, len(frames))
                if max(abs(frames[i][j] - frames[k][j]) for j in JOINTS[:5]) > 0.05)


def block_still_on_table(cam, block_xyz, block_top=0.03):
    """After the lift: look at the red pixels near where the block was and measure their height.
    Mostly at table height => still on the table (True); clearly raised => lifted (False).
    Too few red pixels with valid depth => can't tell; trust the gripper (False)."""
    from .perception.block_detect import hsv_mask, load_config
    from .perception.kinect import deproject
    T = np.array(json.loads((CALIB_PATH).read_text())["T_robot_cam"])
    R, t = T[:3, :3], T[:3, 3]
    c = R.T @ (np.asarray(block_xyz, float) - t)
    u0, v0 = cam.intr["fx"] * c[0] / c[2] + cam.intr["cx"], cam.intr["fy"] * c[1] / c[2] + cam.intr["cy"]
    zs = []
    for _ in range(5):
        time.sleep(0.07)  # ~2 new frames each time
        rgb, _, d, _ = cam.wait_for_frames(2)
        dm = cam.depth_to_meters(d)
        v, u = np.nonzero(hsv_mask(rgb, load_config()))
        near = (u - u0) ** 2 + (v - v0) ** 2 < 80 ** 2
        u, v = u[near], v[near]
        z = dm[v, u]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() >= 10:
            P = np.asarray(deproject(u[ok].astype(float), v[ok].astype(float), z[ok], cam.intr))
            P = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P
            zs.append(np.median((R @ P.T).T[:, 2] + t[2]))
    if len(zs) < 2:
        log.info("lift check: block not measurable (hidden), trusting the gripper")
        return False
    z = float(np.median(zs))
    log.info("lift check: red pixels near the pick spot at median z %.3f m", z)
    return z < block_top + 0.012


def run_arm(frames, real, block_xyz=None, params=None):
    """Returns 'done', 'slipped', 'missed' (jaws closed on nothing; backed off) or 'aborted'."""
    p = params or GraspParams()
    with SafeArm(allow_writes=real) as arm:
        if arm.mismatch:
            log.warning("motor registers differ from the calibration file on %s (using the file)",
                        ", ".join(arm.mismatch))
        start = arm.read_positions()
        log.info("start pose: %s", {j: round(v, 1) for j, v in start.items()})
        rest = dict(REST_POSE)  # always park to the same folded pose (resting on the joint stops)
        if real:
            def stop_requested():
                control.beat()
                return control.is_set(control.STOP) or control.is_set(control.LIMP)
            arm.stop_requested = stop_requested
            arm.enable_torque()
            lift = split_at_lift(frames)
            close = next(i for i, f in enumerate(frames) if f["gripper"] < frames[0]["gripper"] - 1e-6)
            result = "aborted"
            for _k in range(4):   # the Kinect sometimes needs a second try right after being closed
                try:
                    cam = Kinect().start()
                    break
                except RuntimeError as e:
                    log.warning("Kinect did not open (%s); retrying", e)
                    time.sleep(2.0)
            else:
                raise RuntimeError("Kinect would not open for the run")
            try:
                status("moving to start pose")
                segmented_move(arm, frames[0], dry_run=False)
                status("descending to the block")
                arm.follow(frames[:close], rate_hz=STREAM_HZ)
                time.sleep(0.4)                                  # hold still for the camera
                snapshot(cam, "pre_close")
                status("closing on the block")
                # slow sweep so the swinging jaw pushes the block into the fixed jaw, not over
                n = int(p.close_time_s * STREAM_HZ)
                g0 = frames[close - 1]["gripper"]
                sweep = [frames[close - 1] | {"gripper": g0 + (GRIP_CLOSED_CMD - g0) * (k + 1) / n} for k in range(n)]
                arm.follow(sweep, rate_hz=STREAM_HZ)
                grip = arm.settle_gripper(GRIP_CLOSED_CMD)   # wait for the jaws to actually stop
                snapshot(cam, "post_close")
                if grip >= p.grasp_min or not p.tactile:
                    status(f"jaws stopped at {grip:.0f}/100 on something: lifting")
                    # hold just inside the contact point: firm, but not pinned at stall
                    hold = max(GRIP_CLOSED_CMD, grip - p.grip_squeeze)
                    carry = [f | {"gripper": hold} if f["gripper"] <= GRIP_CLOSED_CMD + 1e-6 else f
                             for f in frames[lift:]]
                    up = carry[:p.lift_check_steps]
                    arm.follow(up, rate_hz=STREAM_HZ)
                    time.sleep(0.3)
                    snapshot(cam, "lifted")
                    held = arm.read_positions()["gripper"]
                    if held >= p.grasp_min:      # jaws still propped open: it is in there
                        lost = False
                    elif p.tactile:
                        lost = True
                    else:  # small objects can read ~closed even when held: ask the camera
                        lost = block_still_on_table(cam, block_xyz, block_top=2 * block_xyz[2] + 0.004)
                    if lost:
                        status(f"jaws at {held:.0f}/100 after the lift and the object is not with them: reopening")
                        arm.follow(up[::-1], rate_hz=STREAM_HZ)
                        arm.follow(sweep[::-1] + frames[:close][::-1], rate_hz=STREAM_HZ)
                        result = "slipped" if p.tactile else "missed"
                    else:
                        status(f"holding the block (jaws at {held:.0f}/100): carrying to target")
                        onward = carry[p.lift_check_steps:]
                        rel = next((i for i, f in enumerate(onward) if f["gripper"] > hold + 1e-6), None)
                        if rel is None:
                            arm.follow(onward, rate_hz=STREAM_HZ)
                        else:   # look at the table just before the jaws open and just after, to see the release
                            arm.follow(onward[:rel], rate_hz=STREAM_HZ)
                            time.sleep(0.3)
                            snapshot(cam, "pre_release")
                            opened = next((i for i in range(rel, len(onward))
                                           if onward[i]["gripper"] >= max(f["gripper"] for f in onward) - 1e-6), rel)
                            stop_at = min(len(onward), opened + 5)
                            arm.follow(onward[rel:stop_at], rate_hz=STREAM_HZ)
                            time.sleep(0.3)
                            snapshot(cam, "post_release")
                            arm.follow(onward[stop_at:], rate_hz=STREAM_HZ)
                            time.sleep(0.3)
                            snapshot(cam, "post_retreat")
                        result = "done"
                else:
                    status(f"missed (jaws closed to {grip:.0f}/100): reopening and backing off")
                    arm.follow(sweep[::-1] + frames[:close][::-1], rate_hz=STREAM_HZ)
                    result = "missed"
                status("folding back to rest")
                segmented_move(arm, rest, dry_run=False)
                time.sleep(0.5)
                arm.disable_torque()
                status(f"{result}, torque off")
            except BaseException as e:  # noqa: BLE001  STOP, safety abort, Ctrl+C, anything
                log.error("motion aborted: %r", e)
                try:
                    arm.stop()
                except Exception:  # noqa: BLE001
                    pass
                hold_until_user(arm, rest, str(e) or type(e).__name__)
                result = "aborted"
            finally:
                cam.stop()
                if arm.trace:  # commanded vs measured joints, for sim-to-real analysis
                    t0 = arm.trace[0][0]
                    (control.DIR / "trace.json").write_text(json.dumps(
                        [dict(t=t - t0, goal=g, pos=p) for t, g, p in arm.trace]))
            return result
        segmented_move(arm, frames[0], dry_run=True)
        if not real:
            # The arm hasn't moved, so check the trajectory itself: soft limits and step sizes.
            clamped = [arm.clamp(f) for f in frames]
            worst = max(abs(c[j] - f[j]) for c, f in zip(clamped, frames) for j in JOINTS)
            step = max(abs(b[j] - a[j]) for a, b in zip(frames, frames[1:]) for j in JOINTS)
            log.info("[DRY] %d waypoints; largest clamp %.1f deg, largest step %.1f deg (limit %.0f)",
                     len(frames), worst, step, arm.max_jump_deg)
            return
