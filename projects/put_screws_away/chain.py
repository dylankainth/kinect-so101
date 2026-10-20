"""Pick several screws in one arm session: no camera, torque or fold-to-rest between them.

core.grasp.run_arm does one pick and then folds to rest, switches torque off, and the caller re-opens the Kinect to
check. A plan starts and ends at the home pose, so the next pick can follow straight away; the camera only looks once
when the batch is done (main.py then retries whatever is still on the mat).
"""
import logging

import numpy as np

from core import control
from core.grasp import GraspParams, split_at_lift
from core.motion import GRIP_CLOSED_CMD, REST_POSE, STREAM_HZ, hold_until_user, segmented_move, status
from core.robot.arm import ARM_JOINTS, JOINTS, SafeArm

log = logging.getLogger("chain")

KEEP_DWELL = 3   # identical consecutive samples kept (the plan stands still for 1-2 s around each step; not needed here)


def trim_dwell(frames, keep=KEEP_DWELL):
    out, same = [], 0
    for f in frames:
        same = same + 1 if out and all(abs(f[j] - out[-1][j]) < 1e-9 for j in JOINTS) else 0
        if same < keep:
            out.append(f)
    return out


def run_chain(batch, params=None, close_time_s=0.8):
    """batch: list of (label, frames). Returns the labels whose pick ran to the end ('aborted' stops the rest)."""
    p = params or GraspParams(grasp_min=0.0)
    ran = []
    with SafeArm(allow_writes=True) as arm:
        rest = dict(REST_POSE)

        def stop_requested():
            control.beat()
            return control.is_set(control.STOP) or control.is_set(control.LIMP)
        arm.stop_requested = stop_requested
        arm.enable_torque()
        try:
            for label, frames in batch:
                frames = trim_dwell(frames)
                close = next(i for i, f in enumerate(frames) if f["gripper"] < frames[0]["gripper"] - 1e-6)
                lift = split_at_lift(frames)
                cur = arm.read_positions()
                if max(abs(frames[0][j] - cur[j]) for j in ARM_JOINTS) > 6.0:
                    segmented_move(arm, frames[0], dry_run=False)
                status(f"{label}: descending")
                arm.follow(frames[:close], rate_hz=STREAM_HZ)
                n = max(2, int(close_time_s * STREAM_HZ))
                g0 = frames[close - 1]["gripper"]
                sweep = [frames[close - 1] | {"gripper": g0 + (GRIP_CLOSED_CMD - g0) * (k + 1) / n} for k in range(n)]
                arm.follow(sweep, rate_hz=STREAM_HZ)
                grip = arm.settle_gripper(GRIP_CLOSED_CMD)
                hold = max(GRIP_CLOSED_CMD, grip - p.grip_squeeze)
                status(f"{label}: jaws at {grip:.0f}/100, carrying")
                carry = [f | {"gripper": hold} if f["gripper"] <= GRIP_CLOSED_CMD + 1e-6 else f for f in frames[lift:]]
                arm.follow(carry, rate_hz=STREAM_HZ)
                ran.append(label)
            status("batch done: folding back to rest")
            segmented_move(arm, rest, dry_run=False)
            arm.disable_torque()
            status("parked, torque off")
        except BaseException as e:  # noqa: BLE001  STOP, safety abort, Ctrl+C, anything
            log.error("motion aborted: %r", e)
            try:
                arm.stop()
            except Exception:  # noqa: BLE001
                pass
            hold_until_user(arm, rest, str(e) or type(e).__name__)
            ran.append("aborted")
    return ran
