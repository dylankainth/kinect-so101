"""Arm motion helpers shared by every project: plan -> real keyframes, chunked moves, the stop/park
hold loop, aim correction from the hand-eye fit, and waiting for the servos to cool.

Constants here are properties of the arm and its tuning, not of a task. Project-specific values (grasp
heights, object profiles, success radius) belong in the project.
"""
import json
import logging
import math
import time

import numpy as np

from . import control
from .paths import CONTROL_DIR
from .robot.arm import SafeArm, JOINTS
from .robot.mapping import sim_to_real
from .sim.pick_place import PLAN_DT, GRIPPER_OPEN

log = logging.getLogger("motion")

REAL_SPEED_DEG_S = 25.0     # cap for the real arm (sim plan peaks ~40 deg/s)
SEGMENT_DEG = 25.0          # max joint change per move_to call when getting to/from the start pose
STREAM_HZ = 25.0            # goal positions per second streamed to the real arm
TIME_SCALE = 1.5            # real run takes 1.5x the sim plan's duration (peak ~27 deg/s)
REAL_GRIPPER_OPEN = 70.0    # real 0-100 opening used where the plan says "open"
GRIP_CLOSED_CMD = 1.0       # "closed" command; the jaws stall on the block and squeeze (run 8 slipped at 6)
# folded back against the shoulder/elbow stops: stable when the torque goes off
REST_POSE = {"shoulder_pan": -4.6, "shoulder_lift": -97.0, "elbow_flex": 94.0, "wrist_flex": 80.0,
             "wrist_roll": 0.0, "gripper": 30.0}

COOL_BELOW_C = 42      # start an attempt only when every servo is below this
COOL_MAX_WAIT_S = 1800


def wait_to_cool():
    """The gripper servo stalls against the block for the whole carry and warms up; let it cool first."""
    t0 = time.time()
    while True:
        with SafeArm(allow_writes=False) as arm:
            temps = arm.bus.sync_read("Present_Temperature", normalize=False, num_retry=3)
        hot = max(temps.values())
        if hot < COOL_BELOW_C:
            return
        if time.time() - t0 > COOL_MAX_WAIT_S:
            raise SystemExit(f"servos still at {hot} C after {COOL_MAX_WAIT_S // 60} min: stopping")
        status(f"cooling down: hottest servo {hot} C, waiting for < {COOL_BELOW_C} C")
        time.sleep(20)


def keyframes(plan):
    """Resample the 50 Hz sim plan onto the real streaming rate, slowed down by TIME_SCALE."""
    q = np.array([[wp[j] for j in JOINTS] for wp in plan])
    t_src = np.arange(len(plan)) * PLAN_DT * TIME_SCALE
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / STREAM_HZ)
    out = [dict(zip(JOINTS, (np.interp(t, t_src, q[:, i]) for i in range(len(JOINTS))))) for t in t_dst]
    frames = [sim_to_real(wp) for wp in out]
    open_sim = math.degrees(GRIPPER_OPEN)
    for f in frames:
        # sim "closed" squeezes past the jaw limit (real fully-closed is the soft limit); the real
        # jaw opens wider than the sim's open command for more margin around the block
        g = max(f["gripper"], 1.0)
        f["gripper"] = GRIP_CLOSED_CMD + (g - 1.0) * (REAL_GRIPPER_OPEN - GRIP_CLOSED_CMD) / (open_sim - 1.0)
    return frames


HANDEYE = CONTROL_DIR / "handeye.json"
ABORTED = False  # set by hold_until_user; read it as motion.ABORTED (a from-import would copy the value)


def aim(xyz, radial_offset, tangential=0.0):
    """Where the model must aim so the real jaws land on xyz.

    Uses the measured hand-eye fit (handeye_auto.py: real = A @ model + b, z offset dz) when it
    exists; otherwise pushes the point radially out by radial_offset (the arm lands short)."""
    p = np.array(xyz, float)
    if HANDEYE.exists():
        h = json.loads(HANDEYE.read_text())
        p[:2] = np.linalg.solve(np.array(h["A"]), p[:2] - np.array(h["b"]))
        p[2] -= h["dz"]
        return p
    r = np.linalg.norm(p[:2])
    if r > 1e-6:
        u = p[:2] / r
        p[:2] += radial_offset * u + tangential * np.array([-u[1], u[0]])
    return p


def segmented_move(arm, target, dry_run):
    """Go to a possibly far pose in chunks no bigger than SEGMENT_DEG per joint."""
    cur = arm.read_positions()
    big = max(abs(target[j] - cur[j]) for j in JOINTS)
    n = max(1, math.ceil(big / SEGMENT_DEG))
    for i in range(1, n + 1):
        arm.move_to({j: cur[j] + (target[j] - cur[j]) * i / n for j in JOINTS},
                    max_speed_deg_s=REAL_SPEED_DEG_S, dry_run=dry_run)
        if dry_run:
            break  # dry runs read the real (unmoved) pose each time; one chunk shows the gist


def status(msg):
    log.info(msg)
    control.beat(msg)


def hold_until_user(arm, rest, why):
    """After a STOP or error: keep holding, wait for PARK (fold to rest, limp) or LIMP from stop_gui."""
    global ABORTED
    ABORTED = True  # no further attempts after any stop or fault
    status(f"HOLDING ({why}). Press PARK or LIMP.")
    while True:
        control.beat()
        if control.is_set(control.LIMP):
            arm.disable_torque()
            status("LIMP: torque off")
            return
        if control.is_set(control.PARK):
            control.clear(control.STOP, control.PARK)
            status("PARK: folding back to rest")
            try:
                segmented_move(arm, rest, dry_run=False)
                time.sleep(0.5)
                arm.disable_torque()
                status("parked, torque off")
            except Exception as e:  # noqa: BLE001  (STOP pressed again, or a fault)
                return hold_until_user(arm, rest, f"park interrupted: {e}")
            return
        time.sleep(0.05)


def fixed_jaw_dir(plan):
    """Horizontal unit vector from the swinging jaw to the fixed jaw at the plan's grasp pose."""
    import mujoco
    from .sim import kinematics
    m = mujoco.MjModel.from_xml_path(kinematics.SCENE)
    d = mujoco.MjData(m)
    i = next(k for k, wp in enumerate(plan) if wp["gripper"] < plan[0]["gripper"] - 1e-6) - 1
    for j in JOINTS:
        d.qpos[m.joint(j).qposadr] = plan[i][j]
    mujoco.mj_forward(m, d)
    v = (d.geom_xpos[m.geom("fixed_pad").id] - d.geom_xpos[m.geom("moving_pad").id])[:2]
    return v / np.linalg.norm(v)
