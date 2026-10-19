"""Sim (MuJoCo so101_new_calib.xml, radians) <-> real (LeRobot calibrated, degrees / gripper 0-100).

    real = sign * scale * degrees(sim_rad) + offset        sim_deg = sign * (real - offset) / scale

Basis (TheRobotStudio/SO-ARM100 Simulation/SO101 README): the "new calib" model puts each joint's
virtual zero at the MIDDLE of its range, which is exactly LeRobot's calibrated zero
(deg = (ticks - (range_min+range_max)/2) * 360/4095). So the expected table is identity
(sign +1, offset 0) for the five arm joints. The sim joint ranges (pan +-110, lift +-100,
elbow +-96.8, wrist_flex +-95, roll -157..+163 deg) are close to half-spans of the real
calibration (pan 120, lift 104, elbow 98, wrist_flex 103), which supports offset ~ 0.

UNCERTAINTY (not verified on hardware -- arm was not allowed to move):
  * SIGNS are assumed +1 for all joints. The README only states the zero convention, not the
    axis direction; verify each joint by a small jog (see README.md) and flip `sign` if reversed.
  * OFFSETS are assumed 0. Real mid-range is only as good as how fully the joint was swept during
    calibration; a few degrees error per joint is plausible (the sim elbow range even has a
    documented 5 deg calibration fudge). Fix by `offset` after comparing poses.
  * GRIPPER: real is 0-100 (0 closed, 100 open); sim gripper is a hinge, -10..100 deg,
    larger angle = more open (checked geometrically in the MJCF: jaw-to-fixed-jaw distance grows
    with the angle). We map 1 real unit = 1 sim degree (sim 0 rad ~ closed, 100 deg ~ fully open).
    The real full-open travel is ~129 deg of servo rotation, so `scale` is a guess; the sim jaw
    angle is not a physical measurement of real jaw angle.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# joint: (sign, offset, scale)   real = sign*scale*sim_deg + offset
# EDIT HERE if a joint is reversed (sign -> -1) or offset by a constant (offset, in real units).
TABLE = {
    "shoulder_pan":  (+1, 0.0, 1.0),
    "shoulder_lift": (+1, 0.0, 1.0),
    "elbow_flex":    (+1, 0.0, 1.0),
    "wrist_flex":    (+1, 0.0, 1.0),
    "wrist_roll":    (+1, 0.0, 1.0),
    "gripper":       (+1, 0.0, 1.0),   # real 0-100 units per sim degree
}


# Joint zero offsets measured by calib_joints.py (the model angle that matches the real arm is
# real_to_sim(real) + d), folded into TABLE's offsets: real = sim_deg - d.
_FIT = Path(__file__).with_name("joint_offsets.json")
if _FIT.exists():
    for _j, _d in json.loads(_FIT.read_text()).items():
        _s, _o, _k = TABLE[_j]
        TABLE[_j] = (_s, _o - _s * _k * _d, _k)


def sim_to_real(sim_rad):
    """sim radians (dict or 6-seq in JOINTS order) -> real dict (deg; gripper 0-100)."""
    q = sim_rad if isinstance(sim_rad, dict) else dict(zip(JOINTS, sim_rad))
    return {j: TABLE[j][0] * TABLE[j][2] * math.degrees(q[j]) + TABLE[j][1] for j in JOINTS}


def real_to_sim(real):
    """real dict/6-seq (deg; gripper 0-100) -> sim radians dict."""
    r = real if isinstance(real, dict) else dict(zip(JOINTS, real))
    return {j: math.radians(TABLE[j][0] * (r[j] - TABLE[j][1]) / TABLE[j][2]) for j in JOINTS}


if __name__ == "__main__":
    import random
    for _ in range(100):
        q = {j: random.uniform(-1.5, 1.5) for j in JOINTS}
        b = real_to_sim(sim_to_real(q))
        assert all(abs(b[j] - q[j]) < 1e-9 for j in JOINTS)
    print("roundtrip ok")
