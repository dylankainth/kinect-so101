"""Domain-randomised evaluation of the scripted pick-and-place.   python -m core.sim.eval [--n 50] [--seed 0]

Per trial: block xy jitter +-3 cm around (0.20, +0.12) and uniform yaw; cube half-size, mass +-20%;
cube friction 0.5-1.0 and fingertip-pad friction 1.0-1.5; table friction +-20%; actuator kp (and
matching bias) +-20%; 1-3 control-step (20-60 ms) command latency; 3 mm Gaussian perception noise on the
block position handed to the planner (the true cube pose is used for the simulation).
"""
import argparse

import mujoco
import numpy as np

from .kinematics import SCENE
from .pick_place import plan_pick_place, execute_plan

BLOCK0 = np.array([0.20, 0.12])
TARGET = np.array([0.20, -0.12])


def randomize(rng, nominal_xml=SCENE):
    m = mujoco.MjModel.from_xml_path(nominal_xml)
    s = rng.uniform(0.8, 1.2)  # cube size scale
    half = 0.015 * s
    g = m.geom("cube_geom")
    g.size[:] = half
    g.rbound[:] = half * np.sqrt(3)
    b = m.body("cube")
    mass = 0.02 * rng.uniform(0.8, 1.2)
    b.mass[0] = mass
    b.inertia[:] = mass * (2 * half) ** 2 / 6  # solid cube
    g.friction[0] = rng.uniform(0.5, 1.0)
    for pad in ("fixed_pad", "moving_pad"):
        m.geom(pad).friction[0] = rng.uniform(1.0, 1.5)
    m.geom("table").friction[0] *= rng.uniform(0.8, 1.2)
    for i in range(m.nu):
        k = rng.uniform(0.8, 1.2)
        m.actuator_gainprm[i, 0] *= k
        m.actuator_biasprm[i, 1] *= k
    return m, half


def run_trial(rng):
    m, half = randomize(rng)
    xy = BLOCK0 + rng.uniform(-0.03, 0.03, 2)
    yaw = rng.uniform(-np.pi, np.pi)
    latency = int(rng.integers(1, 4))
    block = np.array([xy[0], xy[1], half])
    target = np.array([TARGET[0], TARGET[1], half])
    seen = block + np.append(rng.normal(0, 0.003, 2), 0.0)  # perception noise (xy only)
    plan = plan_pick_place(seen, target, block_yaw=yaw + rng.normal(0, 0.05))
    res = execute_plan(plan, block, target, block_yaw=yaw, latency_steps=latency, model=m)
    res.update(xy=xy, yaw=yaw, latency=latency, size=2 * half)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    wins, fails = 0, []
    for i in range(a.n):
        try:
            r = run_trial(rng)
        except ValueError as e:
            r = dict(success=False, err_xy=float("nan"), on_table=False, xy=None, yaw=0, latency=0, size=0, note=str(e))
        wins += r["success"]
        if not r["success"]:
            fails.append((i, r))
        print(f"trial {i:2d} {'OK  ' if r['success'] else 'FAIL'} err={r['err_xy']*100:5.2f} cm "
              f"lat={r['latency']} size={r['size']*100:.1f} cm yaw={np.degrees(r['yaw']):6.1f}")
    print(f"\nSUCCESS {wins}/{a.n} = {100*wins/a.n:.0f}%")
    for i, r in fails:
        print(" failed trial", i, {k: v for k, v in r.items() if k in ("cube_xyz", "err_xy", "on_table", "xy", "yaw", "latency", "size", "note")})
