"""Autonomous hand-eye correction: measure where the real jaw tips go versus where the model says.

    python -m core.calibration.handeye_auto            # runs the probe grid, writes control/handeye.json
    python -m core.calibration.handeye_auto --fit-only # refit from saved samples

1. With the arm parked, record a background depth image of the table.
2. For each grid point, follow the planner's own approach (closed gripper) down to a hover just
   above the table, hold, and record depth. Pixels that came >12 mm closer to the camera than the
   background are the arm; their lowest cluster near the commanded point is the jaw tip.
3. The model tip is the lowest vertex of the jaw meshes at the *measured* joint angles.
4. Fit real_xy = A @ model_xy + b (least squares) and a z offset. core.motion.aim applies the
   inverse when aiming.
"""
import argparse
import json
import sys
import time

import mujoco
import numpy as np

from core import control
from core.motion import keyframes, segmented_move, STREAM_HZ
from core.paths import CALIB_PATH
from core.perception.kinect import Kinect, deproject
from core.robot.arm import SafeArm
from core.robot.mapping import sim_to_real, real_to_sim
from core.sim import kinematics
from core.sim.pick_place import plan_pick_place, GRIPPER_OPEN

OUT = control.DIR / "handeye.json"
SAMPLES = control.DIR / "handeye_samples.json"
GRID_X = (0.16, 0.21, 0.26)
GRID_Y = (-0.12, 0.0, 0.12)
PROBE_Z = 0.035          # grasp-point height for the probe (tips ~2 cm above the table)
ARM_DZ = 0.012           # m closer to the camera than background => arm pixel
JAW_BODIES = ("gripper", "moving_jaw_so101_v1")


def load_T():
    return np.array(json.loads(CALIB_PATH.read_text())["T_robot_cam"])


def depth_median(k, n=10):
    return np.nanmedian(np.stack([k.depth_to_meters(k.wait_for_frames(5)[2]) for _ in range(n)]), 0)


def to_robot(depth, k, T):
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    P = np.asarray(deproject(u.astype(float), v.astype(float), depth[v, u], k.intr))
    P = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P
    return (T[:3, :3] @ P.T).T + T[:3, 3], v, u


def real_tip(depth, bg, k, T, near_xy):
    """Lowest cluster of arm pixels within 6 cm (xy) of near_xy, in robot frame."""
    arm = np.isfinite(depth) & np.isfinite(bg) & (bg - depth > ARM_DZ)
    d = np.where(arm, depth, np.nan)
    Q, _, _ = to_robot(d, k, T)
    Q = Q[np.linalg.norm(Q[:, :2] - near_xy, axis=1) < 0.06]
    if len(Q) < 30:
        return None, len(Q)
    low = Q[Q[:, 2] <= np.percentile(Q[:, 2], 5)]
    return low.mean(0), len(Q)


def model_tip(m, d, real_q):
    for j, v in real_to_sim(real_q).items():
        d.qpos[m.joint(j).qposadr] = v
    mujoco.mj_forward(m, d)
    pts = []
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if b in JAW_BODIES and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            V = m.mesh_vert[a:a + n]
            pts.append(d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)
    P = np.concatenate(pts)
    return P[P[:, 2] <= np.percentile(P[:, 2], 5)].mean(0)


def probe_frames(p):
    """Planner approach to point p with a closed gripper: home -> above -> down to p."""
    plan = plan_pick_place(np.array([p[0], p[1], PROBE_Z]), None)
    close_i = next(i for i, wp in enumerate(plan) if wp["gripper"] < GRIPPER_OPEN - 1e-3)
    frames = keyframes(plan[:close_i])
    for f in frames:
        f["gripper"] = 1.0
    return frames


def collect():
    T = load_T()
    control.clear()
    samples = []
    m = mujoco.MjModel.from_xml_path(kinematics.SCENE)
    d = mujoco.MjData(m)
    k = Kinect().start()
    try:
        bg = depth_median(k, 15)
        with SafeArm(allow_writes=True) as arm:
            arm.stop_requested = lambda: (control.beat() or control.is_set(control.STOP)
                                          or control.is_set(control.LIMP))
            rest = arm.read_positions()
            arm.enable_torque()
            try:
                first = probe_frames((GRID_X[1], 0.0))[0]
                # close the gripper at rest, then go to the common start pose
                segmented_move(arm, rest | {"gripper": 1.0}, dry_run=False)
                segmented_move(arm, first, dry_run=False)
                for x in GRID_X:
                    for y in GRID_Y:
                        frames = probe_frames((x, y))
                        control.beat(f"hand-eye probe ({x:.2f}, {y:+.2f})")
                        arm.follow(frames + [frames[-1]] * int(1.5 * STREAM_HZ), rate_hz=STREAM_HZ)
                        meas = arm.read_positions()
                        dep = depth_median(k, 8)
                        mt = model_tip(m, d, meas)
                        rt, n = real_tip(dep, bg, k, T, mt[:2])
                        print(f"({x:.2f},{y:+.2f}) model tip {np.round(mt, 3)}  real tip "
                              f"{None if rt is None else np.round(rt, 3)}  ({n} arm px)")
                        samples.append(dict(cmd=[x, y], model=mt.tolist(),
                                            real=None if rt is None else rt.tolist(), joints=meas))
                        arm.follow(frames[::-1], rate_hz=STREAM_HZ)
                control.beat("hand-eye: parking")
                segmented_move(arm, rest | {"gripper": 1.0}, dry_run=False)
                segmented_move(arm, rest, dry_run=False)
                time.sleep(0.5)
                arm.disable_torque()
                control.beat("hand-eye done, torque off")
            except BaseException as e:  # noqa: BLE001
                print("aborted:", repr(e))
                from core.motion import hold_until_user
                arm.stop()
                hold_until_user(arm, rest, str(e) or type(e).__name__)
    finally:
        k.stop()
        SAMPLES.write_text(json.dumps(samples, indent=1))
    return samples


def fit(samples):
    S = [s for s in samples if s["real"] is not None]
    if len(S) < 4:
        raise SystemExit(f"only {len(S)} usable samples; need >= 4")
    M = np.array([s["model"] for s in S])
    R = np.array([s["real"] for s in S])
    X = np.column_stack([M[:, :2], np.ones(len(S))])
    sol, *_ = np.linalg.lstsq(X, R[:, :2], rcond=None)
    A, b = sol[:2].T, sol[2]
    res = R[:, :2] - (M[:, :2] @ A.T + b)
    dz = float(np.median(R[:, 2] - M[:, 2]))
    err = R[:, :2] - M[:, :2]
    print("raw xy error (real - model), mm:\n", np.round(err * 1000, 1))
    print("mean raw xy error %.1f mm; after fit residual %.1f mm (max %.1f); z offset %.1f mm"
          % (np.linalg.norm(err, axis=1).mean() * 1000, np.linalg.norm(res, axis=1).mean() * 1000,
             np.linalg.norm(res, axis=1).max() * 1000, dz * 1000))
    OUT.write_text(json.dumps(dict(A=A.tolist(), b=b.tolist(), dz=dz, n=len(S),
                                   residual_mm=float(np.linalg.norm(res, axis=1).mean() * 1000)), indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    fit(json.loads(SAMPLES.read_text()) if a.fit_only else collect())
