"""Kinematic calibration from depth: fit joint zero offsets (and a small camera correction) so the
MuJoCo model of the arm lines up with what the Kinect actually sees.

    python -m core.calibration.calib_joints           # move through the poses, collect depth, fit, save
    python -m core.calibration.calib_joints --fit-only

1. Background depth with the arm folded at rest.
2. For each hover pose: follow a smooth joint path there, hold, read the joints, record depth.
   Pixels >15 mm closer to the camera than the background are the arm.
3. Fit offsets d (deg) for pan / lift / elbow / wrist_flex, plus a rigid camera correction
   (dx, dy, dz, dyaw), minimising the distance from observed arm points to the model's mesh surface.
4. Save core/robot/joint_offsets.json (read by core/robot/mapping.py) and the corrected core/perception/calib.json.
"""
import argparse
import json
import time

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from core import control
from core.motion import keyframes, segmented_move, STREAM_HZ, hold_until_user
from core.paths import CALIB_PATH, JOINT_OFFSETS_PATH
from core.perception.kinect import Kinect, deproject
from core.robot.arm import SafeArm
from core.robot.mapping import real_to_sim
from core.sim import kinematics
from core.sim.pick_place import home_pose, _joint_move, _wp, JOINTS

DATA = control.DIR / "calib_joints.npz"
OFFSETS = JOINT_OFFSETS_PATH
CALIB = CALIB_PATH
FIT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
ARM_DZ = 0.015
# (x, y, z) grasp-point hover targets in the robot frame, tilt (rad), wrist roll (rad)
POSES = [((0.18, 0.12, 0.08), 0.0, 0.0), ((0.24, 0.0, 0.07), 0.15, 0.6), ((0.18, -0.13, 0.08), 0.0, -0.5),
         ((0.26, -0.10, 0.07), 0.3, 0.0), ((0.15, 0.05, 0.07), 0.0, 1.0), ((0.30, 0.05, 0.05), 0.45, -0.8),
         ((0.12, -0.08, 0.06), 0.0, 0.3), ((0.28, 0.12, 0.06), 0.45, 0.0), ((0.22, -0.05, 0.12), 0.6, -0.3)]
VIS_BODIES = ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


def depth_median(k, n):
    return np.nanmedian(np.stack([k.depth_to_meters(k.wait_for_frames(5)[2]) for _ in range(n)]), 0)


def cam_points(depth, k):
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    P = np.asarray(deproject(u.astype(float), v.astype(float), depth[v, u], k.intr))
    return P.T if P.shape[0] == 3 and P.shape[1] != 3 else P


def collect():
    T = np.array(json.loads(CALIB.read_text())["T_robot_cam"])
    kin = kinematics.SO101Kinematics()
    h = home_pose()
    hq = np.array([h[j] for j in JOINTS[:5]])
    samples = []
    control.clear()
    k = Kinect().start()
    try:
        bg = depth_median(k, 15)
        with SafeArm(allow_writes=True) as arm:
            arm.stop_requested = lambda: (control.beat() or control.is_set(control.STOP)
                                          or control.is_set(control.LIMP))
            rest = arm.read_positions()
            arm.enable_torque()
            try:
                home_frames = keyframes([_wp(hq, h["gripper"])])
                segmented_move(arm, home_frames[0], dry_run=False)
                for xyz, tilt, roll in POSES:
                    q4, _ = kin.solve(xyz, tilts=(tilt, 0.0, 0.3, 0.45), roll=roll)
                    q = np.append(q4, roll)
                    path = keyframes([_wp(p, h["gripper"]) for p in _joint_move(hq, q)])
                    control.beat(f"joint calibration pose {xyz}")
                    arm.follow(path + [path[-1]] * int(1.5 * STREAM_HZ), rate_hz=STREAM_HZ)
                    meas = arm.read_positions()
                    dep = depth_median(k, 10)
                    arm_px = np.isfinite(dep) & np.isfinite(bg) & (bg - dep > ARM_DZ)
                    P = cam_points(np.where(arm_px, dep, np.nan), k)
                    Q = (T[:3, :3] @ P.T).T + T[:3, 3]
                    keep = (Q[:, 2] > 0.02) & (np.linalg.norm(Q[:, :2], axis=1) < 0.45)
                    print(f"pose {xyz}: {keep.sum()} arm points, joints {[round(meas[j], 1) for j in JOINTS]}")
                    samples.append((meas, P[keep]))
                    arm.follow(path[::-1], rate_hz=STREAM_HZ)
                control.beat("joint calibration: parking")
                segmented_move(arm, rest, dry_run=False)
                time.sleep(0.5)
                arm.disable_torque()
                control.beat("joint calibration done, torque off")
            except BaseException as e:  # noqa: BLE001
                print("aborted:", repr(e))
                arm.stop()
                hold_until_user(arm, rest, str(e) or type(e).__name__)
    finally:
        k.stop()
    np.savez(DATA, joints=np.array([[s[0][j] for j in JOINTS] for s in samples]),
             **{f"pts{i}": s[1] for i, s in enumerate(samples)}, T=T)
    return load()


def load():
    z = np.load(DATA)
    joints = z["joints"]
    pts = [z[f"pts{i}"] for i in range(len(joints))]
    return joints, pts, z["T"]


class Model:
    def __init__(self, n_per_geom=500, seed=0):
        self.m = mujoco.MjModel.from_xml_path(kinematics.SCENE)
        self.d = mujoco.MjData(self.m)
        rng = np.random.default_rng(seed)
        self.geoms = []
        for g in range(self.m.ngeom):
            b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])
            if b in VIS_BODIES and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and self.m.geom_group[g] == 2:
                mid = self.m.geom_dataid[g]
                V = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                self.geoms.append((g, V[rng.choice(len(V), min(n_per_geom, len(V)), replace=False)]))

    def points(self, real_q, offs_deg):
        sim = real_to_sim(dict(zip(JOINTS, real_q)))
        for j, o in zip(FIT_JOINTS, offs_deg):
            sim[j] += np.radians(o)
        for j, v in sim.items():
            self.d.qpos[self.m.joint(j).qposadr] = v
        mujoco.mj_kinematics(self.m, self.d)
        return np.concatenate([self.d.geom_xpos[g] + V @ self.d.geom_xmat[g].reshape(3, 3).T
                               for g, V in self.geoms])


def rigid(p):
    """camera correction: [dx, dy, dz (m), dyaw (deg)] applied in the robot frame."""
    c, s = np.cos(np.radians(p[3])), np.sin(np.radians(p[3]))
    M = np.eye(4)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    M[:3, 3] = p[:3]
    return M


def fit(joints, pts, T, max_pts=1500, clip=0.03):
    model = Model()
    rng = np.random.default_rng(1)
    pts = [P[rng.choice(len(P), min(max_pts, len(P)), replace=False)] for P in pts]

    def cost(x, report=False):
        offs, cam = x[:4], x[4:]
        Tc = rigid(cam) @ T
        total, per = 0.0, []
        for q, P in zip(joints, pts):
            Q = (Tc[:3, :3] @ P.T).T + Tc[:3, 3]
            d, _ = cKDTree(model.points(q, offs)).query(Q)
            e = np.minimum(d, clip)
            per.append(np.median(d))
            total += np.mean(e ** 2)
        if report:
            return per
        return total / len(pts) + 1e-6 * np.sum(x[:4] ** 2)

    x0 = np.zeros(8)
    print("before: median point-to-model distance per pose (mm):", np.round(np.array(cost(x0, True)) * 1000, 1))
    best = None
    for start in (x0, np.r_[np.full(4, 3.0), 0, 0, 0, 0], np.r_[np.full(4, -3.0), 0, 0, 0, 0]):
        r = minimize(cost, start, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=0.05, fatol=1e-9, initial_simplex=None))
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    print("after:  median point-to-model distance per pose (mm):", np.round(np.array(cost(x, True)) * 1000, 1))
    print("joint offsets (deg, added to the sim angle):", dict(zip(FIT_JOINTS, np.round(x[:4], 2))))
    print("camera correction: dx %.1f mm, dy %.1f mm, dz %.1f mm, yaw %.2f deg" % (*(x[4:7] * 1000), x[7]))
    return x


def save(x, T):
    # the fit ran on top of any offsets already in use, so accumulate
    prev = json.loads(OFFSETS.read_text()) if OFFSETS.exists() else {}
    OFFSETS.write_text(json.dumps({j: float(prev.get(j, 0.0) + o) for j, o in zip(FIT_JOINTS, x[:4])}, indent=1))
    calib = json.loads(CALIB.read_text())
    (control.DIR / "calib_before_joint_fit.json").write_text(json.dumps(calib, indent=2))
    calib.update(T_robot_cam=(rigid(x[4:]) @ T).round(6).tolist(), method=calib.get("method", "") + "+armfit")
    CALIB.write_text(json.dumps(calib, indent=2))
    print("saved", OFFSETS, "and", CALIB)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()
    joints, pts, T = load() if a.fit_only else collect()
    x = fit(joints, pts, T)
    save(x, T)
