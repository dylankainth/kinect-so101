"""SO-101 forward kinematics and damped-least-squares IK using MuJoCo Jacobians.

World/base frame: origin = base origin on the table surface, x forward, y left, z up (metres).
Joint angles are the MJCF joint coordinates in radians (see README.md for the real-servo mapping).

The "grasp point" is the site `grasp_point` placed midway between the fingertip pads. The tool
approach direction is the gripper body's -z axis (pointing from the wrist to the fingertips).
With 5 DOF the arm can set grasp-point xyz plus the approach pitch (tilt from vertical in the
vertical plane containing the base axis); wrist_roll then only spins the jaws about the approach axis.
"""
from pathlib import Path

import mujoco
import numpy as np

SCENE = str(Path(__file__).with_name("scene.xml"))
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM_JOINTS + ["gripper"]
PAN_AXIS_XY = np.array([0.0388353, 0.0])  # world xy of the shoulder_pan axis


class SO101Kinematics:
    def __init__(self, model=None):
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.jid = [self.m.joint(n).id for n in JOINTS]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.jid])
        self.lo = np.array([self.m.jnt_range[j, 0] for j in self.jid])
        self.hi = np.array([self.m.jnt_range[j, 1] for j in self.jid])
        self.site = self.m.site("grasp_point").id
        self.grip_body = self.m.body("gripper").id

    # ---- forward kinematics -------------------------------------------------
    def fk(self, q):
        """q: 5 or 6 joint angles (rad). Returns (grasp_xyz, approach_dir, closing_axis_xyz)."""
        q = np.asarray(q, float)
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr[: len(q)]] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.grip_body].reshape(3, 3)
        return self.d.site_xpos[self.site].copy(), -R[:, 2].copy(), R[:, 0].copy()

    # ---- inverse kinematics -------------------------------------------------
    @staticmethod
    def approach_dir(pan, tilt):
        """Desired approach (fingertip) direction: straight down tilted `tilt` rad away from the base
        (tilt>0 => fingertips point slightly outward/forward in the arm's vertical plane)."""
        return np.array([np.sin(tilt) * np.cos(pan), np.sin(tilt) * np.sin(pan), -np.cos(tilt)])

    def ik(self, xyz, tilt=0.0, q0=None, iters=300, tol=5e-4, damping=0.05, rot_w=0.15, roll=0.0):
        """Solve pan/lift/elbow/wrist_flex for grasp_point == xyz with approach tilt `tilt`.
        Returns (q[4], pos_err_m, tilt_err_rad). wrist_roll is held at `roll` (see `roll_for_yaw`)."""
        xyz = np.asarray(xyz, float)
        pan = np.arctan2(xyz[1], xyz[0] - PAN_AXIS_XY[0])
        q = np.zeros(5) if q0 is None else np.array(q0[:5], float)
        q[0] = -pan  # shoulder_pan axis is world -z (positive pan = clockwise from above)
        q[4] = roll  # fixed during the solve: grasp_point is offset from the roll axis, so roll moves it
        a_des = self.approach_dir(pan, tilt)
        for _ in range(iters):
            self.d.qpos[:] = 0
            self.d.qpos[self.qadr[:5]] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[self.site]
            a = -self.d.xmat[self.grip_body].reshape(3, 3)[:, 2]
            e = np.concatenate([xyz - p, rot_w * np.cross(a, a_des)])
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) / rot_w < 2e-3:
                break
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            cols = self.dadr[:4]
            J = np.vstack([jp[:, cols], rot_w * jr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q[:4] = np.clip(q[:4] + np.clip(dq, -0.2, 0.2), self.lo[:4], self.hi[:4])
        p, a, _ = self.fk(q)
        return q[:4], float(np.linalg.norm(p - xyz)), float(np.arccos(np.clip(a @ a_des, -1, 1)))

    def solve(self, xyz, tilts=(0.0, 0.15, 0.3, 0.45, 0.6, -0.15, -0.3), seeds=None, pos_tol=1.5e-3, roll=0.0):
        """Try approach tilts in order of preference and a few seeds; return first (q4, tilt) meeting
        pos_tol with exact tilt, else the best found. Raises ValueError if unreachable."""
        seeds = seeds or [(0.0, -0.3, 0.8, 0.8), (0.0, 0.3, 0.5, 0.5), (0.0, 0.0, 0.0, 0.0), (0.0, -0.8, 1.2, 0.9)]
        best = None
        for t in tilts:
            for s in seeds:
                q0 = np.array(list(s) + [0.0])
                q, pe, te = self.ik(xyz, t, q0, roll=roll)
                if pe < pos_tol and te < 0.02:
                    return q, t
                if best is None or pe + te < best[1]:
                    best = (q, pe + te, t)
        raise ValueError(f"IK failed for {xyz}: best residual {best[1]:.4f}")

    def roll_for_yaw(self, q4, yaw, period=np.pi / 2):
        """wrist_roll making the jaw closing axis (gripper x) horizontal with world yaw angle == yaw
        (mod 180 deg for the axis; mod 90 deg is chosen to keep |roll| minimal, which suits a cube)."""
        q = np.append(q4, 0.0)
        _, a, cx = self.fk(q)
        # closing axis yaw at roll=0 (gripper is pointing ~down so z_g ~ up; roll rotates about z_g)
        y0 = np.arctan2(cx[1], cx[0])
        best = None
        for k in range(int(round(2 * np.pi / period))):
            r = (yaw + k * period - y0 + np.pi) % (2 * np.pi) - np.pi
            # sign of roll vs yaw: z_g ~ +up, so positive roll => positive yaw change (verified in fk test)
            if self.lo[4] <= r <= self.hi[4] and (best is None or abs(r) < abs(best)):
                best = r
        return best if best is not None else 0.0


if __name__ == "__main__":
    k = SO101Kinematics()
    p, a, c = k.fk(np.zeros(5))
    print("zero pose grasp_point", p.round(4), "approach", a.round(3))
    for xyz in [(0.2, 0.12, 0.015), (0.2, -0.12, 0.015), (0.2, 0.12, 0.1)]:
        q, t = k.solve(xyz)
        print(xyz, "q", q.round(3), "tilt", t, "->", k.fk(np.append(q, 0))[0].round(4))
