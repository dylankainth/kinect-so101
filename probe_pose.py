"""Sim-to-real probe: hold the arm at the planned pre-grasp pose over the block and compare
commanded joints, measured joints, and where the Kinect sees the arm.

    python probe_pose.py [--block X Y Z]

Writes control/probe.jpg (red = FK of commanded joints, green = FK of measured joints, drawn on
the Kinect image) and control/probe.json. The arm parks (folds to rest, limp) afterwards.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "sim"))
import kinematics  # noqa: E402
from pick_place import plan_pick_place  # noqa: E402

import control  # noqa: E402
from robot.arm import SafeArm, JOINTS  # noqa: E402
from robot.mapping import sim_to_real, real_to_sim  # noqa: E402
from run_pick_place import perceive_block, segmented_move, keyframes, DEFAULT_TARGET, STREAM_HZ  # noqa: E402
from perception.kinect import deproject  # noqa: E402

CHAIN = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]


def fk_points(m, d, real):
    for j, v in real_to_sim(real).items():
        d.qpos[m.joint(j).qposadr] = v
    mujoco.mj_forward(m, d)
    pts = [d.xpos[m.body(b).id].copy() for b in CHAIN]
    return pts, d.site_xpos[m.site("grasp_point").id].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=float, nargs=3)
    a = ap.parse_args()
    block = np.array(a.block) if a.block else perceive_block()
    plan = plan_pick_place(block, (*DEFAULT_TARGET, block[2]))

    m = mujoco.MjModel.from_xml_path(kinematics.SCENE)
    d = mujoco.MjData(m)
    # pre-grasp = first plan sample whose grasp point is above the block (xy within 5 mm)
    for i, wp in enumerate(plan):
        _, g = fk_points(m, d, sim_to_real(wp))
        if np.linalg.norm(g[:2] - block[:2]) < 0.005:
            break
    pre = sim_to_real(plan[i])
    pre["gripper"] = max(pre["gripper"], 1.0)
    print("pre-grasp sample", i, {j: round(v, 1) for j, v in pre.items()})

    control.clear()
    with SafeArm(allow_writes=True) as arm:
        arm.stop_requested = lambda: control.is_set(control.STOP) or control.is_set(control.LIMP)
        rest = arm.read_positions()
        arm.enable_torque()
        try:
            segmented_move(arm, sim_to_real(plan[0]), dry_run=False)
            # follow the real plan to the pre-grasp (with sag compensation), then hold there 2 s
            frames = keyframes(plan[:i + 1])
            arm.follow(frames + [frames[-1]] * int(2 * STREAM_HZ), rate_hz=STREAM_HZ)
            meas = arm.read_positions()
            load = arm.bus.sync_read("Present_Load", normalize=False)
            from perception.kinect import Kinect
            k = Kinect().start()
            try:
                f = k.wait_for_frames(5)
                rgb, depth = f[0], k.depth_to_meters(f[2])
            finally:
                k.stop()
        finally:
            segmented_move(arm, sim_to_real(plan[0]), dry_run=False)
            segmented_move(arm, rest, dry_run=False)
            time.sleep(0.5)
            arm.disable_torque()

    T = np.array(json.loads((ROOT / "perception/calib.json").read_text())["T_robot_cam"])
    R, t = T[:3, :3], T[:3, 3]
    I = k.intr

    def px(p):
        c = R.T @ (np.asarray(p) - t)
        return int(I["fx"] * c[0] / c[2] + I["cx"]), int(I["fy"] * c[1] / c[2] + I["cy"])

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = {"block": block.tolist(), "commanded": pre, "measured": meas, "load": load}
    for name, q, col in (("commanded", pre, (0, 0, 255)), ("measured", meas, (0, 255, 0))):
        pts, g = fk_points(m, d, q)
        pp = [px(p) for p in pts]
        for p0, p1 in zip(pp, pp[1:]):
            cv2.line(img, p0, p1, col, 2)
        cv2.drawMarker(img, px(g), col, cv2.MARKER_CROSS, 14, 2)
        out[f"grasp_{name}"] = g.tolist()
    cv2.drawMarker(img, px(block), (255, 0, 255), cv2.MARKER_SQUARE, 12, 2)

    # Where the real gripper tip is: lowest point above the table within 8 cm (xy) of the
    # commanded grasp point, from Kinect depth.
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    P = np.asarray(deproject(u.astype(float), v.astype(float), depth[v, u], k.intr))
    P = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P
    Q = (R @ P.T).T + t
    gc = np.array(out["grasp_commanded"])
    sel = (np.linalg.norm(Q[:, :2] - gc[:2], axis=1) < 0.08) & (Q[:, 2] > 0.035) & (Q[:, 2] < 0.2)
    sel &= np.linalg.norm(Q[:, :2] - block[:2], axis=1) > 0.03  # ignore the block itself
    if sel.any():
        low = Q[sel][Q[sel][:, 2] <= np.percentile(Q[sel][:, 2], 3)]
        tip = low.mean(0)
        out["kinect_tip"] = tip.tolist()
        cv2.drawMarker(img, px(tip), (255, 255, 0), cv2.MARKER_DIAMOND, 14, 2)
        print("Kinect-observed lowest arm point", tip.round(3))
    cv2.imwrite(str(control.DIR / "probe.jpg"), img)
    (control.DIR / "probe.json").write_text(json.dumps(out, indent=1))
    print("joint error (commanded - measured):",
          {j: round(pre[j] - meas[j], 1) for j in JOINTS})
    print("grasp point commanded", np.round(out["grasp_commanded"], 3),
          "measured-FK", np.round(out["grasp_measured"], 3))


if __name__ == "__main__":
    main()
