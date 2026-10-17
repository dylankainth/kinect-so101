"""Refine calib.json using the table plane.

A small marker pins down position and heading well but its tilt poorly. This keeps the
marker's position and in-plane heading and replaces the up-axis with the normal of the
table plane fitted from depth, and puts robot z=0 on that plane.

    python -m perception.refine_plane            # mat region default
    python -m perception.refine_plane --roi 40 160 480 460   # u0 v0 u1 v1 pixels, table only
"""
import argparse
import json
from pathlib import Path

import numpy as np

from perception.kinect import Kinect, deproject

HERE = Path(__file__).parent


def fit_plane(P, iters=200, thresh=0.006, rng=np.random.default_rng(0)):
    """RANSAC plane fit; returns unit normal pointing toward the camera, offset, inlier mask."""
    best = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    if n[2] > 0:  # camera looks along +z; the table normal toward the camera has n_z < 0
        n = -n
    return n, c, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, nargs=4, default=[40, 160, 480, 460], metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--frames", type=int, default=15)
    a = ap.parse_args()

    calib_path = HERE / "calib.json"
    calib = json.loads(calib_path.read_text())
    T_rc = np.array(calib["T_robot_cam"])
    marker = json.loads((HERE / "config.json").read_text())["marker"]
    m_xyz = np.array(marker["center_xyz_m"], float)

    k = Kinect().start()
    try:
        d = np.median(np.stack([k.depth_to_meters(k.wait_for_frames(5)[2]) for _ in range(a.frames)]), 0)
    finally:
        k.stop()

    u0, v0, u1, v1 = a.roi
    v, u = np.mgrid[v0:v1:4, u0:u1:4]
    z = d[v, u]
    ok = z > 0
    P = np.asarray(deproject(u[ok].astype(float), v[ok].astype(float), z[ok], k.intr))
    if P.shape[0] == 3 and P.shape[1] != 3:
        P = P.T
    n, c, inl = fit_plane(P)
    print(f"plane: {inl.sum()}/{len(P)} inliers, rms {np.std((P[inl] - c) @ n) * 1000:.1f} mm")

    # Robot axes expressed in the camera frame.
    R_cr = T_rc[:3, :3].T
    t_cr = -R_cr @ T_rc[:3, 3]
    z_ax = n
    x_ax = R_cr[:, 0] - (R_cr[:, 0] @ n) * n
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)

    # Marker centre (camera frame) from the old fit, dropped onto the plane at its height.
    p_m = R_cr @ m_xyz + t_cr
    p_m = p_m - ((p_m - c) @ n - m_xyz[2]) * n
    R_new = np.column_stack([x_ax, y_ax, z_ax])  # camera <- robot
    origin = p_m - R_new @ m_xyz

    T_new = np.eye(4)
    T_new[:3, :3] = R_new.T
    T_new[:3, 3] = -R_new.T @ origin

    Rr = (T_new[:3, :3] @ P[inl].T).T + T_new[:3, 3]
    print(f"table z in robot frame: mean {Rr[:, 2].mean() * 1000:.1f} mm, std {Rr[:, 2].std() * 1000:.1f} mm")
    shift = np.linalg.norm(T_new[:3, 3] - T_rc[:3, 3])
    ang = np.degrees(np.arccos(np.clip((np.trace(T_new[:3, :3] @ T_rc[:3, :3].T) - 1) / 2, -1, 1)))
    print(f"change vs previous: {ang:.1f} deg, {shift * 1000:.0f} mm (camera position)")
    print("camera origin in robot frame (m):", (-T_new[:3, :3] @ np.zeros(3) + T_new[:3, 3]).round(4))

    calib.update(T_robot_cam=T_new.round(6).tolist(), method=calib.get("method", "aruco") + "+plane")
    calib_path.write_text(json.dumps(calib, indent=2))
    print("saved", calib_path)


if __name__ == "__main__":
    main()
