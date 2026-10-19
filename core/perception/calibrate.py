"""Camera -> robot-base calibration. Writes perception/calib.json (T_robot_cam, 4x4).

Robot frame: x forward, y left, z up, origin at arm base on the table.
T_robot_cam maps a camera-frame point (optical: x right, y down, z forward) to robot frame:
    p_robot = T @ [p_cam, 1]

Methods
  aruco  (primary)  ArUco 4x4_50 marker lying flat on the table at a known pose in the
                    robot frame; depth gives 3D corners (plane fit), Kabsch fit.
  pnp    (fallback) same marker, RGB only (solvePnP IPPE_SQUARE); no depth needed.
  points (secondary) click 3+ table points with known robot-frame coords; Kabsch fit.

Marker frame (as printed, viewed from above): origin = marker centre, x = print-right,
y = print-up (towards the top edge of the print), z = out of the table.
--marker-xyz is the marker CENTRE in the robot frame (metres). --marker-yaw (degrees,
CCW seen from above) rotates the marker frame relative to the robot frame:
  yaw 0   : print-right = robot +x (forward), print-up = robot +y (left)
  yaw -90 : print-right = robot -y (right),   print-up = robot +x (forward)
  yaw 90  : print-right = robot +y (left),    print-up = robot -x (backward)
Simply lay the print so its top edge points a known direction and pick the yaw above.

  python -m perception.calibrate aruco      # uses config.json "marker" (below)
  python -m perception.calibrate pnp        # RGB-only fallback
Marker (printed file markers/aruco_4x4_50_id0_80mm.png): DICT_4X4_50, id 0, black square
edge 80 mm. Its pose comes from config.json:
  "marker": {"id": 0, "size_mm": 80, "center_xyz_m": [x, y, z], "yaw_deg": 0}
CLI flags --id/--size-mm/--marker-xyz/--marker-yaw override the config.

  python -m perception.calibrate points --points 0.15,0.10 0.15,-0.10 0.30,0.10 0.30,-0.10
  python -m perception.calibrate verify
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from .kinect import Kinect, deproject

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "calib.json")
DICT = cv2.aruco.DICT_4X4_50


# ---------------------------------------------------------------- io / math
CONFIG_PATH = os.path.join(HERE, "config.json")
MARKER_DEFAULTS = dict(id=0, size_mm=80.0, center_xyz_m=[0.0, 0.0, 0.0], yaw_deg=0.0)


def load_marker_cfg(path=CONFIG_PATH):
    cfg = dict(MARKER_DEFAULTS)
    try:
        with open(path) as f:
            cfg.update(json.load(f).get("marker", {}))
    except (OSError, ValueError):
        pass
    return cfg


def resolve_marker(args):
    """Merge CLI overrides onto config.json defaults (writes into args)."""
    c = load_marker_cfg()
    if args.id is None: args.id = c["id"]
    if args.size_mm is None: args.size_mm = c["size_mm"]
    if args.marker_xyz is None: args.marker_xyz = c["center_xyz_m"]
    if args.marker_yaw is None: args.marker_yaw = c["yaw_deg"]
    if args.marker_xyz == [0.0, 0.0, 0.0] and args.marker_yaw == 0.0 and not os.path.exists(CONFIG_PATH):
        print("WARNING: no marker pose in config.json; assuming marker centre at robot origin")


def save_calib(T, method, rms, extra=None, path=CALIB_PATH):
    d = dict(T_robot_cam=np.asarray(T).tolist(), method=method, rms_error_m=float(rms),
             time=time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {}))
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"saved {path}  (method={method}, fit RMS={rms*1000:.1f} mm)")


def load_calib(path=CALIB_PATH):
    with open(path) as f:
        return np.array(json.load(f)["T_robot_cam"], dtype=np.float64)


def kabsch(src, dst):
    """Rigid transform T (4x4) minimising |T src - dst|; src,dst Nx3. Returns T, rms."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    res = (src @ R.T + t) - dst
    return T, float(np.sqrt((res ** 2).sum(1).mean()))


def marker_in_robot(xyz, yaw_deg):
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = xyz
    return T


def marker_corners_local(size_m):
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])  # TL TR BR BL


# ---------------------------------------------------------------- detection
def find_marker(rgb, marker_id):
    # Small markers (~20 px at 0.75 m) are only found after upscaling; corners are
    # mapped back to native pixels.
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for scale in (1, 3, 2, 4):
        img = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale,
                                                 interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            if marker_id is None or i == marker_id:
                return c.reshape(4, 2).astype(np.float64) / scale
    return None


def corners_3d_from_depth(corners_px, depth_m, intr):
    """Fit a plane to depth inside the marker quad, intersect corner rays with it."""
    mask = np.zeros(depth_m.shape, np.uint8)
    cen = corners_px.mean(0)
    cv2.fillConvexPoly(mask, (cen + (corners_px - cen) * 0.8).astype(np.int32), 1)
    vs, us = np.nonzero((mask > 0) & np.isfinite(depth_m))
    if len(us) < 50:
        return None
    pts = deproject(us, vs, depth_m[vs, us], intr)
    for _ in range(3):  # iterative outlier rejection
        c = pts.mean(0)
        n = np.linalg.svd(pts - c)[2][-1]
        dist = np.abs((pts - c) @ n)
        pts = pts[dist < max(0.004, 2 * dist.std())]
    c = pts.mean(0)
    n = np.linalg.svd(pts - c)[2][-1]
    rays = np.stack([(corners_px[:, 0] - intr["cx"]) / intr["fx"],
                     (corners_px[:, 1] - intr["cy"]) / intr["fy"], np.ones(4)], 1)
    s = (c @ n) / (rays @ n)
    return rays * s[:, None]


# ---------------------------------------------------------------- methods
def calib_aruco(kin, args):
    local = marker_corners_local(args.size_mm / 1000.0)
    Tm = marker_in_robot(args.marker_xyz, args.marker_yaw)
    robot_pts = (Tm[:3, :3] @ local.T).T + Tm[:3, 3]
    acc = []
    for _ in range(args.frames):
        rgb, _, d, _ = kin.get_frames()
        c = find_marker(rgb, args.id)
        if c is not None:
            p3 = corners_3d_from_depth(c, kin.depth_to_meters(d), kin.intr)
            if p3 is not None:
                acc.append(p3)
        time.sleep(0.05)
    if len(acc) < max(3, args.frames // 4):
        raise SystemExit(f"marker id {args.id} with usable depth seen in only {len(acc)} frames")
    cam_pts = np.median(np.array(acc), axis=0)
    T, rms = kabsch(cam_pts, robot_pts)
    return T, rms, dict(marker_id=args.id, size_mm=args.size_mm, frames_used=len(acc))


def calib_pnp(kin, args):
    local = marker_corners_local(args.size_mm / 1000.0)
    Tm = marker_in_robot(args.marker_xyz, args.marker_yaw)
    K = np.array([[kin.intr["fx"], 0, kin.intr["cx"]], [0, kin.intr["fy"], kin.intr["cy"]], [0, 0, 1]])
    poses = []
    for _ in range(args.frames):
        rgb, *_ = kin.get_frames()
        c = find_marker(rgb, args.id)
        if c is not None:
            ok, rv, tv = cv2.solvePnP(local, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                poses.append(np.r_[tv.flatten(), rv.flatten()])
        time.sleep(0.05)
    if len(poses) < 3:
        raise SystemExit("marker not seen")
    p = np.median(np.array(poses), axis=0)
    T_cm = np.eye(4)
    T_cm[:3, :3] = cv2.Rodrigues(p[3:])[0]
    T_cm[:3, 3] = p[:3]
    T = Tm @ np.linalg.inv(T_cm)
    return T, 0.0, dict(marker_id=args.id, size_mm=args.size_mm, frames_used=len(poses))


def calib_points(kin, args):
    robot = []
    for s in args.points:
        v = [float(x) for x in s.split(",")]
        robot.append(v if len(v) == 3 else v + [0.0])
    robot = np.array(robot)
    if len(robot) < 3:
        raise SystemExit("need >= 3 points")
    clicks = []
    win = "click points IN ORDER (u=undo, Enter=done, q=abort)"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(robot):
            clicks.append((x, y))
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        rgb, *_ = kin.get_frames()
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(vis, f"{i+1}", (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        msg = (f"click point {len(clicks)+1} = robot {robot[len(clicks)]}" if len(clicks) < len(robot)
               else "all clicked: press Enter")
        cv2.putText(vis, msg, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and clicks:
            clicks.pop()
        elif k in (13, 10) and len(clicks) == len(robot):
            break
        elif k in (ord("q"), 27):
            raise SystemExit("aborted")
    cv2.destroyAllWindows()
    zs = [[] for _ in clicks]
    for _ in range(args.frames):
        _, _, d, _ = kin.get_frames()
        dm = kin.depth_to_meters(d)
        for i, (x, y) in enumerate(clicks):
            patch = dm[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            if np.isfinite(patch).any():
                zs[i].append(np.nanmedian(patch))
        time.sleep(0.05)
    cam = []
    for i, (x, y) in enumerate(clicks):
        if not zs[i]:
            raise SystemExit(f"no valid depth at click {i+1}; pick another point")
        cam.append(deproject(x, y, np.median(zs[i]), kin.intr))
    T, rms = kabsch(np.array(cam), robot)
    return T, rms, dict(points_robot=robot.tolist(), clicks_px=clicks)


def verify(kin, T):
    """Live view; hover to read robot-frame XYZ of the pixel under the mouse."""
    mouse = {}
    cv2.namedWindow("verify")
    cv2.setMouseCallback("verify", lambda e, x, y, *_: mouse.update(p=(x, y)))
    while True:
        rgb, _, d, _ = kin.get_frames()
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if "p" in mouse:
            x, y = mouse["p"]
            z = kin.depth_to_meters(d)[y, x]
            if np.isfinite(z):
                p = T @ np.r_[kin.deproject(x, y, z), 1]
                cv2.putText(vis, f"robot x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("verify", vis)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("aruco", "pnp"):
        p = sub.add_parser(name)
        p.add_argument("--id", type=int, default=None)
        p.add_argument("--size-mm", type=float, default=None)
        p.add_argument("--marker-xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                       help="marker CENTRE in robot frame (m); default from config.json")
        p.add_argument("--marker-yaw", type=float, default=None)
        p.add_argument("--frames", type=int, default=30)
    p = sub.add_parser("points")
    p.add_argument("--points", nargs="+", required=True, metavar="X,Y[,Z]")
    p.add_argument("--frames", type=int, default=20)
    sub.add_parser("verify")
    a = ap.parse_args()

    with Kinect() as kin:
        kin.wait_for_frames()
        time.sleep(0.5)
        if a.cmd == "verify":
            return verify(kin, load_calib())
        if a.cmd in ("aruco", "pnp"):
            resolve_marker(a)
            print(f"marker id {a.id}, {a.size_mm} mm, centre {list(a.marker_xyz)} yaw {a.marker_yaw}")
        T, rms, extra = dict(aruco=calib_aruco, pnp=calib_pnp, points=calib_points)[a.cmd](kin, a)
        np.set_printoptions(precision=4, suppress=True)
        print(T)
        print("camera origin in robot frame (m):", T[:3, 3], "(z should be camera height above table)")
        save_calib(T, a.cmd, rms, extra)


if __name__ == "__main__":
    main()
