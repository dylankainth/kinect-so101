"""Kinect -> planner -> SO-101: move the block from where it is to the target, then check with the Kinect.

    python run_pick_place.py                 # perceive, plan, rehearse in MuJoCo, dry-run on the arm (no motion)
    python run_pick_place.py --viewer        # same, with the MuJoCo viewer
    python run_pick_place.py --real          # REAL MOTION: asks you to type GO first
    python run_pick_place.py --block 0.20 0.12 0.015   # skip perception, use this block centre (m)

Frames: robot base frame, metres, x forward, y left, z up, origin on the table at the pan axis
(same in sim/, perception/calib.json and robot/). The real run starts and ends at the arm's current
(rest) pose, so torque is only dropped once the arm is folded back down.
"""
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "sim"))  # sim modules import each other by bare name

from pick_place import plan_pick_place, execute_plan, PLAN_DT, GRIPPER_OPEN  # noqa: E402
from robot.arm import SafeArm, JOINTS  # noqa: E402
from robot.mapping import sim_to_real  # noqa: E402
import control  # noqa: E402

log = logging.getLogger("pick_place")

DEFAULT_TARGET = (0.20, -0.12)
SUCCESS_RADIUS = 0.04       # m, block centre must end within this of the target
REAL_SPEED_DEG_S = 25.0     # cap for the real arm (sim plan peaks ~40 deg/s)
SEGMENT_DEG = 25.0          # max joint change per move_to call when getting to/from the start pose
STREAM_HZ = 25.0            # goal positions per second streamed to the real arm
TIME_SCALE = 1.5            # real run takes 1.5x the sim plan's duration (peak ~27 deg/s)
RADIAL_OFFSET = 0.02        # m; real jaws landed ~2 cm short of the model at ~30 cm reach (run 5)
REAL_GRIPPER_OPEN = 70.0    # real 0-100 opening used where the plan says "open"
GRIP_CLOSED_CMD = 1.0       # "closed" command; the jaws stall on the block and squeeze (run 8 slipped at 6)
# folded back against the shoulder/elbow stops: stable when the torque goes off
REST_POSE = {"shoulder_pan": -4.6, "shoulder_lift": -97.0, "elbow_flex": 94.0, "wrist_flex": 80.0,
             "wrist_roll": 0.0, "gripper": 30.0}
REST_GRIPPER_XY = np.array([0.29, 0.13])  # where the jaws lie when the arm is folded (Kinect view)
WORLD_AIM_OFFSET = np.array([0.0, 0.0])  # m in the robot frame, added to every pick and place aim
FIXED_GRASP_Z = None        # m; set per object profile to grasp at a fixed height
YAW_PERIOD = np.pi / 2      # object symmetry about z (cube: 90 deg; elongated objects: 180 deg)
FIXED_JAW_BIAS = 0.0        # m toward the fixed jaw before the learned offset (learner.py finds it now)
CLOSE_TIME_S = 1.5          # s for the slow closing sweep
SIM_CUBE_HALF = 0.015       # m, the sim cube's centre height when resting
GRASP_Z_RAISE = 0.008       # m above the block centre for the grasp point
LIFT_CHECK_STEPS = 40       # carry steps (25 Hz) before pausing to check the block left the table
TACTILE = True              # False: decide grasp success by camera after a short lift (small objects)
GRIP_SQUEEZE = 8.0          # 0-100 units commanded inside the contact point while carrying
GRASP_MIN = 12.0           # jaws stopped above this (0-100) => something is between them
# aim corrections (radial, tangential) in m, tried in order after misses; tangential + = counter-clockwise
# joint offsets and the camera are now calibrated against depth (calib_joints.py), so search around 0
AIM_SEARCH = [(0.0, 0.0), (0.012, 0.0), (-0.012, 0.0), (0.0, 0.012), (0.0, -0.012)]


def kinect_retry(fn, what, tries=4):
    """Run a camera step; if the Kinect stalls or fails to open, restart it and try again."""
    from perception import block_pose as bp
    last = None
    for k in range(tries):
        try:
            return fn()
        except (TimeoutError, RuntimeError) as e:
            last = e
            log.warning("Kinect trouble during %s (%s); restarting it (%d/%d)", what, e, k + 1, tries)
            try:
                bp.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3.0)
    raise last


def perceive_block():
    from perception.block_pose import get_block_xyz_robot, close

    def once():
        try:
            return get_block_xyz_robot(n_frames=15)
        finally:
            close()

    pose = kinect_retry(once, "block detection")
    if not pose.valid:
        raise SystemExit(f"block not found by the Kinect: {pose.reason}")
    log.info("block at %s, yaw %.0f deg (confidence %.2f, spread %.1f mm)", np.round(pose.xyz, 3),
             np.degrees(pose.yaw), pose.confidence, pose.spread_m * 1000)
    return np.asarray(pose.xyz, float), pose.yaw


def xy_from_pixel(intr, T, px, z_plane):
    """Where the camera ray through pixel px meets the horizontal plane z = z_plane (robot frame)."""
    ray_c = np.array([(px[0] - intr["cx"]) / intr["fx"], (px[1] - intr["cy"]) / intr["fy"], 1.0])
    ray = T[:3, :3] @ ray_c
    origin = T[:3, 3]
    return (origin + ray * (z_plane - origin[2]) / ray[2])[:2]


def check_with_kinect(target):
    return kinect_retry(lambda: _check_with_kinect(target), "final check")


def _check_with_kinect(target):
    """Is the block inside the target circle? Uses depth when the block's depth is valid, and falls
    back to the block's pixel position on the table plane when it is not (shiny/shadowed top)."""
    from perception import block_pose as bp
    from perception.block_detect import detect_block, load_config
    from perception.success import Region, check_success
    region = Region.circle(target[0], target[1], SUCCESS_RADIUS)
    try:
        ok, info = check_success(region, z_tol=0.03)  # the block's height depends on which face is up
        if info.get("detected", True):
            return ok, info
        kin = bp._get_kinect()
        T = np.array(json.loads((ROOT / "perception" / "calib.json").read_text())["T_robot_cam"])
        half = bp._settings()["block_size_m"] / 2
        for _ in range(5):
            rgb, _, d, _ = kin.get_frames()
            det = detect_block(rgb, kin.depth_to_meters(d), load_config(), kin.intr)
            if det.centroid_px is not None:
                xy = xy_from_pixel(kin.intr, T, det.centroid_px, half)
                inside = bool(np.linalg.norm(xy - np.asarray(target[:2])) <= SUCCESS_RADIUS)
                return inside, dict(detected=True, via="pixel", xyz=(float(xy[0]), float(xy[1]), half),
                                    reason=det.reason)
            time.sleep(0.1)
        return False, info
    finally:
        bp.close()


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


HANDEYE = ROOT / "control" / "handeye.json"
ABORTED = False


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
    import kinematics
    m = mujoco.MjModel.from_xml_path(kinematics.SCENE)
    d = mujoco.MjData(m)
    i = next(k for k, wp in enumerate(plan) if wp["gripper"] < plan[0]["gripper"] - 1e-6) - 1
    for j in JOINTS:
        d.qpos[m.joint(j).qposadr] = plan[i][j]
    mujoco.mj_forward(m, d)
    v = (d.geom_xpos[m.geom("fixed_pad").id] - d.geom_xpos[m.geom("moving_pad").id])[:2]
    return v / np.linalg.norm(v)


def snapshot(cam, name):
    """Save the current Kinect RGB frame to control/<name>.jpg (best effort)."""
    try:
        import cv2
        rgb = cam.wait_for_frames(2)[0]
        cv2.imwrite(str(control.DIR / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:  # noqa: BLE001
        log.warning("snapshot %s failed: %r", name, e)


def split_at_lift(frames):
    """Index where the arm starts lifting after the gripper has closed on the block."""
    k = next(i for i, f in enumerate(frames) if f["gripper"] <= GRIP_CLOSED_CMD + 1e-6)
    return next(i for i in range(k + 1, len(frames))
                if max(abs(frames[i][j] - frames[k][j]) for j in JOINTS[:5]) > 0.05)


def block_still_on_table(cam, block_xyz, block_top=0.03):
    """After the lift: look at the red pixels near where the block was and measure their height.
    Mostly at table height => still on the table (True); clearly raised => lifted (False).
    Too few red pixels with valid depth => can't tell; trust the gripper (False)."""
    from perception.block_detect import hsv_mask, load_config
    from perception.kinect import deproject
    T = np.array(json.loads((ROOT / "perception" / "calib.json").read_text())["T_robot_cam"])
    R, t = T[:3, :3], T[:3, 3]
    c = R.T @ (np.asarray(block_xyz, float) - t)
    u0, v0 = cam.intr["fx"] * c[0] / c[2] + cam.intr["cx"], cam.intr["fy"] * c[1] / c[2] + cam.intr["cy"]
    zs = []
    for _ in range(5):
        time.sleep(0.07)  # ~2 new frames each time
        rgb, _, d, _ = cam.wait_for_frames(2)
        dm = cam.depth_to_meters(d)
        v, u = np.nonzero(hsv_mask(rgb, load_config()))
        near = (u - u0) ** 2 + (v - v0) ** 2 < 80 ** 2
        u, v = u[near], v[near]
        z = dm[v, u]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() >= 10:
            P = np.asarray(deproject(u[ok].astype(float), v[ok].astype(float), z[ok], cam.intr))
            P = P.T if P.shape[0] == 3 and P.shape[1] != 3 else P
            zs.append(np.median((R @ P.T).T[:, 2] + t[2]))
    if len(zs) < 2:
        log.info("lift check: block not measurable (hidden), trusting the gripper")
        return False
    z = float(np.median(zs))
    log.info("lift check: red pixels near the pick spot at median z %.3f m", z)
    return z < block_top + 0.012


def run_arm(frames, real, block_xyz=None):
    """Returns 'done', 'missed' (jaws closed on nothing; backed off) or 'aborted'."""
    with SafeArm(allow_writes=real) as arm:
        if arm.mismatch:
            log.warning("motor registers differ from the calibration file on %s (using the file)",
                        ", ".join(arm.mismatch))
        start = arm.read_positions()
        log.info("start pose: %s", {j: round(v, 1) for j, v in start.items()})
        rest = dict(REST_POSE)  # always park to the same folded pose (resting on the joint stops)
        if real:
            def stop_requested():
                control.beat()
                return control.is_set(control.STOP) or control.is_set(control.LIMP)
            arm.stop_requested = stop_requested
            arm.enable_torque()
            lift = split_at_lift(frames)
            close = next(i for i, f in enumerate(frames) if f["gripper"] < frames[0]["gripper"] - 1e-6)
            result = "aborted"
            from perception.kinect import Kinect
            for _k in range(4):   # the Kinect sometimes needs a second try right after being closed
                try:
                    cam = Kinect().start()
                    break
                except RuntimeError as e:
                    log.warning("Kinect did not open (%s); retrying", e)
                    time.sleep(2.0)
            else:
                raise RuntimeError("Kinect would not open for the run")
            try:
                status("moving to start pose")
                segmented_move(arm, frames[0], dry_run=False)
                status("descending to the block")
                arm.follow(frames[:close], rate_hz=STREAM_HZ)
                time.sleep(0.4)                                  # hold still for the camera
                snapshot(cam, "pre_close")
                status("closing on the block")
                # slow sweep so the swinging jaw pushes the block into the fixed jaw, not over
                n = int(CLOSE_TIME_S * STREAM_HZ)
                g0 = frames[close - 1]["gripper"]
                sweep = [frames[close - 1] | {"gripper": g0 + (GRIP_CLOSED_CMD - g0) * (k + 1) / n} for k in range(n)]
                arm.follow(sweep, rate_hz=STREAM_HZ)
                grip = arm.settle_gripper(GRIP_CLOSED_CMD)   # wait for the jaws to actually stop
                snapshot(cam, "post_close")
                if grip >= GRASP_MIN or not TACTILE:
                    status(f"jaws stopped at {grip:.0f}/100 on something: lifting")
                    # hold just inside the contact point: firm, but not pinned at stall
                    hold = max(GRIP_CLOSED_CMD, grip - GRIP_SQUEEZE)
                    carry = [f | {"gripper": hold} if f["gripper"] <= GRIP_CLOSED_CMD + 1e-6 else f
                             for f in frames[lift:]]
                    up = carry[:LIFT_CHECK_STEPS]
                    arm.follow(up, rate_hz=STREAM_HZ)
                    time.sleep(0.3)
                    snapshot(cam, "lifted")
                    held = arm.read_positions()["gripper"]
                    if held >= GRASP_MIN:      # jaws still propped open: it is in there
                        lost = False
                    elif TACTILE:
                        lost = True
                    else:  # small objects can read ~closed even when held: ask the camera
                        lost = block_still_on_table(cam, block_xyz, block_top=2 * block_xyz[2] + 0.004)
                    if lost:
                        status(f"jaws at {held:.0f}/100 after the lift and the object is not with them: reopening")
                        arm.follow(up[::-1], rate_hz=STREAM_HZ)
                        arm.follow(sweep[::-1] + frames[:close][::-1], rate_hz=STREAM_HZ)
                        result = "slipped" if TACTILE else "missed"
                    else:
                        status(f"holding the block (jaws at {held:.0f}/100): carrying to target")
                        arm.follow(carry[LIFT_CHECK_STEPS:], rate_hz=STREAM_HZ)
                        result = "done"
                else:
                    status(f"missed (jaws closed to {grip:.0f}/100): reopening and backing off")
                    arm.follow(sweep[::-1] + frames[:close][::-1], rate_hz=STREAM_HZ)
                    result = "missed"
                status("folding back to rest")
                segmented_move(arm, rest, dry_run=False)
                time.sleep(0.5)
                arm.disable_torque()
                status(f"{result}, torque off")
            except BaseException as e:  # noqa: BLE001  STOP, safety abort, Ctrl+C, anything
                log.error("motion aborted: %r", e)
                try:
                    arm.stop()
                except Exception:  # noqa: BLE001
                    pass
                hold_until_user(arm, rest, str(e) or type(e).__name__)
                result = "aborted"
            finally:
                cam.stop()
                if arm.trace:  # commanded vs measured joints, for sim-to-real analysis
                    t0 = arm.trace[0][0]
                    (control.DIR / "trace.json").write_text(json.dumps(
                        [dict(t=t - t0, goal=g, pos=p) for t, g, p in arm.trace]))
            return result
        segmented_move(arm, frames[0], dry_run=True)
        if not real:
            # The arm hasn't moved, so check the trajectory itself: soft limits and step sizes.
            clamped = [arm.clamp(f) for f in frames]
            worst = max(abs(c[j] - f[j]) for c, f in zip(clamped, frames) for j in JOINTS)
            step = max(abs(b[j] - a[j]) for a, b in zip(frames, frames[1:]) for j in JOINTS)
            log.info("[DRY] %d waypoints; largest clamp %.1f deg, largest step %.1f deg (limit %.0f)",
                     len(frames), worst, step, arm.max_jump_deg)
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=float, nargs=3, metavar=("X", "Y", "Z"), help="skip perception")
    ap.add_argument("--target", type=float, nargs=2, default=DEFAULT_TARGET, metavar=("X", "Y"))
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo rehearsal")
    ap.add_argument("--radial-offset", type=float, default=RADIAL_OFFSET, help="m, see RADIAL_OFFSET")
    ap.add_argument("--real", action="store_true", help="move the real arm (asks for confirmation)")
    ap.add_argument("--yes", action="store_true", help="skip the GO prompt (stop_gui.py must be open)")
    ap.add_argument("--target2", type=float, nargs=2, metavar=("X", "Y"),
                    help="second spot for --practice (default: --target mirrored in y)")
    ap.add_argument("--keep-going", action="store_true", help="with --practice: carry on after a failed cycle")
    ap.add_argument("--no-learn", action="store_true", help="no learned offsets: aim straight at the detected object")
    ap.add_argument("--practice", type=int, default=0, help="N pick-and-place cycles alternating between the spots")
    ap.add_argument("--object", help="profile in perception/config.json 'objects' (e.g. yellow_candy)")
    ap.add_argument("--attempts", type=int, default=5, help="re-detect and retry until the Kinect sees success")
    a = ap.parse_args()
    if a.object:
        import os
        os.environ["KINECT_OBJECT"] = a.object  # perception reads it for colour, size and yaw period
        prof = json.loads((ROOT / "perception" / "config.json").read_text())["objects"][a.object]
        globals().update(prof.get("run", {}))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from learner import GraspLearner
    learner = GraspLearner(a.object)
    spots = [np.array(a.target, float), np.array(a.target2 if a.target2 else (a.target[0], -a.target[1]), float)]
    cycles = max(1, a.practice)
    delivered = tries = 0
    for cycle in range(cycles):
        a.practice_spots = spots if a.practice else None   # the target is chosen per attempt, from where the block is
        a.target = spots[cycle % 2] if a.practice else spots[0]
        ok = False
        for attempt in range(1, a.attempts + 1):
            tries += 1
            log.info("=== cycle %d/%d, attempt %d/%d -> target %s ===", cycle + 1, cycles, attempt, a.attempts,
                     np.round(a.target, 3))
            try:
                ok, result, rec = attempt_once(a, None if (tries > 1 or not a.block) else np.array(a.block), learner)
            except ValueError as e:  # unreachable detection (e.g. a false positive): look again next attempt
                log.warning("skipping attempt: %s", e)
                ok, result, rec = False, "no-plan", None
            except SystemExit as e:
                if "not found" not in str(e):
                    raise
                log.warning("skipping attempt: %s", e)
                ok, result, rec = False, "not-found", None
                time.sleep(1.0)
            if a.real:  # keep the evidence for every trial
                import shutil
                (control.DIR / "trials").mkdir(exist_ok=True)
                for nm in ("pre_close", "post_close", "lifted"):
                    src = control.DIR / f"{nm}.jpg"
                    if src.exists():
                        shutil.copy(src, control.DIR / "trials" / f"t{tries:02d}_{nm}.jpg")
                        src.unlink()
                with open(control.DIR / "trials" / "log.jsonl", "a") as f:
                    f.write(json.dumps(dict(trial=tries, cycle=cycle + 1, attempt=attempt, target=[float(v) for v in a.target],
                                            ok=bool(ok), result=result,
                                            rec=rec)) + chr(10))
            if rec is not None and a.real and not a.no_learn and result in ("done", "slipped", "missed"):
                reward = {"done": 1.0, "slipped": 0.5, "missed": 0.0}[result]
                if result == "done" and not ok:
                    # carried but not delivered: if it never left the pick spot, it was never held
                    seen = rec.get("final_xyz")
                    never_moved = seen is not None and np.linalg.norm(np.array(seen[:2]) - rec["xy"]) < 0.02
                    reward = 0.0 if never_moved else 0.5
                learner.record(rec["xy"], rec["offset"], reward, result=result, placed=bool(ok), yaw=rec["yaw"])
                log.info("learned from %s (reward %.1f). %s", result, reward, learner.summary(rec["xy"]))
            if ok or not a.real:
                break
        delivered += ok
        if not ok and a.practice and not a.keep_going:
            log.info("cycle %d failed after %d attempts; stopping practice", cycle + 1, a.attempts)
            break
    if a.real:
        log.info("RESULT: %d/%d cycles delivered in %d attempts", delivered, cycles, tries)

def attempt_once(a, block, learner=None):
    """Perceive (unless given), plan with the learner's grasp offset, rehearse, run.
    Returns (success, arm result, record for the learner)."""
    block, yaw = (block, 0.0) if block is not None else perceive_block()
    if getattr(a, "practice_spots", None):
        # always carry it to the spot farther from where it is now, so a block that did not move
        # (a failed trial) is still asked to travel a real distance
        far = max(a.practice_spots, key=lambda sp: np.linalg.norm(np.asarray(block[:2]) - sp))
        a.target = far
        log.info("block is at %s: carrying it to %s", np.round(block[:2], 3), np.round(far, 3))
    if a.real and np.linalg.norm(block[:2] - REST_GRIPPER_XY) < 0.07:
        raise SystemExit(f"block at {np.round(block, 3)} is where the folded gripper rests: it is probably "
                         "still in the jaws. Take it out and put it back on the table.")
    if FIXED_GRASP_Z is not None:  # thin objects: depth noise on their top is a big fraction of them
        block = np.r_[block[:2], FIXED_GRASP_Z - GRASP_Z_RAISE]
    target = np.array([a.target[0], a.target[1], block[2]])
    # plan (and rehearse) against where the model must aim for the real jaws to land on the points
    block_aim, target_aim = aim(block, 0.0), aim(target, 0.0)
    # observed by eye: the jaws land a consistent ~4 cm toward the arm's right (-y) of the target
    block_aim[:2] += WORLD_AIM_OFFSET
    target_aim[:2] += WORLD_AIM_OFFSET
    block_aim[2] += GRASP_Z_RAISE   # keep the swinging jaw off the table (it stalled there at ~20/100)
    target_aim[2] += GRASP_Z_RAISE
    goal_xy = target[:2].copy()  # the Kinect checks the real target, not the aim point
    block_xyz = block.copy()
    # grasp offset in the gripper's frame: u = toward the fixed jaw, v = across the closing axis
    u = fixed_jaw_dir(plan_pick_place(block_aim, target_aim, block_yaw=yaw, yaw_period=YAW_PERIOD))
    v = np.array([-u[1], u[0]])
    off = learner.propose(block[:2]) if (learner is not None and not a.no_learn) else np.zeros(3)
    block_aim[:2] += (FIXED_JAW_BIAS + off[0]) * u + off[1] * v
    block_aim[2] += off[2]
    log.info("block at %s; grasp offset along %+.1f / across %+.1f / height %+.1f cm -> aim %s",
             np.round(block_xyz, 3), off[0] * 100, off[1] * 100, off[2] * 100, np.round(block_aim, 3))
    rec = dict(xy=block_xyz[:2].tolist(), offset=off.tolist(), yaw=float(yaw))
    block, target = block_aim, target_aim
    plan = plan_pick_place(block, target, block_yaw=yaw, yaw_period=YAW_PERIOD)
    # The real block may be a brick on its side or end; the sim rehearsal checks reach and approach
    # with its 3 cm cube resting at the same x, y and yaw.
    sim_z = SIM_CUBE_HALF + GRASP_Z_RAISE
    sim_plan = plan if abs(block[2] - sim_z) < 0.004 else plan_pick_place(
        np.r_[block[:2], sim_z], np.r_[target[:2], sim_z], block_yaw=yaw, yaw_period=YAW_PERIOD)
    block = np.r_[block[:2], SIM_CUBE_HALF]
    target = np.r_[target[:2], SIM_CUBE_HALF]
    log.info("plan: %d samples, %.1f s in sim", len(plan), len(plan) * PLAN_DT)

    if a.viewer:
        import mujoco
        import mujoco.viewer
        import kinematics
        m = mujoco.MjModel.from_xml_path(kinematics.SCENE)
        d = mujoco.MjData(m)
        with mujoco.viewer.launch_passive(m, d) as v:
            sim = execute_plan(sim_plan, block, target, block_yaw=yaw, model=m, data=d, viewer=v)
    else:
        sim = execute_plan(sim_plan, block, target, block_yaw=yaw)
    log.info("sim rehearsal: %s (xy error %.1f cm)", "SUCCESS" if sim["success"] else "FAILED",
             sim["err_xy"] * 100)
    if not sim["success"] and a.real:
        log.error("sim rehearsal failed for this block pose; not running on the real arm")
        return False, "sim-failed", None

    frames = keyframes(plan)
    if a.real:
        print(f"\nREAL RUN: block {np.round(block, 3)} -> target {np.round(target[:2], 3)}, "
              f"{len(frames)} waypoints at <= {REAL_SPEED_DEG_S} deg/s.")
        if any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)):
            raise SystemExit("a STOP/PARK/LIMP flag is set: press 'clear' in stop_gui.py first")
        if not a.yes:
            print("Clear the area; stop_gui.py STOP freezes the arm.")
            if input("Type GO to start: ").strip() != "GO":
                raise SystemExit("aborted")
    result = run_arm(frames, real=a.real, block_xyz=block_xyz)

    if a.real:
        if result == "aborted":
            raise SystemExit("motion was stopped or faulted: no more attempts")
        time.sleep(1.0)
        ok, info = check_with_kinect(goal_xy)
        log.info("Kinect check: %s %s", "SUCCESS" if ok else "FAILED", info)
        rec["final_xyz"] = list(info["xyz"]) if info.get("xyz") is not None else None
        if ABORTED or control.is_set(control.STOP) or control.is_set(control.LIMP):
            raise SystemExit("motion was stopped or faulted: no more attempts")
        return ok, result, rec
    return sim["success"], result, rec


if __name__ == "__main__":
    main()
