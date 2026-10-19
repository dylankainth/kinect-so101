"""Kinect -> planner -> SO-101: move the block from where it is to the target, then check with the Kinect.

    python -m projects.pick_place.run_pick_place                 # perceive, plan, rehearse in MuJoCo, dry-run on the arm (no motion)
    python -m projects.pick_place.run_pick_place --viewer        # same, with the MuJoCo viewer
    python -m projects.pick_place.run_pick_place --real          # REAL MOTION: asks you to type GO first
    python -m projects.pick_place.run_pick_place --block 0.20 0.12 0.015   # skip perception, use this block centre (m)

(run from the repo root)

Frames: robot base frame, metres, x forward, y left, z up, origin on the table at the pan axis
(same in core/sim, core/perception/calib.json and core/robot). The real run starts and ends at the arm's current
(rest) pose, so torque is only dropped once the arm is folded back down.
"""
import argparse
import json
import logging
import time

import numpy as np

from core import control, grasp, motion
from core.motion import REAL_SPEED_DEG_S, aim, fixed_jaw_dir, keyframes, wait_to_cool
from core.paths import CALIB_PATH, CORE
from core.perception.util import kinect_retry, perceive_block, xy_from_pixel
from core.sim.pick_place import plan_pick_place, execute_plan, PLAN_DT

log = logging.getLogger("pick_place")

DEFAULT_TARGET = (0.20, -0.12)
SUCCESS_RADIUS = 0.04       # m, block centre must end within this of the target
RADIAL_OFFSET = 0.02        # m; real jaws landed ~2 cm short of the model at ~30 cm reach (run 5)
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


def check_with_kinect(target):
    return kinect_retry(lambda: _check_with_kinect(target), "final check")


def _check_with_kinect(target):
    """Is the block inside the target circle? Uses depth when the block's depth is valid, and falls
    back to the block's pixel position on the table plane when it is not (shiny/shadowed top)."""
    from core.perception import block_pose as bp
    from core.perception.block_detect import detect_block, load_config
    from core.perception.success import Region, check_success
    region = Region.circle(target[0], target[1], SUCCESS_RADIUS)
    try:
        ok, info = check_success(region, z_tol=0.03)  # the block's height depends on which face is up
        if info.get("detected", True):
            return ok, info
        kin = bp._get_kinect()
        T = np.array(json.loads((CALIB_PATH).read_text())["T_robot_cam"])
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


def run_arm(frames, real, block_xyz=None):
    """core.grasp.run_arm with this module's tunables (an --object profile overrides them at startup)."""
    return grasp.run_arm(frames, real, block_xyz, grasp.GraspParams(
        close_time_s=CLOSE_TIME_S, grasp_min=GRASP_MIN, tactile=TACTILE, grip_squeeze=GRIP_SQUEEZE,
        lift_check_steps=LIFT_CHECK_STEPS))


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
    ap.add_argument("--object", help="profile in core/perception/config.json 'objects' (e.g. yellow_candy)")
    ap.add_argument("--attempts", type=int, default=5, help="re-detect and retry until the Kinect sees success")
    a = ap.parse_args()
    if a.object:
        import os
        os.environ["KINECT_OBJECT"] = a.object  # perception reads it for colour, size and yaw period
        prof = json.loads((CORE / "perception" / "config.json").read_text())["objects"][a.object]
        globals().update(prof.get("run", {}))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    from .learner import GraspLearner
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
    if a.real:
        wait_to_cool()
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
        from core.sim import kinematics
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
        if motion.ABORTED or control.is_set(control.STOP) or control.is_set(control.LIMP):
            raise SystemExit("motion was stopped or faulted: no more attempts")
        return ok, result, rec
    return sim["success"], result, rec


if __name__ == "__main__":
    main()
