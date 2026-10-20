"""Stack N blocks (default 3) on top of each other, looking at the table with the Kinect between moves.

    python -m projects.stack_blocks.main --look                  # just report what the Kinect sees
    python -m projects.stack_blocks.main --blocks 0.18 0.08 0.24 -0.08 0.20 0.0   # sim only, given positions
    python -m projects.stack_blocks.main                         # look, then plan + rehearse in sim (no motion)
    python -m projects.stack_blocks.main --real                  # REAL MOTION: asks you to type GO first

(run from the repo root; keep core/tools/stop_gui.py open during real runs)

Each step starts from a fresh look: the stack is the group of blocks at the base spot (its height tells how
many blocks are in it), every other block is loose. The next loose block is carried onto the stack. Because
the state always comes from what the camera sees, a block that was dropped or missed simply shows up as
loose again and is retried, up to --attempts per block.

Blocks must be on the mat (the region core/perception/multi_block.DEFAULT_ROI) and far enough apart for the
open jaws to clear their neighbours. --object says what the blocks are (OBJECTS below): the 8 x 5 x 3 cm boxes
are gripped across their 5 cm side; cubes across any face.
"""
import argparse
import json
import logging
import time

import numpy as np

from core import control, grasp, motion
from core.motion import REAL_SPEED_DEG_S, aim, keyframes, wait_to_cool
from core.perception.multi_block import BlockObs, perceive_blocks
from core.sim.pick_place import PLAN_DT, execute_plan, plan_pick_place, scene_with_support

log = logging.getLogger("stack_blocks")

PREFERRED_BASE = np.array([0.21, -0.02])  # where to build the stack if you do not say (middle of good reach)
BASE_TOL = 0.03             # m, a block this close to the base spot is part of the stack
JAW_CLEAR = 0.025           # m, extra room the open jaws need beside a block (see Obj.min_gap)
STACK_XY_TOL = 0.02         # m, finished stack must be this close to the base spot
GRASP_Z_RAISE = 0.008       # m above the block centre for the grasp point (as in pick_place)
PLACE_HOVER = 0.035          # m, approach point above the stack: reach is the limit up there, and the carried
                            # block still clears the stack top by about 3 cm
OPEN_SIM = 1.2              # rad, jaw opening in the plan: 1.2 rad is what the real arm's 70/100 opens to
                            # (about 8 cm between the pads), so the rehearsal has the real clearance

# height = one layer of the stack; footprint = top face (long, short). Elongated blocks are gripped across
# the short side, so yaw (the closing direction) repeats every 180 deg; cubes every 90.
OBJECTS = {
    "box8x5": dict(height=0.03, footprint=(0.08, 0.05)),
    "cube3": dict(height=0.03, footprint=(0.03, 0.03)),
    "box_on_edge": dict(height=0.05, footprint=(0.08, 0.03)),   # the 8 x 5 x 3 cm box standing on its 8 x 3 side
    "box_on_end": dict(height=0.08, footprint=(0.05, 0.03)),    # ... or on its 5 x 3 end
}


class Obj:
    def __init__(self, name):
        self.name = name
        self.height = OBJECTS[name]["height"]
        self.long, self.short = OBJECTS[name]["footprint"]
        self.elongated = self.long > 1.25 * self.short
        self.period = np.pi if self.elongated else np.pi / 2
        self.min_gap = self.long / 2 + self.short / 2 + JAW_CLEAR   # centre-to-centre, any orientation
WORLD_AIM_OFFSET = np.array([0.0, 0.0])  # m, added to every pick and place aim (as in pick_place)


class SimWorld:
    """Stands in for the camera when no hardware is used: blocks move exactly where the plan sends them."""

    def __init__(self, blocks, obj):
        self.obj = obj
        self.blocks = [BlockObs(np.array(xy, float), obj.height, yaw, 1, 100, (obj.long, obj.short), obj.period)
                       for xy, yaw in blocks]

    @classmethod
    def from_obs(cls, obs, obj):
        """A simulated copy of what the camera saw, for a dry run."""
        w = cls([], obj)
        w.blocks = list(obs)
        return w

    def observe(self):
        return list(self.blocks)

    def moved(self, block, stack_xy, level):
        old = [b for b in self.blocks if np.linalg.norm(b.xy - stack_xy) <= BASE_TOL]
        yaw = old[0].yaw if old else block.yaw
        self.blocks = [b for b in self.blocks if b is not block and all(b is not o for o in old)]
        self.blocks.append(BlockObs(np.array(stack_xy, float), level * self.obj.height, yaw, level, 100,
                                    (self.obj.long, self.obj.short), self.obj.period))


class CameraWorld:
    def __init__(self, obj):
        self.obj = obj

    def observe(self):
        rej = []
        obs = perceive_blocks(self.obj.height, footprint=(self.obj.long, self.obj.short), rejected=rej)
        for xy, why in rej:
            log.warning("ignoring something at %s: %s", np.round(xy, 3), why)
        return obs


def describe(obs):
    return ", ".join(f"({b.xy[0]:+.3f}, {b.xy[1]:+.3f}) level {b.level} top {b.top_z * 100:.1f} cm "
                     f"close-axis {np.degrees(b.yaw):+.0f} deg" for b in obs) or "nothing"


def split(obs, base_xy):
    """-> (stack block or None, loose blocks)"""
    near = [b for b in obs if np.linalg.norm(b.xy - base_xy) <= BASE_TOL]
    stack = max(near, key=lambda b: b.level) if near else None
    return stack, [b for b in obs if b not in near]


def check_gaps(pick, stack_xy, loose, min_gap, has_stack):
    """The open jaws are wide: refuse if the pick block or the stack has a neighbour too close. An empty base
    spot is not a neighbour: the pick block is gone by the time anything sits there."""
    for o in loose:
        if o is pick:
            continue
        for name, p in (("the block being moved", pick.xy), ("the stack", stack_xy)):
            if np.linalg.norm(o.xy - p) < min_gap:
                raise SystemExit(f"block at {np.round(o.xy, 3)} is only {np.linalg.norm(o.xy - p) * 100:.1f} cm "
                                 f"from {name} (need {min_gap * 100:.1f}). Spread the blocks out and rerun.")
    if has_stack and np.linalg.norm(pick.xy - stack_xy) < min_gap:
        raise SystemExit("the block to move is touching the stack: separate them and rerun.")


PLACE_HOVERS = (PLACE_HOVER, 0.025, 0.02, 0.015)   # tried in turn: high stacks leave little reach for the hover
# What the rehearsal loop tries, in order, until the sim holds up: how far above the block centre the grasp point is
# when the block is released (place_raise), the jaw opening in rad (open), and the place hovers to try (hovers).
TUNINGS = (dict(), dict(place_raise=0.0), dict(place_raise=0.0, open=1.0), dict(place_raise=0.004, open=1.0),
           dict(place_raise=0.0, open=0.9, hovers=(0.025, 0.02, 0.015)), dict(place_raise=0.012),
           dict(place_raise=0.0, open=1.4))
PLACE_YAW_TRIM = np.radians(-11.8)  # aim this far off the stack's yaw: the placed block lands about this much rotated
                            # the other way. 9 real trials at trims -8.2/-16.4/-24.6 deg, 2026-10-19: yaw error =
                            # 10.3 + 0.87 * trim (deg), 2.7 deg scatter, zero at -11.8
REFINE_XY_TOL = 0.004       # m, a placed block this close to where it should be is left alone ...
REFINE_YAW_TOL = np.radians(3.0)   # ... and so is one this close in yaw (camera noise is about 1-2 mm and 1 deg)
RELEASE_CLEAR = 0.015       # m, slide the open gripper this far from the placed block before lifting (see
                            # plan_pick_place release_clear)
SIM_HOLD = 0.08             # rad: how hard the rehearsal's jaws hold the box. A 3 cm box stalls the sim jaws at about
                            # 0.18 rad; the real run eases off from the stall, where fully closed (-0.1) squeezes far harder
SIM_SPEED = 3.0             # --show-sim plays the rehearsal this many times faster than real time


def plan_hovering(pick_c, place_c, pick_yaw, obj, place_yaw, tune=None, held_gripper=None):
    """plan_pick_place with the largest place hover (PLACE_HOVERS) the arm can reach."""
    tune = tune or {}
    err = None
    for hover in tune.get("hovers", PLACE_HOVERS):
        try:
            return plan_pick_place(pick_c, place_c, block_yaw=pick_yaw, yaw_period=obj.period, place_z=place_c[2],
                                   gripper_open=tune.get("open", OPEN_SIM), held_width=obj.short,
                                   place_hover=hover, place_yaw=place_yaw, held_gripper=held_gripper,
                                   release_clear=tune.get("clear", RELEASE_CLEAR))
        except ValueError as e:
            err = e
    raise err


def plan_step(pick, place_xy, place_cz, obj, place_yaw=None, tune=None):
    """Plan, as pick_place does, against where the model must aim for the real jaws to land. pick.top_z and
    place_cz (the block centre's height once placed, from the measured stack top) come from the camera.
    tune: one of TUNINGS. Returns (plan, (pick_aim, place_aim))."""
    tune = tune or {}
    pick_c = np.array([pick.xy[0], pick.xy[1], pick.top_z - obj.height / 2])
    place_c = np.array([place_xy[0], place_xy[1], place_cz])
    pick_aim, place_aim = aim(pick_c, 0.0), aim(place_c, 0.0)
    pick_aim[:2] += WORLD_AIM_OFFSET
    place_aim[:2] += WORLD_AIM_OFFSET
    pick_aim[2] += GRASP_Z_RAISE
    place_aim[2] += tune.get("place_raise", GRASP_Z_RAISE)
    plan = plan_hovering(pick_aim, place_aim, pick.yaw, obj, place_yaw, tune)
    return plan, (pick_aim, place_aim)


def rehearse(pick_aim, place_aim, place_level, obj, pick_yaw, stack_yaw, tune=None, show=False):
    """Same moves in MuJoCo: a box of the real size, and a fixed box standing in for the blocks already in
    the stack (sim table is z = 0, so heights are layers of obj.height). show: watch it in the MuJoCo viewer."""
    tune = tune or {}
    h = obj.height
    below = (place_level - 1) * h
    sim_pick = np.array([pick_aim[0], pick_aim[1], h / 2 + GRASP_Z_RAISE])
    sim_place = np.array([place_aim[0], place_aim[1], below + h / 2 + tune.get("place_raise", GRASP_Z_RAISE)])
    sim_plan = plan_hovering(sim_pick, sim_place, pick_yaw, obj, stack_yaw, tune, held_gripper=SIM_HOLD)
    cube0 = np.array([pick_aim[0], pick_aim[1], h / 2])
    half = (obj.short / 2, obj.long / 2, h / 2)   # the box's short axis is its local x: it lies along the yaw
    model = scene_with_support(place_aim[:2] if place_level > 1 else None, below, (half[0], half[1]),
                               cube_half=half, yaw=stack_yaw)
    if not show:
        res = execute_plan(sim_plan, cube0, place_aim, block_yaw=pick_yaw, model=model, rest_z=below + h / 2)
        return res, len(sim_plan) * PLAN_DT
    import mujoco
    import mujoco.viewer
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as v:
        class Paced:
            def sync(self):
                if v.is_running():
                    v.sync()
                    time.sleep(PLAN_DT / SIM_SPEED)
        res = execute_plan(sim_plan, cube0, place_aim, block_yaw=pick_yaw, model=model, data=data,
                           rest_z=below + h / 2, viewer=Paced())
        time.sleep(1.0)   # leave the final state on screen for a moment
    return res, len(sim_plan) * PLAN_DT


def write_overlay(points):
    """Targets for core/tools/kinect_live.py to draw on the camera image: [(label, xyz in robot frame, BGR)]."""
    (control.DIR / "overlay.json").write_text(json.dumps(
        [dict(label=l, xyz=[float(c) for c in p], color=list(c)) for l, p, c in points]))


def do_move(a, obj, pick, dest_xy, dest_cz, dest_level, dest_yaw):
    """Plan, rehearse in the sim (a.tune variants; 0 = no rehearsal) and run one block move: pick -> dest_xy with
    the block's centre at dest_cz, dest_level layers up. Returns the arm's result ('done', 'slipped', 'missed',
    'aborted'), or 'dry' when --real is off."""
    first = passed = None
    last_err = None
    if a.tune == 0:   # no rehearsal: plan the default variant and go
        try:
            passed = (plan_step(pick, dest_xy, dest_cz, obj, dest_yaw)[0], {})
        except ValueError as e:
            raise SystemExit(f"cannot plan this move: {e}")
    for epoch, tune in enumerate(TUNINGS[:a.tune], 1):
        try:
            cand, (pick_aim, place_aim) = plan_step(pick, dest_xy, dest_cz, obj, dest_yaw, tune=tune)
        except ValueError as e:
            last_err = e
            log.info("rehearsal %d/%d %s: cannot plan (%s)", epoch, a.tune, tune or "default", e)
            continue
        sim, secs = rehearse(pick_aim, place_aim, dest_level, obj, pick.yaw, dest_yaw, tune=tune, show=a.show_sim)
        log.info("rehearsal %d/%d %s: %s (%.0f s of sim, xy error %.1f cm, ended at z %.1f cm)", epoch, a.tune,
                 tune or "default", "SUCCESS" if sim["success"] else "FAILED", secs, sim["err_xy"] * 100,
                 sim["cube_xyz"][2] * 100)
        if first is None:
            first = (cand, tune)
        if sim["success"]:
            passed = (cand, tune)
            break
    if passed is None:
        if first is None:
            raise SystemExit(f"cannot plan this move: {last_err}")
        if not a.skip_rehearsal:
            raise SystemExit("no variant of this move survived the sim rehearsal; not running it on the real arm")
        log.warning("no rehearsal passed but --skip-rehearsal was given: going ahead with the default plan")
        passed = first
    plan, tune = passed

    if not a.real:
        return "dry"

    wait_to_cool()
    frames = keyframes(plan, tune.get("open", OPEN_SIM))
    write_overlay([("pick", (*pick.xy, pick.top_z), (255, 255, 0)),
                   ("place", (*dest_xy, dest_cz + obj.height / 2), (255, 0, 255))])
    print(f"\nREAL RUN: block ({pick.xy[0]:.3f}, {pick.xy[1]:.3f}) -> layer {dest_level} at "
          f"({dest_xy[0]:.3f}, {dest_xy[1]:.3f}), {len(frames)} waypoints at <= {REAL_SPEED_DEG_S} deg/s.")
    if any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)):
        raise SystemExit("a STOP/PARK/LIMP flag is set: press 'clear' in stop_gui.py first")
    if not a.yes:
        print("Clear the area; stop_gui.py STOP freezes the arm.")
        if input("Type GO to start: ").strip() != "GO":
            raise SystemExit("aborted")
    result = grasp.run_arm(frames, real=True, block_xyz=np.array([*pick.xy, pick.top_z - obj.height / 2]))
    log.info("arm result: %s", result)
    return result


def refine_place(a, world, obj, below, level, cz, nominal_yaw):
    """After placing a block on `below` (the stack as it was before), look: if the new top block is off by more than
    the tolerance, pick it up again and re-place it aimed at (target - the error just measured), up to a.refine
    times. The error of a place is largely repeatable, so subtracting what was just seen cancels it.
    UNPROVEN: on the real arm (2026-10-20) the correction overshot (10.0 mm -> 10.4 mm on the other side, yaw error
    unchanged), so the error is not a fixed bias for a re-picked block. Off by default; do not rely on it."""
    aim_xy, aim_yaw = below.xy.copy(), nominal_yaw
    for k in range(a.refine):
        time.sleep(1.0)
        top, _ = split(world.observe(), below.xy)
        if top is None or top.level != level:
            log.warning("refine: the camera does not see the block at layer %d; leaving it", level)
            return
        e = top.xy - below.xy
        ey = (top.yaw - below.yaw + obj.period / 2) % obj.period - obj.period / 2
        log.info("placement error: %.1f mm (%+.1f, %+.1f), %+.1f deg", np.linalg.norm(e) * 1000, e[0] * 1000,
                 e[1] * 1000, np.degrees(ey))
        if np.linalg.norm(e) <= REFINE_XY_TOL and abs(ey) <= REFINE_YAW_TOL:
            return
        aim_xy, aim_yaw = aim_xy - e, aim_yaw - ey
        log.info("refine %d/%d: picking the block up again, aimed at (%.3f, %.3f), yaw %+.1f deg", k + 1, a.refine,
                 *aim_xy, np.degrees(aim_yaw))
        if do_move(a, obj, top, aim_xy, cz, level, aim_yaw) != "done" or motion.ABORTED:
            return


def run(a):
    obj = Obj(a.object)
    if a.blocks:
        pts = np.array(a.blocks, float).reshape(-1, 2)
        world = SimWorld([(p, 0.0) for p in pts], obj)
    else:
        world = CameraWorld(obj)

    obs = world.observe()
    log.info("I see %d block(s): %s", len(obs), describe(obs))
    if a.look:
        return
    if not a.real and not a.blocks:
        world = SimWorld.from_obs(obs, obj)   # dry run: the moves are simulated, not looked at again
    start_level = sum(b.level for b in obs)
    if start_level < a.count:
        raise SystemExit(f"need {a.count} blocks, the camera accounts for {start_level}: {describe(obs)}. "
                         "Fix the table (blocks touching or off the mat?) or pass a smaller --count.")

    if a.base:
        base_xy = np.array(a.base, float)
    else:
        loose_first = [b for b in obs if b.level == 1] or obs
        base_xy = min(loose_first, key=lambda b: np.linalg.norm(b.xy - PREFERRED_BASE)).xy.copy()
    log.info("building the stack at (%.3f, %.3f)", *base_xy)

    stack, _ = split(obs, base_xy)
    todo = a.count - (stack.level if stack else 0)
    steps_left = todo * a.attempts
    while True:
        obs = world.observe()
        stack, loose = split(obs, base_xy)
        level = stack.level if stack else 0
        seen = level + sum(b.level for b in loose)
        log.info("stack level %d of %d; loose: %s", level, a.count, describe(loose))
        if level >= a.count:
            off = np.linalg.norm(stack.xy - base_xy)
            log.info("DONE: %d blocks stacked at (%.3f, %.3f), top at %.1f cm%s", level, *stack.xy,
                     stack.top_z * 100, "" if off <= STACK_XY_TOL else f" (WARNING {off * 100:.1f} cm off the base)")
            return stack
        if seen < a.count or any(b.level != 1 for b in loose):
            raise SystemExit(f"need {a.count} blocks, the camera accounts for {seen}: {describe(obs)}. "
                             "Something was knocked over or is out of view; stopping.")
        if steps_left <= 0:
            raise SystemExit(f"gave up: stack is {level}/{a.count} after {a.attempts} attempt(s) per block")
        steps_left -= 1

        stack_xy = stack.xy if stack else base_xy
        near = np.array(a.prefer, float) if a.prefer else stack_xy
        pick = min(loose, key=lambda b: np.linalg.norm(b.xy - near))
        check_gaps(pick, stack_xy, loose, obj.min_gap, stack is not None)
        place_level = level + 1
        # heights come from the camera: the measured top of the stack (or of the floor under the loose blocks)
        floor_z = float(np.median([b.top_z - b.level * obj.height for b in obs]))
        place_cz = (stack.top_z if stack else floor_z) + obj.height / 2
        log.info("=== block at (%.3f, %.3f) -> layer %d of the stack (centre height %.1f cm) ===", *pick.xy,
                 place_level, place_cz * 100)

        result = do_move(a, obj, pick, stack_xy, place_cz, place_level, stack.yaw + PLACE_YAW_TRIM if stack else None)
        if result == "dry":
            world.moved(pick, stack_xy, place_level)   # dry run: assume the move worked
            continue
        if result == "aborted" or motion.ABORTED:
            raise SystemExit("motion was stopped or faulted: no more attempts")
        if result == "done" and getattr(a, "refine", 0) and stack is not None:
            refine_place(a, world, obj, stack, place_level, place_cz, stack.yaw + PLACE_YAW_TRIM)
        if motion.ABORTED:   # a STOP/PARK/fault during the re-place must end the run, not start the next move
            raise SystemExit("motion was stopped or faulted: no more attempts")
        time.sleep(1.0)   # let the table settle before the next look


def make_args(**kw):
    """The command-line arguments with their defaults, for code that drives the stacker (projects.stack_blocks.trials)."""
    d = dict(count=3, object="box8x5", base=None, blocks=None, look=False, prefer=None, skip_rehearsal=False,
             show_sim=False, tune=len(TUNINGS), refine=0, real=False, yes=False, attempts=3)
    d.update(kw)
    return argparse.Namespace(**d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=3, help="blocks in the finished stack")
    ap.add_argument("--object", choices=sorted(OBJECTS), default="box8x5", help="what the blocks are")
    ap.add_argument("--base", type=float, nargs=2, metavar=("X", "Y"), help="build the stack here (default: at "
                    "the block nearest the middle of the mat)")
    ap.add_argument("--blocks", type=float, nargs="+", metavar="XY", help="x y pairs: skip the camera, sim only")
    ap.add_argument("--look", action="store_true", help="only report what the Kinect sees")
    ap.add_argument("--prefer", type=float, nargs=2, metavar=("X", "Y"), help="move the loose block nearest here "
                    "first (default: the one nearest the stack)")
    ap.add_argument("--skip-rehearsal", action="store_true", help="go ahead on the real arm even if the sim "
                    "rehearsal fails (the rehearsal still runs and is logged)")
    ap.add_argument("--show-sim", action="store_true", help="watch each sim rehearsal in the MuJoCo viewer")
    ap.add_argument("--tune", type=int, default=len(TUNINGS), metavar="N", help="rehearsal variants to try per "
                    "move (TUNINGS), stopping at the first that passes (default: all); 0 = no rehearsal at all")
    ap.add_argument("--refine", type=int, default=0, metavar="N", help="after each place onto the stack, look and, if the "
                    "block is more than ~4 mm / 3 deg off, pick it up and re-place it with the error subtracted "
                    "(up to N times)")
    ap.add_argument("--real", action="store_true", help="move the real arm (asks for confirmation)")
    ap.add_argument("--yes", action="store_true", help="skip the GO prompt (stop_gui.py must be open)")
    ap.add_argument("--attempts", type=int, default=3, help="tries per block before giving up")
    a = ap.parse_args()
    if a.blocks and (len(a.blocks) % 2 or a.real):
        ap.error("--blocks needs x y pairs and cannot be combined with --real (the camera must see the blocks)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    run(a)


if __name__ == "__main__":
    main()
