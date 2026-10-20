"""Put the screws lying on the mat into the tray.

    .venv\Scripts\python -m projects.put_screws_away.main --look      # what does the camera see? (control/screws.jpg)
    .venv\Scripts\python -m projects.put_screws_away.main             # plan every pick and the drop; the arm is not touched
    .venv\Scripts\python -m projects.put_screws_away.main --real      # real arm (keep core/tools/stop_gui open)

Screws are a few mm tall, so they are found in the colour image (detect.py), not in depth. Each step is: look, take the
most isolated screw (its neighbours are the least likely to be hit by the open jaws), grasp it across its width, carry it
over the tray, let go, look again. The camera is the only judge of success: a 3 mm screw barely changes the jaw reading,
so a pick counts when that screw is gone from the mat (a dropped one shows up in the next look).
"""
import argparse
import logging
import time

import numpy as np

from core import control, grasp, motion
from core.motion import REAL_SPEED_DEG_S, aim, fixed_jaw_dir, keyframes, wait_to_cool
from core.sim.pick_place import plan_pick_place
from projects.put_screws_away.detect import draw, perceive_screws

log = logging.getLogger("screws_away")

DROP_RADIUS = 0.26                    # m from the base axis: the tray is long, release over its near end where the arm
                                     # reaches comfortably (a drop over its middle at 7 cm up was out of reach)
BASE_XY = np.array([0.0388, 0.0])   # the shoulder_pan axis
RIM_CLEAR = 0.04                     # m: the screw is released this far above the tray's rim (the camera measures the rim)
OPEN_REAL = 30.0                     # 0-100: the jaws open this far, not the usual 70: a screw needs a few cm, and the wide
                                     # swing of the moving jaw shoved the tray when it opened inside it
SCREW_Z = 0.003                      # m, the screw's axis above the mat
GRASP_Z_RAISE = 0.004                # m above the screw's axis for the grasp point (keeps the swinging jaw off the mat)
GONE_DIST = 0.015                    # m: a screw that is not within this of where it was has been picked up
NEW_DIST = 0.015                     # m: a screw this far from every earlier one is a new (dropped) one
YAW_PERIOD = np.pi                   # a screw looks the same turned 180 deg


RETRY_DIST = 0.03                    # m: a screw this close to one already tried is the same screw, nudged


def attempts_at(tried, xy):
    return max((n for p, n in tried if np.linalg.norm(p - xy) <= RETRY_DIST), default=0)


def grasp_offset(n_tried, learned):
    """The offset for the next attempt: the learned one first, then the SEARCH list (skipping the learned one)."""
    cands = [learned] + [o for o in SEARCH if o != learned]
    return cands[n_tried % len(cands)]


def isolation(s, screws):
    """Distance (m) from this screw to the nearest other one: bigger = safer to pick."""
    others = [np.linalg.norm(s.xy - o.xy) for o in screws if o is not s]
    return min(others) if others else 1.0


def pick_drop_point(floor):
    """Where over the tray to release: on its centre line (the middle of the floor across), at the point DROP_RADIUS
    from the base, kept clear of the tray's ends."""
    y0 = np.median(floor[:, 1])
    line = floor[np.abs(floor[:, 1] - y0) < 0.012]
    lo, hi = np.percentile(line[:, 0], [15, 85])
    line = line[(line[:, 0] >= lo) & (line[:, 0] <= hi)]
    r = np.linalg.norm(line - BASE_XY, axis=1)
    return line[int(np.argmin(np.abs(r - DROP_RADIUS)))]


# Grasp offsets (m) tried in turn on a screw that was missed, in the gripper's frame: along = toward the fixed jaw,
# across = along the jaw faces. The first one the jaws catch on becomes the starting offset for the next screws.
SEARCH = [(0.0, 0.0), (0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01), (0.01, 0.01), (-0.01, -0.01),
          (0.01, -0.01), (-0.01, 0.01)]
NUT_YAWS = np.radians([0, 30, -30, 60, -60, 90])   # jaw directions tried for a nut (it has no length to grasp across)


TRAY_KEEP_OUT = 0.04   # m beyond the tray's floor in x and y where nothing is ever picked (its walls are there)


def over_tray(xy, floor):
    lo, hi = floor.min(0) - TRAY_KEEP_OUT, floor.max(0) + TRAY_KEEP_OUT
    return bool(np.all(xy >= lo) and np.all(xy <= hi))


def plan_screw(s, a, tray_xy, drop_z, offset=(0.0, 0.0)):
    """Plan the pick of screw s and the drop over the tray. Returns the plan or None if it cannot be reached."""
    pick = aim(np.array([*s.xy, SCREW_Z]), 0.0) + [0, 0, a.grasp_raise]
    drop = aim(np.array([*tray_xy, drop_z]), 0.0)
    if a.kind == "nut":
        yaws = list(NUT_YAWS)
    else:   # the jaws close across the length
        yaws = [(s.yaw + np.pi / 2 + YAW_PERIOD / 2) % YAW_PERIOD - YAW_PERIOD / 2]
    err = None
    for close_yaw in yaws:
        try:
            plan = plan_pick_place(pick, drop, block_yaw=close_yaw, yaw_period=YAW_PERIOD, place_z=drop_z)
            if offset != (0.0, 0.0):   # shift the aim in the gripper's own frame and plan again
                u = fixed_jaw_dir(plan)
                v = np.array([-u[1], u[0]])
                pick2 = pick + [*(offset[0] * u + offset[1] * v), 0.0]
                plan = plan_pick_place(pick2, drop, block_yaw=close_yaw, yaw_period=YAW_PERIOD, place_z=drop_z)
            return plan
        except ValueError as e:
            err = e
    log.info("%s at (%.3f, %.3f) cannot be planned: %s", a.kind, *s.xy, err)
    return None


def mat_changed(before, after, picked):
    """-> (picked screw gone?, new screws that were not there before)"""
    gone = all(np.linalg.norm(picked.xy - s.xy) > GONE_DIST for s in after)
    new = [s for s in after if all(np.linalg.norm(s.xy - o.xy) > NEW_DIST for o in before)]
    return gone, new


def run(a):
    screws, rgb, tray = perceive_screws(kind=a.kind)
    if tray is None:
        raise SystemExit("cannot find the tray in the camera image (TRAY_ROI in detect.py)")
    floor, rim_z, _ = tray
    on_tray = [s for s in screws if over_tray(s.xy, floor)]
    if on_tray:
        log.warning("ignoring %d detection(s) over the tray: %s", len(on_tray), [tuple(np.round(s.xy, 3)) for s in on_tray])
        screws = [s for s in screws if s not in on_tray]
    tray_xy = pick_drop_point(floor)
    drop_z = rim_z + RIM_CLEAR
    log.info("tray floor centre (%.3f, %.3f), rim %.1f cm: releasing at %.1f cm", *tray_xy, rim_z * 100, drop_z * 100)
    draw(rgb, screws, control.DIR / "screws.jpg")
    log.info("I see %d screw(s): %s", len(screws), ", ".join(f"({s.xy[0]:.3f}, {s.xy[1]:+.3f})" for s in screws))
    if a.look:
        return
    done = fails = tries = 0
    learned = (0.0, 0.0)   # the grasp offset that last worked (see SEARCH)
    tried = []   # [xy, attempts]: attempts on whatever lay within RETRY_DIST of xy (a miss can nudge the screw)
    while screws and not (a.limit and tries >= a.limit):
        order = sorted(screws, key=lambda s: -isolation(s, screws))
        pick, plan = None, None
        for s in order:
            if attempts_at(tried, s.xy) >= a.attempts:
                continue
            offset = grasp_offset(attempts_at(tried, s.xy), learned) if a.search else (0.0, 0.0)
            plan = plan_screw(s, a, tray_xy, drop_z, offset)
            if plan is not None:
                pick = s
                break
        if pick is None:
            log.warning("nothing left that can be planned or has attempts remaining: %d screw(s) stay on the mat",
                        len(screws))
            break
        tries += 1
        n_try = attempts_at(tried, pick.xy) + 1
        for t in tried:
            if np.linalg.norm(t[0] - pick.xy) <= RETRY_DIST:
                t[0], t[1] = pick.xy, n_try
                break
        else:
            tried.append([pick.xy, n_try])
        log.info("grasp offset along %+.0f / across %+.0f mm", offset[0] * 1000, offset[1] * 1000)
        log.info("=== screw at (%.3f, %.3f), yaw %+.0f deg, %.1f cm from the nearest other (attempt %d) ===",
                 *pick.xy, np.degrees(pick.yaw), isolation(pick, screws) * 100, n_try)
        if not a.real:
            log.info("[DRY] planned %d samples; not moving the arm", len(plan))
            screws = [s for s in screws if s is not pick]
            continue
        wait_to_cool()
        frames = keyframes(plan)
        for f in frames:   # open less than usual (see OPEN_REAL)
            f["gripper"] = min(f["gripper"], OPEN_REAL)
        print(f"\nREAL RUN: screw ({pick.xy[0]:.3f}, {pick.xy[1]:.3f}) -> tray ({tray_xy[0]:.3f}, {tray_xy[1]:.3f}), "
              f"{len(frames)} waypoints at <= {REAL_SPEED_DEG_S} deg/s.")
        if any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)):
            raise SystemExit("a STOP/PARK/LIMP flag is set: press 'clear' in stop_gui.py first")
        if not a.yes:
            print("Clear the area; stop_gui.py STOP freezes the arm.")
            if input("Type GO to start: ").strip() != "GO":
                raise SystemExit("aborted")
        # grasp_min 0: a 3 mm screw barely moves the jaw reading, so always carry it and let the camera judge
        result = grasp.run_arm(frames, real=True, block_xyz=np.array([*pick.xy, SCREW_Z]),
                               params=grasp.GraspParams(grasp_min=0.0, grip_squeeze=a.squeeze))
        if result == "aborted" or motion.ABORTED:
            raise SystemExit("motion was stopped or faulted: no more attempts")
        time.sleep(1.0)
        after, rgb, tray_after = perceive_screws(kind=a.kind)
        if tray_after is not None:
            after = [s for s in after if not over_tray(s.xy, tray_after[0])]
        draw(rgb, after, control.DIR / "screws_after.jpg")
        gone, new = mat_changed(screws, after, pick)
        if new:   # often a screw that was hidden or merged with the picked one: only noted, the next look handles it
            log.info("%d screw(s) newly seen on the mat (uncovered, or dropped on the way)", len(new))
        if gone:
            if a.search and offset != learned:
                learned = offset
                log.info("LEARNED grasp offset along %+.0f / across %+.0f mm", offset[0] * 1000, offset[1] * 1000)
            done += 1
            log.info("PICKED: that screw is off the mat (%d done, %d left on the mat)", done, len(after))
        else:
            fails += 1
            log.warning("NOT PICKED: still on the mat")
        screws = after
    log.info("finished: %d picked, %d failed attempts, %d screw(s) left on the mat", done, fails, len(screws))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("bolt", "nut"), default="bolt", help="what is on the mat: long bolts/screws, or flat nuts")
    ap.add_argument("--search", action="store_true", help="on a miss, retry with the grasp shifted 1 cm along/across the jaws")
    ap.add_argument("--look", action="store_true", help="only report what the camera sees")
    ap.add_argument("--real", action="store_true", help="move the real arm")
    ap.add_argument("--yes", action="store_true", help="skip the GO prompts (stop_gui.py must be open)")
    ap.add_argument("--attempts", type=int, default=3, help="tries per screw before it is left alone")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many pick attempts (0 = no limit)")
    ap.add_argument("--grasp-raise", type=float, default=GRASP_Z_RAISE, metavar="M",
                    help="m above the screw's axis for the grasp point")
    ap.add_argument("--squeeze", type=float, default=2.0, help="0-100 units commanded inside the contact point")
    ap.add_argument("--speed", type=float, default=1.0, metavar="X",
                    help="speed-up over the usual pace (1.0 = ~25 deg/s; the servos top out near 60 deg/s)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not 1.0 <= a.speed <= 2.0:
        raise SystemExit("--speed must be between 1 and 2")
    if a.speed > 1.0:
        from core.robot import arm as arm_mod
        motion.TIME_SCALE /= a.speed
        motion.REAL_SPEED_DEG_S *= a.speed
        arm_mod.GOAL_VELOCITY = int(arm_mod.GOAL_VELOCITY * a.speed)
        log.info("speed x%.2f: plans at %.1fx time, moves capped at %.0f deg/s", a.speed, motion.TIME_SCALE,
                 motion.REAL_SPEED_DEG_S)
    run(a)


if __name__ == "__main__":
    main()
