"""Put the screws lying on the mat into the tray.

    .venv\Scripts\python -m projects.put_screws_away.main --look      # what does the camera see? (control/screws.jpg)
    .venv\Scripts\python -m projects.put_screws_away.main             # plan every pick and the drop; the arm is not touched
    .venv\Scripts\python -m projects.put_screws_away.main --real      # real arm (keep core/tools/stop_gui open)

Screws are a few mm tall, so they are found in the colour image (detect.py), not in depth. Each step is: look, take the
most isolated screw (its neighbours are the least likely to be hit by the open jaws), grasp it across its width, carry it
over the tray, let go, look again. The camera is the only judge of success: a 3 mm screw barely changes the jaw reading,
so a pick counts when that screw is gone from the mat and no new one appeared (a dropped one).
"""
import argparse
import logging
import time

import numpy as np

from core import control, grasp, motion
from core.motion import REAL_SPEED_DEG_S, aim, keyframes, wait_to_cool
from core.sim.pick_place import plan_pick_place
from projects.put_screws_away.detect import draw, perceive_screws

log = logging.getLogger("screws_away")

TRAY_XY = np.array([0.277, 0.147])   # robot frame, the middle of the tray's floor (from the camera; check with --look)
DROP_Z = 0.06                        # m, height of the screw when released: well above the tray's rim, it falls in
SCREW_Z = 0.002                      # m, the screw's axis above the mat
GRASP_Z_RAISE = 0.004                # m above the screw's axis for the grasp point (keeps the swinging jaw off the mat)
GONE_DIST = 0.015                    # m: a screw that is not within this of where it was has been picked up
NEW_DIST = 0.015                     # m: a screw this far from every earlier one is a new (dropped) one
YAW_PERIOD = np.pi                   # a screw looks the same turned 180 deg


def isolation(s, screws):
    """Distance (m) from this screw to the nearest other one: bigger = safer to pick."""
    others = [np.linalg.norm(s.xy - o.xy) for o in screws if o is not s]
    return min(others) if others else 1.0


def plan_screw(s, a):
    """Plan the pick of screw s and the drop over the tray. Returns the plan or None if it cannot be reached."""
    close_yaw = (s.yaw + np.pi / 2 + YAW_PERIOD / 2) % YAW_PERIOD - YAW_PERIOD / 2   # the jaws close across the length
    pick = aim(np.array([*s.xy, SCREW_Z]), 0.0) + [0, 0, a.grasp_raise]
    drop = aim(np.array([*TRAY_XY, DROP_Z]), 0.0)
    try:
        return plan_pick_place(pick, drop, block_yaw=close_yaw, yaw_period=YAW_PERIOD, place_z=DROP_Z)
    except ValueError as e:
        log.info("screw at (%.3f, %.3f) cannot be planned: %s", *s.xy, e)
        return None


def mat_changed(before, after, picked):
    """-> (picked screw gone?, new screws that were not there before)"""
    gone = all(np.linalg.norm(picked.xy - s.xy) > GONE_DIST for s in after)
    new = [s for s in after if all(np.linalg.norm(s.xy - o.xy) > NEW_DIST for o in before)]
    return gone, new


def run(a):
    screws, rgb = perceive_screws()
    draw(rgb, screws, control.DIR / "screws.jpg")
    log.info("I see %d screw(s): %s", len(screws), ", ".join(f"({s.xy[0]:.3f}, {s.xy[1]:+.3f})" for s in screws))
    if a.look:
        return
    done = fails = 0
    tried = {}   # rounded xy -> attempts
    while screws:
        order = sorted(screws, key=lambda s: -isolation(s, screws))
        pick, plan = None, None
        for s in order:
            if tried.get(tuple(np.round(s.xy, 2)), 0) >= a.attempts:
                continue
            plan = plan_screw(s, a)
            if plan is not None:
                pick = s
                break
        if pick is None:
            log.warning("nothing left that can be planned or has attempts remaining: %d screw(s) stay on the mat",
                        len(screws))
            break
        key = tuple(np.round(pick.xy, 2))
        tried[key] = tried.get(key, 0) + 1
        log.info("=== screw at (%.3f, %.3f), yaw %+.0f deg, %.1f cm from the nearest other (attempt %d) ===",
                 *pick.xy, np.degrees(pick.yaw), isolation(pick, screws) * 100, tried[key])
        if not a.real:
            log.info("[DRY] planned %d samples; not moving the arm", len(plan))
            screws = [s for s in screws if s is not pick]
            continue
        wait_to_cool()
        frames = keyframes(plan)
        print(f"\nREAL RUN: screw ({pick.xy[0]:.3f}, {pick.xy[1]:.3f}) -> tray ({TRAY_XY[0]:.3f}, {TRAY_XY[1]:.3f}), "
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
        after, rgb = perceive_screws()
        draw(rgb, after, control.DIR / "screws_after.jpg")
        gone, new = mat_changed(screws, after, pick)
        if gone and not new:
            done += 1
            log.info("PICKED: that screw is off the mat (%d done, %d left on the mat)", done, len(after))
        else:
            fails += 1
            log.warning("NOT PICKED (%s)%s", "still on the mat" if not gone else "gone",
                        f", {len(new)} new screw(s) on the mat: dropped on the way" if new else "")
        screws = after
    log.info("finished: %d picked, %d failed attempts, %d screw(s) left on the mat", done, fails, len(screws))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--look", action="store_true", help="only report what the camera sees")
    ap.add_argument("--real", action="store_true", help="move the real arm")
    ap.add_argument("--yes", action="store_true", help="skip the GO prompts (stop_gui.py must be open)")
    ap.add_argument("--attempts", type=int, default=3, help="tries per screw before it is left alone")
    ap.add_argument("--grasp-raise", type=float, default=GRASP_Z_RAISE, metavar="M",
                    help="m above the screw's axis for the grasp point")
    ap.add_argument("--squeeze", type=float, default=2.0, help="0-100 units commanded inside the contact point")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    run(a)


if __name__ == "__main__":
    main()
