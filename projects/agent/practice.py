"""Practise stacking: put one flat box on another, measure how well it sits, take it off again, repeat. No language model.

    .venv\Scripts\python -m projects.agent.practice                       # look, plan the first stack, arm untouched
    .venv\Scripts\python -m projects.agent.practice --real --n 5          # real arm (keep core/tools/stop_gui open)

It drives the same skills the planner uses (SkillBox.move_object), so every attempt is logged and learned from
(experience.py: placement bias, release strategy, grasp offset). Lay two 8 x 5 x 3 cm boxes flat on the mat, long sides
along x, at least 9 cm apart, around x 0.25-0.30 (see projects/stack_blocks/README.md). A tipped box needs a person to
lay it flat again, so the run stops there and says what it saw; it also stops on any arm fault or STOP.
"""
import argparse
import logging
import time

import numpy as np

from core import control, motion
from projects.agent import grounding
from projects.agent.skills import SET_DOWN, SkillBox

log = logging.getLogger("practice")

BOX_TOP = 0.03          # m: a flat box's top face is 3 cm above the table
BOX_SIZE = (0.08, 0.05)
PREFERRED_BASE = np.array([0.26, -0.05])
MIN_GAP = 0.09          # m centre to centre: the open jaws are wide
UNSTACK_SPOTS = ((0, 0), (0.01, 0), (0.01, 0.01), (0.02, 0), (0.02, 0.01), (-0.01, 0.01))   # m around where it was picked from


def ignorable(o):
    """Small hardware on the mat (bolts, nuts) and the parked gripper (x below about 0.20 near the middle) are not boxes."""
    return o.length < 0.035 or (o.xy[0] < 0.20 and abs(o.xy[1]) < 0.09)


def flat_boxes(objs):
    """The objects that look like an 8 x 5 x 3 cm box lying flat on the table, and the box-sized ones that do not."""
    good, odd = [], []
    for o in objs:
        if ignorable(o):
            continue
        ok = (abs(o.top_z - BOX_TOP) < 0.008 and not o.flat and abs(o.length - BOX_SIZE[0]) < 0.02
              and abs(o.width - BOX_SIZE[1]) < 0.015)
        (good if ok else odd).append(o)
    return good, odd


def table_state(box):
    objs = box.observe()
    good, odd = flat_boxes(objs)
    if odd or len(good) != 2:
        raise SystemExit("need exactly two flat 8 x 5 x 3 cm boxes on the mat (a box on its edge or end, or two touching, "
                         "cannot be used). The camera sees:\n" + "\n".join(o.describe() for o in objs))
    base, loose = sorted(good, key=lambda o: np.linalg.norm(o.xy - PREFERRED_BASE))
    if np.linalg.norm(base.xy - loose.xy) < MIN_GAP:
        raise SystemExit(f"the boxes are {np.linalg.norm(base.xy - loose.xy) * 100:.1f} cm apart; the open jaws need "
                         f"{MIN_GAP * 100:.0f}. Spread them out.")
    return base, loose


def check_flags():
    if any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)) or motion.ABORTED:
        raise SystemExit("a STOP/PARK/LIMP flag is set or the arm faulted: stopping")


def trial(box, i):
    base, loose = table_state(box)
    home = loose.xy.copy()
    log.info("TRIAL %d: stack %s at (%.3f, %.3f) on %s at (%.3f, %.3f)", i, loose.id, *loose.xy, base.id, *base.xy)
    row = dict(trial=i)
    for attempt in range(1, 4):
        check_flags()
        res = box.call("move_object", dict(object_id=loose.id, to_object=base.id, how=SET_DOWN))
        log.info("  stack attempt %d: %s %s", attempt, "OK" if res.ok else "FAILED", res.text)
        if res.ok or "grasp" not in res.text:
            break
        base, loose = table_state(box)   # a missed grasp may have nudged the box: look again and retry
    row["stack_ok"] = res.ok
    row["stack_text"] = res.text
    if not res.ok:
        return row
    for dx, dy in UNSTACK_SPOTS:
        check_flags()
        tops = [o for o in box.observe() if o.base_z > 0]
        if not tops:
            row["unstack_ok"], row["unstack_text"] = False, "the camera does not see a stack to take apart"
            return row
        top = tops[0]
        res = box.call("move_object", dict(object_id=top.id, to_xy=[home[0] + dx, home[1] + dy], how=SET_DOWN))
        log.info("  unstack to (%.3f, %.3f): %s %s", home[0] + dx, home[1] + dy, "OK" if res.ok else "FAILED", res.text)
        if res.ok or "not reachable" not in res.text:
            break
    row["unstack_ok"], row["unstack_text"] = res.ok, res.text
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="stack/unstack trials")
    ap.add_argument("--real", action="store_true", help="move the real arm (default: plan only)")
    ap.add_argument("--stop-after", type=int, default=2, help="stop after this many failed trials in a row")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    box = SkillBox(grounding.DepthGrounder(), real=a.real)
    log.info("experience so far: %s", box.exp.summary())
    if not a.real:
        base, loose = table_state(box)
        res = box.call("move_object", dict(object_id=loose.id, to_object=base.id, how=SET_DOWN))
        print(("OK: " if res.ok else "FAILED: ") + res.text)
        return
    rows, failed_in_a_row = [], 0
    for i in range(1, a.n + 1):
        rows.append(trial(box, i))
        r = rows[-1]
        failed_in_a_row = 0 if r.get("stack_ok") and r.get("unstack_ok") else failed_in_a_row + 1
        log.info("after trial %d: %s", i, box.exp.summary())
        if failed_in_a_row >= a.stop_after:
            log.warning("%d failed trials in a row: stopping so the table can be reset and the log read", failed_in_a_row)
            break
        time.sleep(1.0)
    ok = sum(bool(r.get("stack_ok") and r.get("unstack_ok")) for r in rows)
    print(f"\n{ok} of {len(rows)} trials stacked and unstacked cleanly")
    for r in rows:
        print(f"  trial {r['trial']}: stack {'ok' if r.get('stack_ok') else 'FAILED'}: {r.get('stack_text', '')}")


if __name__ == "__main__":
    main()
