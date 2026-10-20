"""Stack blocks, measure how well the top one sits on the base, unstack them, repeat.

    python -m projects.stack_blocks.trials --n 10 --real --yes            # two-high
    python -m projects.stack_blocks.trials --n 3 --count 3 --real --yes   # three-high, every layer measured

--count flat blocks on the mat (see main.py for spacing). Each trial: build the stack, and after every move look and
log the top block's offset from the base block (along the base's short axis, its long axis, and in yaw), then carry
the blocks back to where they were picked from, top first. Two CSVs in control/trials/: one row per trial, and
*_moves.csv with one row per move (level, pick position, pan and wrist roll at the grasp and at the release, and the
error), for fitting the error against the geometry. The summary at the end gives the bias (the average error, which
an aim correction can remove) and the scatter (what is left).

Stops at the first trial that does not end with exactly two level-1 blocks, so a tipped or dropped block gets your
eyes before anything else moves. Keep core/tools/stop_gui.py open.
"""
import argparse
import csv
import logging
import time

import numpy as np

from core import control, motion
from projects.stack_blocks import main as sb

log = logging.getLogger("stack_trials")


def placement_error(base, top, period):
    """Offset of the top block from the base block: (m along the base's closing (short) axis, m along its long axis,
    deg of yaw, wrapped to +-period/2)."""
    s = np.array([np.cos(base.yaw), np.sin(base.yaw)])
    long_axis = np.array([-s[1], s[0]])
    e = top.xy - base.xy
    dyaw = (top.yaw - base.yaw + period / 2) % period - period / 2
    return float(e @ s), float(e @ long_axis), float(np.degrees(dyaw))


def release_geometry(frames):
    """Joint angles (deg) of the plan at the grasp and at the release."""
    g0 = frames[0]["gripper"]
    close = next((i for i, f in enumerate(frames) if f["gripper"] < g0 - 1e-6), 0)
    rel = next((i for i in range(close, len(frames)) if frames[i]["gripper"] > frames[close]["gripper"] + 5.0),
               len(frames) - 1)
    return dict(pick_pan_deg=frames[close]["shoulder_pan"], place_pan_deg=frames[rel]["shoulder_pan"],
                pick_roll_deg=frames[close]["wrist_roll"], place_roll_deg=frames[rel]["wrist_roll"],
                roll_change_deg=frames[rel]["wrist_roll"] - frames[close]["wrist_roll"],
                place_flex_deg=frames[rel]["wrist_flex"])


def unstack(a, obj, base_xy, park_xy, level=2):
    """Carry the top block of the `level`-high stack at base_xy back to the table at park_xy."""
    stack, _ = sb.split(sb.CameraWorld(obj).observe(), base_xy)
    if stack is None or stack.level != level:
        raise SystemExit(f"unstack: the camera does not see a {level}-high stack at the base spot")
    floor_z = stack.top_z - level * obj.height
    return sb.do_move(a, obj, stack, park_xy, floor_z + obj.height / 2, 1, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="trials")
    ap.add_argument("--count", type=int, default=2, help="blocks in the stack (2 or 3)")
    ap.add_argument("--prefer", type=float, nargs=2, metavar=("X", "Y"), help="move the loose block nearest here "
                    "first, so the other one is the last (top) layer")
    ap.add_argument("--min-gap", type=float, metavar="CM", help="override the centre-to-centre clearance between "
                    "blocks (default: the jaw-based one for the object); smaller risks the open jaws brushing a neighbour")
    ap.add_argument("--object", choices=sorted(sb.OBJECTS), default="box8x5")
    ap.add_argument("--base", type=float, nargs=2, metavar=("X", "Y"), help="where the stack is built (default: the "
                    "block nearest the middle of the mat)")
    ap.add_argument("--trims", type=float, nargs="+", metavar="DEG", help="place-yaw trims to cycle through, one per "
                    "trial (default: the PLACE_YAW_TRIM in main.py); the log says which one each trial used")
    ap.add_argument("--tune", type=int, default=0, help="sim rehearsal variants per move (0 = none, the default)")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the GO prompts (stop_gui.py must be open)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not a.real:
        ap.error("this drives the real arm: pass --real")

    if a.min_gap:   # Obj.min_gap = long/2 + short/2 + JAW_CLEAR, so this moves it for every Obj built from here on
        o = sb.Obj(a.object)
        sb.JAW_CLEAR = a.min_gap / 100 - (o.long + o.short) / 2
    obj = sb.Obj(a.object)
    obs = sb.CameraWorld(obj).observe()
    if len(obs) != a.count or any(b.level != 1 for b in obs):
        raise SystemExit(f"need exactly {a.count} level-1 blocks on the mat, the camera sees: {sb.describe(obs)}")
    base_xy = np.array(a.base, float) if a.base else min(obs, key=lambda b: np.linalg.norm(b.xy - sb.PREFERRED_BASE)).xy
    log.info("base at (%.3f, %.3f), the loose blocks start at %s", *base_xy,
             [tuple(np.round(b.xy, 3)) for b in obs if np.linalg.norm(b.xy - base_xy) > sb.BASE_TOL])

    out = control.DIR / "trials"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"stack_trials_{stamp}.csv"
    moves_path = out / f"stack_trials_{stamp}_moves.csv"
    move_cols = ["trial", "level", "pick_x", "pick_y", "pick_yaw_deg", "pick_pan_deg", "place_pan_deg",
                 "pick_roll_deg", "place_roll_deg", "roll_change_deg", "place_flex_deg", "err_short_mm",
                 "err_long_mm", "err_yaw_deg"]
    moves = []
    cols = ["trial", "base_x", "base_y", "base_yaw_deg", "top_x", "top_y", "top_yaw_deg", "err_short_mm",
            "err_long_mm", "err_yaw_deg", "top_z_cm", "seconds", "trim_deg", "loose_yaw_deg"]
    rows = []
    sa = sb.make_args(count=a.count, object=a.object, base=list(base_xy), real=True, yes=a.yes, attempts=1, tune=a.tune,
                      skip_rehearsal=True, prefer=a.prefer)
    cur = dict(trial=0, base=None, loose=[])
    real_run_arm = sb.grasp.run_arm

    def recording_run_arm(frames, real, block_xyz=None, params=None):
        """The arm run as usual; after a good place, look and log this move's geometry and error."""
        result = real_run_arm(frames, real, block_xyz=block_xyz, params=params)
        if result == "done" and cur["base"] is not None and block_xyz is not None:
            time.sleep(1.5)
            top, _ = sb.split(sb.CameraWorld(obj).observe(), base_xy)
            if top is not None and top.level >= 2:
                es, el, ey = placement_error(cur["base"], top, obj.period)
                src = min(cur["loose"], key=lambda b: np.linalg.norm(b.xy - block_xyz[:2]))
                moves.append(dict(trial=cur["trial"], level=top.level, pick_x=block_xyz[0], pick_y=block_xyz[1],
                                  pick_yaw_deg=np.degrees(src.yaw), err_short_mm=es * 1000, err_long_mm=el * 1000,
                                  err_yaw_deg=ey, **release_geometry(frames)))
                log.info("MOVE level %d: pick (%.3f, %.3f), roll change %+.0f deg, error %+.1f / %+.1f mm, %+.1f deg",
                         top.level, block_xyz[0], block_xyz[1], moves[-1]["roll_change_deg"], es * 1000, el * 1000, ey)
                with open(moves_path, "w", newline="") as f:
                    w = csv.DictWriter(f, move_cols)
                    w.writeheader()
                    w.writerows({k: round(float(v), 3) for k, v in r.items()} for r in moves)
        return result

    sb.grasp.run_arm = recording_run_arm
    for i in range(1, a.n + 1):
        t0 = time.time()
        obs = sb.CameraWorld(obj).observe()
        base, _ = sb.split(obs, base_xy)
        if base is None or len(obs) != a.count or any(b.level != 1 for b in obs):
            raise SystemExit(f"trial {i}: expected {a.count} level-1 blocks, the camera sees: {sb.describe(obs)}")
        loose_yaw = next(b.yaw for b in obs if b is not base)
        cur.update(trial=i, base=base, loose=[b for b in obs if b is not base])
        trim = a.trims[(i - 1) % len(a.trims)] if a.trims else float(np.degrees(sb.PLACE_YAW_TRIM))
        sb.PLACE_YAW_TRIM = np.radians(trim)
        top = sb.run(sa)
        if top is None or top.level != a.count:
            raise SystemExit(f"trial {i}: the stack did not reach {a.count} high")
        es, el, ey = placement_error(base, top, obj.period)
        row = dict(trial=i, base_x=base.xy[0], base_y=base.xy[1], base_yaw_deg=np.degrees(base.yaw), top_x=top.xy[0],
                   top_y=top.xy[1], top_yaw_deg=np.degrees(top.yaw), err_short_mm=es * 1000, err_long_mm=el * 1000,
                   err_yaw_deg=ey, top_z_cm=top.top_z * 100, seconds=time.time() - t0, trim_deg=trim,
                   loose_yaw_deg=np.degrees(loose_yaw))
        rows.append(row)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows({k: round(v, 3) for k, v in r.items()} for r in rows)
        log.info("TRIAL %d: top block is %+.1f mm along the base's short axis, %+.1f mm along its long axis, "
                 "%+.1f deg of yaw (place-yaw trim %+.1f deg)", i, es * 1000, el * 1000, ey, trim)
        if i < a.n:
            picked = [m for m in moves if m["trial"] == i]   # the block placed last goes back to where it was picked
            if len(picked) != a.count - 1:
                raise SystemExit(f"trial {i}: logged {len(picked)} moves, expected {a.count - 1}; not unstacking")
            cur["base"] = None   # the unstacking moves are not trial moves
            for level in range(a.count, 1, -1):
                home = np.array([picked[level - 2]["pick_x"], picked[level - 2]["pick_y"]])
                for dx, dy in ((0, 0), (0.01, 0), (0.01, 0.01), (0.02, 0), (0.02, 0.01), (-0.01, 0.01)):
                    try:   # the exact spot is sometimes out of the planner's reach: try a little further around
                        result = unstack(sa, obj, base_xy, home + (dx, dy), level=level)
                        break
                    except SystemExit as e:
                        if "cannot plan" not in str(e):
                            raise
                        log.warning("unstack: %s; trying another spot", e)
                else:
                    raise SystemExit(f"trial {i}: no reachable spot to unstack layer {level}")
                if result != "done" or motion.ABORTED:
                    raise SystemExit(f"trial {i}: unstacking layer {level} returned {result!r}; stopping")
                time.sleep(1.0)

    if len({r["trim_deg"] for r in rows}) > 1:
        print("\nyaw error by place-yaw trim:")
        for tr in sorted({r["trim_deg"] for r in rows}):
            ys = [r["err_yaw_deg"] for r in rows if r["trim_deg"] == tr]
            print(f"  trim {tr:+6.1f} deg: yaw error {np.mean(ys):+6.1f} deg (n={len(ys)}, spread {np.std(ys):.1f})")
    E = np.array([[r["err_short_mm"], r["err_long_mm"], r["err_yaw_deg"]] for r in rows])
    print(f"\n{len(rows)} trials, log: {path}")
    print("             mean (bias)   std (scatter)   worst")
    for name, col, unit in (("short axis", 0, "mm"), ("long axis", 1, "mm"), ("yaw", 2, "deg")):
        print(f"  {name:10s} {E[:, col].mean():+8.1f} {unit:3s}   {E[:, col].std():8.1f} {unit:3s}   "
              f"{np.abs(E[:, col]).max():6.1f} {unit}")


if __name__ == "__main__":
    main()
