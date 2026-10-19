"""Stack two blocks, measure how well the top one sits on the base, unstack it, repeat.

    python -m projects.stack_blocks.trials --n 10 --real --yes

Two blocks lying flat on the mat (see main.py for spacing). Each trial: stack the loose block on the base block, look,
log the top block's offset from the base block (along the base's short axis, its long axis, and in yaw), then carry
the top block back to where the loose block started. The CSV (control/trials/) and the summary at the end give the
bias (the average error, which an aim correction can remove) and the scatter (what is left).

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


def unstack(a, obj, base_xy, park_xy):
    """Carry the top block of the stack at base_xy back to the table at park_xy."""
    stack, _ = sb.split(sb.CameraWorld(obj).observe(), base_xy)
    if stack is None or stack.level != 2:
        raise SystemExit("unstack: the camera does not see a two-high stack at the base spot")
    floor_z = stack.top_z - 2 * obj.height
    return sb.do_move(a, obj, stack, park_xy, floor_z + obj.height / 2, 1, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="trials")
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

    obj = sb.Obj(a.object)
    obs = sb.CameraWorld(obj).observe()
    if len(obs) != 2 or any(b.level != 1 for b in obs):
        raise SystemExit(f"need exactly two level-1 blocks on the mat, the camera sees: {sb.describe(obs)}")
    base_xy = np.array(a.base, float) if a.base else min(obs, key=lambda b: np.linalg.norm(b.xy - sb.PREFERRED_BASE)).xy
    park_xy = next(b.xy for b in obs if np.linalg.norm(b.xy - base_xy) > sb.BASE_TOL).copy()
    log.info("base at (%.3f, %.3f), the loose block starts at (%.3f, %.3f)", *base_xy, *park_xy)

    out = control.DIR / "trials"
    out.mkdir(exist_ok=True)
    path = out / time.strftime("stack_trials_%Y%m%d_%H%M%S.csv")
    cols = ["trial", "base_x", "base_y", "base_yaw_deg", "top_x", "top_y", "top_yaw_deg", "err_short_mm",
            "err_long_mm", "err_yaw_deg", "top_z_cm", "seconds", "trim_deg", "loose_yaw_deg"]
    rows = []
    sa = sb.make_args(count=2, object=a.object, base=list(base_xy), real=True, yes=a.yes, attempts=1, tune=a.tune,
                      skip_rehearsal=True)
    for i in range(1, a.n + 1):
        t0 = time.time()
        obs = sb.CameraWorld(obj).observe()
        base, _ = sb.split(obs, base_xy)
        if base is None or len(obs) != 2 or any(b.level != 1 for b in obs):
            raise SystemExit(f"trial {i}: expected two level-1 blocks, the camera sees: {sb.describe(obs)}")
        loose_yaw = next(b.yaw for b in obs if b is not base)
        trim = a.trims[(i - 1) % len(a.trims)] if a.trims else float(np.degrees(sb.PLACE_YAW_TRIM))
        sb.PLACE_YAW_TRIM = np.radians(trim)
        top = sb.run(sa)
        if top is None or top.level != 2:
            raise SystemExit(f"trial {i}: the stack did not reach two high")
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
            result = unstack(sa, obj, base_xy, park_xy)
            if result != "done" or motion.ABORTED:
                raise SystemExit(f"trial {i}: unstacking returned {result!r}; stopping")
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
