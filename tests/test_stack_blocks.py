"""Control flow of projects.stack_blocks in --real mode, with a scripted world instead of hardware.
(Planning and the sim rehearsal are stubbed; running the project without --real covers them.)

    .venv\Scripts\python -m tests.test_stack_blocks
"""
import argparse
import logging

import numpy as np

from core import control
from projects.stack_blocks import main as sm

OBJ = sm.Obj("cube3")
sm.wait_to_cool = lambda: None
sm.time.sleep = lambda s: None
sm.rehearse = lambda *a, **k: (dict(success=True, err_xy=0.001, cube_xyz=np.array([0, 0, 0.045])), 20.0)
sm.keyframes = lambda plan, open_sim_rad=None: plan      # the plan is only passed through to the (fake) arm
sm.plan_step = lambda pick, place_xy, place_cz, obj, place_yaw=None, tune=None: ([], (np.zeros(3), np.zeros(3)))   # planning takes ~30 s; the dry run covers it


def args(**kw):
    d = dict(count=3, object="cube3", base=None, blocks=None, look=False, real=True, yes=True, attempts=3,
             prefer=None, skip_rehearsal=False, show_sim=False, tune=1, refine=0)
    d.update(kw)
    return argparse.Namespace(**d)


def run(positions, results, stop_flag=False, obj=OBJ, **kw):
    """positions: loose block xy list. results: what each arm run returns ('done', 'missed', ...)."""
    world = sm.SimWorld([(p, 0.0) for p in positions], obj)
    sm.CameraWorld = lambda o: world
    calls = []

    def fake_run_arm(frames, real, block_xyz=None, params=None):
        res = results[min(len(calls), len(results) - 1)]
        calls.append(res)
        if res == "done":
            blk = min((b for b in world.blocks), key=lambda b: np.linalg.norm(b.xy - block_xyz[:2]))
            stack, _ = sm.split(world.blocks, world.base)
            lvl = (stack.level if stack else 0) + 1
            world.moved(blk, world.base, lvl)
        return res

    sm.grasp.run_arm = fake_run_arm
    sm.motion.ABORTED = False
    control.clear()
    if stop_flag:
        control.request(control.STOP)
    world.base = None
    real_split = sm.split

    def split(obs, base_xy):               # remember the base spot the project chose
        world.base = base_xy
        return real_split(obs, base_xy)

    sm.split = split
    try:
        sm.run(args(object=obj.name, **kw))
        return "ok", calls
    except SystemExit as e:
        return str(e), calls
    finally:
        sm.split = real_split


def refine_case(top_err, lands, refine=2, camera_sees_top=True, abort=False):
    """refine_place with a base box at the origin and a top box off by top_err (x, y, yaw). `lands(aim_xy, aim_yaw)`
    -> where the re-placed box ends up as (xy, yaw). Returns the (aim_xy, aim_yaw) of every re-place."""
    base = sm.BlockObs(np.zeros(2), OBJ.height, 0.0, 1, 100, (OBJ.long, OBJ.short), OBJ.period)
    top = sm.BlockObs(np.array(top_err[:2], float), 2 * OBJ.height, top_err[2], 2, 100, (OBJ.long, OBJ.short), OBJ.period)
    state = dict(top=top)
    aims = []

    class World:
        def observe(self):
            return [base, state["top"]] if camera_sees_top else [base]

    def fake_move(a, obj, pick, xy, cz, level, yaw):
        aims.append((np.array(xy), yaw))
        xy2, yaw2 = lands(np.array(xy), yaw)
        state["top"] = sm.BlockObs(xy2, 2 * OBJ.height, yaw2, 2, 100, (OBJ.long, OBJ.short), OBJ.period)
        sm.motion.ABORTED = abort
        return "done"

    real_move, sm.do_move = sm.do_move, fake_move
    sm.motion.ABORTED = False
    try:
        sm.refine_place(argparse.Namespace(refine=refine), World(), OBJ, base, 2, 0.045, 0.0)
    finally:
        sm.do_move = real_move
        sm.motion.ABORTED = False
    return aims


def test_refine():
    perfect = lambda xy, yaw: (xy, yaw)
    assert not refine_case((0.002, -0.001, 0.02), perfect)
    print("refine: within tolerance      -> block left alone")

    bias = lambda xy, yaw: (xy + [-0.0088, 0.001], yaw + np.radians(-3.5))   # the arm's repeatable placement error
    aims = refine_case((-0.0088, 0.001, np.radians(-3.5)), bias)
    assert len(aims) == 1, aims
    assert np.allclose(aims[0][0], [0.0088, -0.001]) and np.isclose(aims[0][1], np.radians(3.5)), aims
    print("refine: 9 mm / 3.5 deg off    -> one re-place aimed at (target - error)")

    assert not refine_case((0.02, 0, 0), perfect, camera_sees_top=False)
    print("refine: top block not seen    -> nothing moved")

    stubborn = lambda xy, yaw: (np.array([0.02, 0.0]), 0.0)   # the box always lands 2 cm off
    assert len(refine_case((0.02, 0, 0), stubborn, refine=2)) == 2
    print("refine: never converges       -> stops after --refine attempts")

    assert len(refine_case((0.02, 0, 0), stubborn, refine=3, abort=True)) == 1
    print("refine: abort flag in re-place -> no further attempt")

    # the run must end, not go on to the next block, when a re-place sets the abort flag
    real_refine = sm.refine_place
    sm.refine_place = lambda *a, **k: setattr(sm.motion, "ABORTED", True)
    try:
        out, calls = run([(0.18, 0.08), (0.24, -0.08), (0.21, -0.02)], ["done"], refine=1)
    finally:
        sm.refine_place = real_refine
        sm.motion.ABORTED = False
    assert "stopped or faulted" in out and len(calls) == 1, (out, calls)
    print("refine: abort after refine    -> run ends:", out)


def main():
    logging.basicConfig(level=logging.WARNING)
    P = [(0.18, 0.08), (0.24, -0.08), (0.21, -0.02)]

    out, calls = run(P, ["done"])
    assert out == "ok" and calls == ["done", "done"], (out, calls)
    print("clean run              -> 2 moves, stack of 3")

    out, calls = run(P, ["missed", "done", "slipped", "done", "done"])
    assert out == "ok" and calls == ["missed", "done", "slipped", "done"], (out, calls)
    print("miss + slip, retried   ->", calls)

    out, calls = run(P, ["missed"])
    assert "gave up" in out and len(calls) == 6, (out, calls)
    print("always missing         ->", out)

    out, calls = run(P, ["aborted"])
    assert "stopped or faulted" in out and calls == ["aborted"], (out, calls)
    print("arm aborted            ->", out)

    out, calls = run(P[:2], ["done"])
    assert "accounts for 2" in out and not calls, out
    print("a block missing        ->", out)

    out, calls = run([(0.18, 0.08), (0.20, 0.06), (0.21, -0.02)], ["done"])
    assert "Spread the blocks" in out and not calls, out
    print("neighbour too close    ->", out)

    out, calls = run(P, ["done"], base=(0.30, -0.18), count=3)
    assert out == "ok" and len(calls) == 3, (out, calls)
    print("stack at a given spot  ->", out, calls)

    try:
        out, calls = run(P, ["done"], stop_flag=True)
    finally:
        control.clear()
    assert "STOP/PARK/LIMP flag" in out and not calls, out
    print("STOP flag set          ->", out)
    box = sm.Obj("box8x5")
    out, calls = run([(0.327, -0.18), (0.333, -0.058), (0.355, 0.038)], ["done"], obj=box, base=(0.26, -0.06))
    assert out == "ok" and calls == ["done"] * 3, (out, calls)
    print("8 x 5 cm boxes as on the table, stack at a given spot -> ok", calls)

    out, calls = run([(0.327, -0.18), (0.333, -0.12), (0.355, 0.038)], ["done"], obj=box, base=(0.26, -0.06))
    assert "Spread the blocks" in out and not calls, out
    print("boxes side by side too close ->", out)
    test_refine()
    print("ALL OK")


if __name__ == "__main__":
    main()
