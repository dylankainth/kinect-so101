"""Exercise core.grasp.run_arm (real branch) and the pick_place wrapper against fake hardware.

    .venv\Scripts\python -m tests.test_grasp
"""
import numpy as np

from core import control, grasp
from core.motion import keyframes
from core.sim.pick_place import plan_pick_place
from tests.fake_hardware import FakeKinect, FakeSafeArm

grasp.SafeArm = FakeSafeArm
grasp.Kinect = FakeKinect
grasp.snapshot = lambda cam, name: None
grasp.time.sleep = lambda s: None

BLOCK = np.array([0.20, 0.12, 0.015])
FRAMES = keyframes(plan_pick_place(BLOCK, (0.20, -0.12, 0.015)))


def run(grip_reads=(25.0,), fail_on_follow=None, params=None):
    FakeSafeArm.reset(grip_reads, fail_on_follow)
    control.clear()
    if fail_on_follow:
        control.request(control.LIMP)   # lets hold_until_user return instead of waiting for a button
    try:
        res = grasp.run_arm(FRAMES, real=True, block_xyz=BLOCK, params=params)
    finally:
        control.clear()
    return res, FakeSafeArm.instances[0]


def main():
    res, arm = run((25.0, 25.0))
    assert res == "done", res
    assert arm.log[-3:] == ["torque_off", "exit", "exit"][-3:] or "torque_off" in arm.log, arm.log
    print("holding block  ->", res)

    res, arm = run((7.0,))
    assert res == "missed", res
    print("closed on air  ->", res)

    res, arm = run((25.0, 5.0))   # held at the close, jaws shut at the lift check => slipped
    assert res == "slipped", res
    print("slipped        ->", res)

    res, arm = run(fail_on_follow=2)
    assert res == "aborted", res
    assert "stop" in arm.log, arm.log
    print("abort mid-run  ->", res)

    res, arm = run((7.0,), params=grasp.GraspParams(tactile=False))
    print("camera mode    ->", res)

    FakeSafeArm.reset()
    assert grasp.run_arm(FRAMES, real=False, block_xyz=BLOCK) is None   # dry run only checks the trajectory
    print("dry run        -> ok")

    import projects.pick_place.run_pick_place as rp
    FakeSafeArm.reset((25.0, 25.0))
    assert rp.run_arm(FRAMES, real=True, block_xyz=BLOCK) == "done"
    rp.GRASP_MIN = 30.0             # an --object profile overriding a module tunable must reach core.grasp
    FakeSafeArm.reset((25.0,))
    assert rp.run_arm(FRAMES, real=True, block_xyz=BLOCK) == "missed"
    print("pick_place wrapper, profile override reaches core.grasp -> ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
