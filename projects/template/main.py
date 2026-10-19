"""Starting point for a new project. Copy this folder to projects/<your_project>/ and edit.

    python -m projects.template.main                 # look, plan, rehearse in sim; the arm is not touched
    python -m projects.template.main --block 0.20 0.12 0.015   # skip the camera

This only uses what core already provides: Kinect perception, the MuJoCo planner and rehearsal.
To move the real arm, stream `keyframes(plan)` through `SafeArm` as projects/pick_place/run_pick_place.py
does (see run_arm there), and keep the STOP/PARK/LIMP handling from core.control.
"""
import argparse
import logging

import numpy as np

from core.perception.util import perceive_block
from core.sim.pick_place import plan_pick_place, execute_plan

log = logging.getLogger("template")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=float, nargs=3, metavar=("X", "Y", "Z"), help="skip perception")
    ap.add_argument("--target", type=float, nargs=2, default=(0.20, -0.12), metavar=("X", "Y"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    if a.block:
        block, yaw = np.array(a.block, float), 0.0
    else:
        block, yaw = perceive_block()
    target = np.array([a.target[0], a.target[1], block[2]])

    plan = plan_pick_place(block, target, block_yaw=yaw)
    result = execute_plan(plan, block, target, block_yaw=yaw)
    log.info("sim rehearsal: %s (xy error %.1f cm)", "SUCCESS" if result["success"] else "FAILED",
             result["err_xy"] * 100)


if __name__ == "__main__":
    main()
