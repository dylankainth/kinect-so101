# kinect-so101

SO-101 arm + Kinect v1, with the reusable pieces in `core/` and each application in `projects/`.

```
core/
  robot/         SafeArm (servo bus, safety limits), sim <-> real joint mapping, read-only check
  perception/    Kinect driver, block detection/pose, camera calibration, success check, util
  sim/           MuJoCo SO-101 model, IK, scripted pick-and-place planner, randomised eval
  calibration/   calib_joints (joint offsets + camera from depth), handeye_auto, probe_pose
  tools/         stop_gui (STOP/PARK/LIMP), kinect_live, view_kinect, arm_readonly
  motion.py      plan -> real keyframes, chunked moves, hold/park loop, aim correction, cool-down
  grasp.py       run one planned pick-and-place on the real arm with the grasp checks
  control.py     stop/park/limp flag files shared by stop_gui and runners
  paths.py       repo, calibration and runtime-data locations
projects/
  pick_place/    Kinect-guided pick-and-place with retries (see its RESULTS.md)
  stack_blocks/  stack 3 blocks using depth-based block finding (see its README)
  template/      starting point for a new project
tests/           no-hardware tests: fake arm/camera, synthetic depth scenes
markers/         printable ArUco calibration marker
control/         runtime state, logs, trial evidence (gitignored)
```

Run everything from the repo root, as modules, with the venv:

```
.venv\Scripts\python -m projects.pick_place.run_pick_place --help
.venv\Scripts\python -m core.sim.pick_place            # sim only, no hardware
.venv\Scripts\python -m core.calibration.calib_joints  # redo when the arm mount moves
.venv\Scripts\pythonw -m core.tools.stop_gui           # keep open during real runs
```

Tests (no hardware needed): `.venv\Scripts\python -m tests.test_grasp`, `-m tests.test_multi_block`,
`-m tests.test_stack_blocks`.

Setup and calibration details: `core/robot/README.md`, `core/perception/README.md`, `core/sim/README.md`.
