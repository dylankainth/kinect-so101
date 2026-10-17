# SO-101 pick-and-place simulation (MuJoCo)

Scripted pick-and-place of a 3 cm cube (left side -> right side of the table) using the official SO-101
MJCF. Everything lives in `sim/`.

```
sim/
  assets/so101/            official model: so101_new_calib.xml (untouched), meshes, upstream README
                           so101_sim.xml = generated copy (see "Model changes")
  build_model.py           regenerates so101_sim.xml from so101_new_calib.xml
  scene.xml                table + SO-101 + cube + target marker
  kinematics.py            FK + damped-least-squares IK (MuJoCo Jacobians), joint-range aware
  pick_place.py            plan_pick_place(), execute_plan()
  eval.py                  50-trial domain-randomised evaluation
```

## Run

Use the repo venv (`mujoco 3.x`, `numpy`). Scripts import each other by plain module name, so run them as scripts:

```
.venv\Scripts\python sim\pick_place.py            # one nominal trial, prints success
.venv\Scripts\python sim\pick_place.py --view     # same, in the MuJoCo viewer
.venv\Scripts\python sim\eval.py --n 50 --seed 0  # randomised evaluation (about 1 min)
.venv\Scripts\python sim\kinematics.py            # FK/IK smoke test
```

API:

```python
from pick_place import plan_pick_place, execute_plan, PLAN_DT
plan = plan_pick_place(block_xyz=(0.20, 0.12, 0.015), target_xyz=(0.20, -0.12, 0.015), block_yaw=0.0)
# plan: list of dicts, one per PLAN_DT = 0.02 s (50 Hz), keys:
#   shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper   (all radians)
result = execute_plan(plan, (0.20, 0.12, 0.015), (0.20, -0.12, 0.015), latency_steps=0)
# -> dict(success, cube_xyz, err_xy, on_table, max_cube_height)
```

`block_xyz` / `target_xyz` are the cube **centre** (z = half the cube size, 0.015 for 3 cm). `block_yaw` is the
cube's yaw about world z (rad; only yaw mod 90 deg matters). Phases: home -> pre-grasp (grasp point 6.5 cm above
the rest height) -> straight descent -> close -> lift -> transfer above target -> descend -> open -> retreat ->
home. The plan is ~20 s long (descents at 5 cm/s, transfer 12 cm/s, joint moves < 0.75 rad/s, minimum-jerk).
`execute_plan` plays the stream with 10 physics steps (2 ms) per waypoint, then settles 1 s, and reports
success = cube centre within 3 cm (xy) of the target **and** resting flat on the table.

## Coordinate frame

* Origin = base frame of the robot (`base` body), on the table surface. Table top is z = 0.
* x forward (away from the robot), y left, z up. Metres. Right-handed.
* Shoulder-pan axis is at (0.0388, 0) in this frame.
* Nominal task: cube at (0.20, +0.12), target at (0.20, -0.12), cube centre z = 0.015.

## Joints, units, zero pose

All angles are MJCF joint coordinates in **radians**, exactly the keys `shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper`. Position actuators (`kp`=998, `kv`=2.7, force limit +-3.35 N m, from the upstream
model) take the same radians. Ranges (rad / deg):

| joint | range rad | range deg | positive direction (verified with FK) |
|---|---|---|---|
| shoulder_pan  | -1.920 .. 1.920 | -110 .. 110 | **clockwise seen from above** (pan axis points to world -z): +pan moves the gripper toward -y (right). Cube on the left (y>0) => negative pan. |
| shoulder_lift | -1.745 .. 1.745 | -100 .. 100 | arm pitches **forward/down** (gripper tip descends) |
| elbow_flex    | -1.69 .. 1.69   | -96.8 .. 96.8 | forearm pitches down (same sense as lift) |
| wrist_flex    | -1.658 .. 1.658 | -95 .. 95 | gripper pitches down (same sense) |
| wrist_roll    | -2.744 .. 2.841 | -157 .. 163 | rotation about the gripper approach axis; with the tool pointing down, + roll turns the jaws counter-clockwise seen from above |
| gripper       | -0.1745 .. 1.745 | -10 .. 100 | + = jaw **opens** |

Zero pose (all joints 0; this is the "middle of range" pose of the `so101_new_calib` model): upper arm vertical,
forearm horizontal pointing forward, gripper horizontal pointing forward; grasp point at (0.381, 0, 0.234).
Lift, elbow and wrist_flex all rotate the chain the same way, so the tool points straight down when
`lift + elbow + wrist_flex = +pi/2` (+90 deg); e.g. the nominal pick pose is
pan -38.7, lift 6.5, elbow 22.7, wrist_flex 60.8, roll -34.1 deg (sum = 90).

Gripper values used: `GRIPPER_OPEN = 0.8 rad` (46 deg, ~4.5 cm gap between pads), `GRIPPER_CLOSED = -0.10 rad`
(commanded slightly under the fully-closed limit -0.17 so the jaw stalls on the cube; the 3.35 N m force limit
saturates, i.e. a firm grip). Note the upstream README: in LeRobot the gripper is a 0..100 value
(0 = closed, 100 = open), which the MJCF does **not** model.

### Mapping sim angles to real servo degrees

What I verified in the installed lerobot (`robots/so_follower/so_follower.py`, `motors/motors_bus.py`) and the
upstream SO-ARM100 `Simulation/SO101/README.md`:

* Upstream: `so101_new_calib` = "each joint's virtual zero is set to the **middle of its joint range**" (the old
  calibration used the fully-extended pose). The ranges in the MJCF are symmetric about that zero (elbow range
  includes a 5 deg calibration offset).
* LeRobot calibration: you hold the arm at the middle of its range, `set_half_turn_homings()` makes that raw
  position 2047 (the half turn), then you sweep every joint (except wrist_roll, fixed to 0..4095) to record
  `range_min/range_max`; `drive_mode` = 0 for all joints.
* With `use_degrees=True` the body joints read `deg = (raw - (range_min+range_max)/2) * 360/4095`, i.e. 0 deg =
  **middle of the recorded range** = the sim zero **if** your recorded range is symmetric about the calibration
  pose like the model's. With the default `RANGE_M100_100`, `raw = range_min + (v+100)/200*(range_max-range_min)`
  and then use the formula above (read min/max from the calibration JSON in
  `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<id>.json`).
* Therefore: `servo_deg = degrees(sim_rad)` per joint, same sign, if the real calibration matches the model. Two
  things I **could not verify without hardware**: (1) the direction of positive rotation of each servo (confirm
  on the real arm with one small move per joint: +lift should lower the gripper forward, +pan should swing it to
  the right, +gripper should open); flip signs if not. (2) That the user's recorded ranges are centred; if a
  range is asymmetric, `sim_deg = raw_deg_from_2047` is still right (position relative to the 2047 calibration
  pose) but the `use_degrees` mid-range offset differs, so prefer `(raw - 2047)*360/4095` when you have raw
  ticks.
* Gripper: LeRobot value `g` in 0..100 (0 = closed at recorded min, 100 = open at recorded max) ->
  `sim_rad ~= -0.1745 + g/100 * (1.745 + 0.1745)` assuming the recorded range spans the model range
  (-10 deg .. 100 deg); otherwise calibrate the two end points. For grasping you only need "open enough for 3 cm"
  (sim 0.8 rad) and "closed on the cube".

## Model changes (assets/so101/so101_sim.xml, generated by `build_model.py`)

The upstream file `so101_new_calib.xml` is kept unmodified. The sim copy:
1. disables collision on the convex-hull meshes of the fixed and moving jaw (MuJoCo collides meshes as convex
   hulls, which would fill the gap between the jaws);
2. adds two box fingertip pads (`fixed_pad`, `moving_pad`; 5 x 14 x 35 mm) with friction 1.5 (condim 4, torsional
   0.02) as the TPU-tip stand-in; randomised 1.0-1.5 in eval;
3. adds the site `grasp_point` = the point between the pads (gripper frame (0.012, 0, -0.088) m), which IK
   places at the cube centre. It is ~12 mm off the wrist-roll axis, so wrist_roll moves it; IK takes roll into account.
4. sets `meshdir` for use from `sim/scene.xml`.

The scene sets `cone=elliptic, impratio=10, noslip_iterations=3` for stable grasping, timestep 2 ms. Cube: 3 cm,
20 g, friction 0.8.

## IK and workspace

`kinematics.py`: damped least squares (lambda 0.05) on [position error; 0.15 x approach-direction error] using
`mj_jacSite`, variables pan/lift/elbow/wrist_flex, joint limits by clipping, multi-seed. The approach tilt tries
0 (straight down), then 0.15, 0.3, 0.45 rad (fingertips leaning away from the base) only when vertical is
unreachable.

Reachable region of the grasp point (base frame, **vertical** tool, cube-centre height z = 0.015), from the
IK reach map:

* Vertical tool at z = 0.015: x 0.08 .. 0.20 for |y| <= 0.24 (below x ~0.10 the middle |y| < 0.08 is blocked by
  the base), x = 0.22-0.24 for |y| <= 0.20-0.22, x = 0.26-0.28 for |y| <= 0.16, nothing beyond x ~0.30.
* Hover height: with a vertical tool the grasp point cannot go much above z ~0.10 m (wrist_flex limit); that is
  why the transit height is only z = 0.08 (cube bottom 6.5 cm above the table). At z = 0.08 the vertical-tool
  reach is x 0.14-0.26 for |y| <= 0.2. Tilted approaches extend reach (z = 0.015: x up to 0.28 at y = +-0.24).
* The nominal task (0.20, +-0.12) and +-3 cm jitter around it are comfortably inside.

## Evaluation (`eval.py`)

Randomisation per trial: block xy +-3 cm and uniform yaw; cube size +-20 %, mass +-20 %; cube friction 0.5-1.0,
pad friction 1.0-1.5, table friction +-20 %; per-actuator kp (and bias) +-20 %; 1-3 control-step (20-60 ms)
command latency; 3 mm Gaussian position noise and 0.05 rad yaw noise on what the planner sees.

Results: seed 0, 50 trials: **49/50 (98 %)**; seed 1, 100 trials: **97/100 (97 %)**. Failures are all large cubes
(~3.5 cm) that were pushed/not lifted; cube xy error on success is typically 3-7 mm.

## Caveats

* The sim servo is a stiff PD with torque limit; the real STS3215 with LeRobot P=16 is softer/slower, with deadband
  and backlash (the upstream `backlash` class is not enabled). Expect to slow the plan or tune gains.
* The home pose is a hover "ready" pose (pan 0, lift -22, elbow 25, wrist_flex 87, gripper open), not the folded
  rest pose of the real arm. Replace `HOME_XYZ` or the first/last waypoint if you want a stowed home.
* Cube is assumed to be on the table with known centre z; yaw only matters mod 90 deg.
* No camera/perception model; Kinect-to-base calibration error is represented only by the 3 mm noise.
