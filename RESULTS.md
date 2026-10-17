# Pick-and-place results (SO-101 + Kinect v1)

Scripted IK planner (MuJoCo SO-101 model), Kinect for perception and verification, no learned offsets.

    python run_pick_place.py --target 0.21 0.08 --target2 0.21 -0.10 \
        --practice 10 --keep-going --no-learn --attempts 3 --real --yes

Each cycle: find the block with the Kinect, plan and rehearse in sim, pick it up, carry it to whichever of
the two spots is farther from it, then check with the Kinect that it is inside a 4 cm circle at the target.
A cycle gets up to 3 attempts; a miss (jaws close to ~7/100 on nothing) re-detects the block and retries.

## Run of 10 cycles (2026-10-17)

| | |
|---|---|
| Cycles delivered | **10 / 10** |
| Attempts used | 13 |
| First attempt delivered | 7 / 10 cycles |
| Per-attempt success | 10 / 13 (77 %) |
| Every attempt where the jaws stopped open (>= 12/100) | delivered (10 / 10) |

Misses (all "jaws closed on nothing", all retried successfully): block yaw -23, -44, -19 degrees.
Hits at positive or near-zero yaw: 5 / 5 on the first attempt. Hypothesis (3 misses only, not established):
wrist roll is slightly off for negative block yaw.

An earlier 10-trial run, scored by the same method on its 8 valid trials (two carried the block under 4 cm
and were excluded), delivered 6 / 8 on a single attempt.

## Second run, 20 cycles (2026-10-17, evening)

20 cycles delivered, in three sessions because the 5 V supply sagged while torque was enabled
("Input voltage error") twice; `enable_torque` now waits and retries. The first 16 cycles took 18 attempts
(first attempt succeeded in 14 of 16). Of the 4 misses over the whole evening, the block yaws were +45, -20 and
one unlogged during a Kinect stall. The earlier negative-yaw hypothesis is not supported: 8 first attempts at
yaw -17 to -42 degrees all succeeded.

## What made it work

1. Depth-based kinematic calibration (`calib_joints.py`): fits joint zero offsets and the camera pose to the
   Kinect's 3D view of the real arm; redo whenever the arm's mount moves.
2. An honest grasp check: fast gripper, wait for the jaws to stop, read the position (about 7 = empty, 20+ =
   holding a block).
3. Grasping across the object's narrow width (yaw measured from the depth blob).
4. Gravity-sag compensation, voltage filter for the weak 5 V supply, gripper overload-protection settings.

## Known limits

* Camera search is limited to the mat (`roi` in `perception/config.json`); bright daylight changed colours
  enough to pick up the red/orange colour wheel off the mat.
* The Kinect needs restarting now and then; camera steps retry automatically.
* Reach: blocks beyond ~33 cm from the base are marginal.
