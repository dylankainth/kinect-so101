# stack_blocks

Stack three blocks on top of each other. Run from the repo root.

```
.venv\Scripts\python -m projects.stack_blocks.main --look     # 1. what does the Kinect see?
.venv\Scripts\python -m projects.stack_blocks.main            # 2. plan + rehearse every move in sim, no motion
.venv\Scripts\python -m projects.stack_blocks.main --real     # 3. real arm (keep core/tools/stop_gui open)
```

## How it works

1. **Look.** `core/perception/multi_block.py` finds blocks from depth alone, so colour does not matter.
   It lifts the depth image into the robot frame, groups everything more than 1.5 cm above the table, and
   reads each group's top face: x, y, yaw and height. Height / block size is the layer number.
2. **Decide.** The stack is whatever sits at the base spot (the block nearest the middle of the mat, unless
   you pass `--base X Y`). Every other block is loose. The loose block nearest the stack goes next.
3. **Plan and rehearse.** (Planning takes about 30 s per move on this machine.) `plan_pick_place(..., place_z=...)` puts the block down on top of the stack. The
   move is first run in MuJoCo with a fixed box standing in for the blocks already stacked; a failed
   rehearsal stops the run before the real arm moves.
4. **Move.** `core/grasp.run_arm` does the pick with the jaw-position grasp check, carries, places, folds to
   rest.
5. **Look again.** The next step starts from a fresh look, so a block that missed or was dropped is just
   loose again and gets retried (`--attempts` per block, default 3). The run ends when the stack is 3 high.

It stops instead of guessing when the table is not as expected: the camera does not account for exactly
`--count` blocks, two blocks touch (their top face is then two blocks long), or a neighbour is closer than
5.5 cm to the block being moved or to the stack (the open jaws are wide).

## Setting up the table

* Pick the block type with `--object`: `box8x5` (default, 8 x 5 x 3 cm boxes, gripped across the 5 cm side) or
  `cube3` (3 cm cubes). Edit `OBJECTS` in `main.py` for others.
* Blocks on the mat, at least 9 cm apart centre to centre for the boxes (5.5 cm for cubes). The detector only
  looks inside the mat window (`DEFAULT_ROI` and `WORKSPACE` in `multi_block.py`: x from 0.10 to 0.45 m, y from
  -0.30 m to about +0.09 m at block-top height): that keeps the parked arm out of view, and a block cut by the
  edge is reported as "not one block".
* Reach: with the boxes' yaw, picks fail beyond x of about 0.31 m and stacking three high needs x of about
  0.18-0.30 m. Put the blocks around x = 0.26 m, y = -0.15 / -0.05 / +0.05 (long sides along x), clear of the
  calibration marker and the parked gripper (x below about 0.20 near y = -0.08).

## Status (2026-10-20)

**Two-high stacks of flat 8 x 5 x 3 cm boxes work on the real arm, to about 2 mm and 3 deg.** Six trials in a row
(`python -m projects.stack_blocks.trials`: stack, measure the top block against the base block, unstack, repeat):
across the base +0.3 +- 1.2 mm, along it +1.0 +- 1.0 mm, yaw -0.7 +- 1.4 deg, none failed. Lay the boxes flat
(`--object box8x5`, the default), long sides away from the arm, at least 9 cm apart, at x about 0.25-0.30.
Standing them on their 8 x 3 edge does not work: a stack of those has about 7 mm of tolerance and tips.

What it took (in the order it was found, all on the real arm):

* **Camera calibration.** The Kinect had moved 83 mm sideways; every pick was aimed at the wrong place.
  `python -m core.calibration.calib_joints` re-fits it from the arm itself (clear the mat first; it hovers low).
  `core.calibration.handeye_auto` is not usable: from above it sees the gripper body, not the jaw tips (it fitted a
  90 mm z offset), so its `control/handeye.json` must not be left in place (`core.motion.aim` would apply it).
* **Wrist roll.** The place roll is only fixed modulo the block's symmetry; taking the one nearest the pick roll
  (not the smallest |roll|) stops a 160 deg turn that swung the off-centre held block 4-5 cm.
* **Release.** The open gripper slides 1.5 cm along the closing axis, away from the fixed pad, before lifting
  (`RELEASE_CLEAR`); lifting straight up dragged the block off the stack.
* **Held-block offset** (`held_width`): the grasp point is not the middle of the open jaws (moving jaw swings on an
  arc), so the pick and the place are shifted onto the jaw gap.
* **Place yaw trim** (`PLACE_YAW_TRIM`, -11.8 deg): fitted from 9 trials at three trims (yaw error = 10.3 + 0.87 *
  trim, 2.7 deg scatter).
* **Servo temperature abort** needs two hot reads in a row; a single corrupted packet read 62 C off a 31 C servo.

**The sim rehearsal does not predict the real arm for these boxes.** It fails every variant (the box is released
tilted on small pads and slides off), including moves the real arm does fine. Use `--skip-rehearsal` (or
`--tune 0` for none at all, which the trial loop does). The sim is still right about reach. `--show-sim` plays each
rehearsal in the MuJoCo viewer, `--tune N` tries up to N release variants (`TUNINGS`), and
`python -m core.tools.kinect_live` shows the live camera with the planner's pick and place targets.

**Three-high is not reliable yet** (2026-10-20, 4 trials built, `trials --count 3`; the rest were cut short by supply
faults). Layer 2 is as good as in the two-high trials (within 3 mm and 3 deg, picked from either side). Layer 3 was
6-12 mm and up to 8 deg off, always picked from the far-left block (y about 0.15, pan about -45 deg), so the cause
(far pick, higher tilt, or stack height) is open. `--refine N` (look, pick the block up again, re-place it aimed at
target minus the measured error) did NOT help: 10.0 mm became 10.4 mm on the other side and the yaw error stayed,
so the error of a re-picked block is not a fixed bias. It is off by default. Do not run with `--min-gap` below the
default: at 7 cm the arm bumped the neighbouring block. `trials` also writes `*_moves.csv` (pick position, pan, roll
and error per move) for fitting the error against the geometry, and `SafeArm` now appends the lowest servo voltage
to `control/voltage_log.csv`: the 5 V supply sagged to 3.3 V and 2.9 V mid-move and aborted two runs.
