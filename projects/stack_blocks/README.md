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

## Status (2026-10-19)

Working, verified on the real camera: `--look` finds the three boxes (8.0-8.3 x 4.9-5.0 cm, level 1, jaw-closing
direction from the short side). Verified without the arm: depth detection on rendered scenes (1-2 mm), the
real-run control flow against fakes (`tests/`), and the planner changes (`place_z`, `gripper_open`,
`held_width`), with the default pick-and-place unchanged.

**Blocked: the sim rehearsal of the 8 x 5 x 3 cm boxes fails, so nothing was run on the real arm.** The cause is
the gripper, not the planning:

* The SO-101 moving jaw swings on an arc: open, its pad is about 5 cm above the box centre, and it comes down as it
  closes. Across a 5 cm side it reaches the box face above the top of a 3 cm tall box, so it lands on the top edge
  (pad centre inside the box footprint at the grasp). The box is lifted by friction and slides off when the jaws
  open. A 3 cm cube is narrow enough that the arc has come down in time.
* The grasp point is also not the middle of the open jaws (gap centre about 0.9 cm off, with the fixed pad at
  +2.35 cm and the moving pad at -4.11 cm at 0.8 rad). `plan_pick_place(..., held_width=...)` corrects for this
  (verified: with it the box is lifted, carried and put on the stack at the right place).
* Tilting the gripper (0.3 / 0.45 rad) does not change the result.

Ways forward: 3 cm cubes (`--object cube3`) are within what the gripper has done before; taller or wider blocks
(the arc then meets the side lower), a taller moving-jaw pad, or a different grip would be needed for the boxes.
The rehearsal gate stays on: a failed rehearsal stops the run before the real arm moves.
