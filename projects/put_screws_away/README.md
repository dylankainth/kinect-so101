# put_screws_away

Put the screws lying on the mat into the tray. Run on the real arm with screws and bolts, see below.

```
.venv\Scripts\python -m projects.put_screws_away.main --look   # what does the camera see? -> control/screws.jpg
.venv\Scripts\python -m projects.put_screws_away.main          # plan every pick + the drop, arm not touched
.venv\Scripts\python -m projects.put_screws_away.main --real   # real arm (stop_gui.py open)
.venv\Scripts\python -m tests.test_screws
```

How it works:

* **Detection** (`detect.py`) is from the colour image: the screws are ~3 mm thick, far below the 1.5 cm depth threshold
  of `core/perception/multi_block.py`. A screw is a pixel run at least 25 V-levels brighter than the median of its
  surroundings (the mat's grid lines are fainter), 10-400 px, at least 3x longer than wide. Colour fringes on 3 mm
  features make saturation useless, so it is not used. Centre and direction come from the min-area rectangle; the
  pixel is intersected with the table plane through the camera calibration. On the first image it found 9 of 9
  screws (about 18 px = 2.4 cm long, 3 mm wide).
* **Grasp**: the jaws close across the screw's width (screw direction + 90 deg, period 180 deg), the grasp point
  4 mm above the screw's axis (`--grasp-raise`), the jaws squeezed only `--squeeze` units inside the contact point.
* **Drop**: over the tray's middle (`TRAY_XY`, 0.277 / 0.147, measured from the camera), released 6 cm up so it
  falls in.
* **Judging**: the jaw reading barely changes for a 3 mm screw (empty is about 7/100), so the arm always carries
  and the camera decides: a pick counts when that screw is gone from the mat and no new one appeared. A screw is tried
  `--attempts` times, most isolated first (the open jaws are wide).

## Real-arm results (2026-10-20, afternoon)

* **First try, screws (3 mm):** the jaws read 10/100 (empty is 7) but the run pushed the tray about 4 cm and scattered
  screws. Cause: the jaws opened to 70/100 inside a 9 cm-wide tray, and the release point was fixed, not measured.
* **Fixes:** the tray (position, rim height) is found from the camera on every run (`find_tray`, `TRAY_ROI`) and
  excluded from detection; the release is over the tray's near end (`DROP_RADIUS` 0.26 m from the base, the middle of the
  floor was out of reach), 4 cm above the rim (`RIM_CLEAR`); the jaws open to 30/100 only (`OPEN_REAL`).
* **Second try, screws:** the tray stayed put, but the grasp missed (7/100, empty).
* **Bolts** (about 6 mm shaft, bigger than the screws): 7 on the mat, touching in pairs. Detection: aspect >= 2.5,
  length 10-28 px (a longer blob is two bolts touching and is skipped). One run, `--attempts 2`: **3 bolts in the
  tray**, the jaw reading was 11-14/100 on a held bolt and 7/100 on a miss, so for bolts the jaw reading does tell a
  held one from an empty grasp. Misses and picks were mixed, about 4 attempts per 3 delivered.
* The 3 bolts left were beside the marker and the arm base (x about 0.16-0.24): the planner cannot reach those at the
  7 mm grasp height for most bolt directions.
* Supply: lowest servo voltage 3.8-4.8 V per pick (one dip to 3.8 V), no faults.

* **Afternoon, bolts and a screw:** all lying-flat bolts and the later screw went into the tray (the jaw reading 11-14/100
  on a held one, 7/100 on a miss). A bolt standing on its end is invisible to the detector (lay it flat).
* **Nuts** (`--kind nut`, about 1 cm across, compact blobs): 1 of 4 delivered. Every attempt on a nut closes to
  7/100 (empty) and pushes it a little; a nut drifted 3 cm over 8 attempts. `--search` retries a missed nut with the
  grasp shifted 10 mm along/across the jaws: one offset caught one nut once and then failed on the next, so it is
  not a calibration offset. The gripper's pads are probably too big for a 1 cm nut lying flat. Not reliable.
* **A mistake worth keeping:** the loop picked up a "screw" at the tray's near end (a bolt resting on its rim) and
  started to descend onto the rim; I killed the process while the arm moved, which leaves it powered and holding.
  Never kill a run mid-move: use the STOP flag, or wait for the status to say "done"/"parked". Now anything over the
  tray's footprint plus 4 cm (`TRAY_KEEP_OUT`) is ignored, and the tray's exclusion box has a 30 px margin.

Known limits (dry run on the 9 screws, 2026-10-20):

* At the lowest grasp height (6 mm) about a quarter of the screw directions are out of the arm's reach near the base
  (IK residual about 2 cm; yaw window about 105 deg of the 180), so 2 of the 9 screws were not plannable. The loop
  leaves those and says so.
* Nothing here has been tried with the real jaws: the grasp height, the squeeze and whether a 3 mm screw is held at
  all are open. The first real run should be one screw, watched, with `--attempts 1`.
