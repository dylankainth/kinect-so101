# put_screws_away

Put the screws lying on the mat into the tray. Not yet run on the arm (the 5 V supply is being replaced).

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

Known limits (dry run on the 9 screws, 2026-10-20):

* At the lowest grasp height (6 mm) about a quarter of the screw directions are out of the arm's reach near the base
  (IK residual about 2 cm; yaw window about 105 deg of the 180), so 2 of the 9 screws were not plannable. The loop
  leaves those and says so.
* Nothing here has been tried with the real jaws: the grasp height, the squeeze and whether a 3 mm screw is held at
  all are open. The first real run should be one screw, watched, with `--attempts 1`.
