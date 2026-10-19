# core/perception/ - Kinect v1 block perception for the SO-101

Run everything from the repo root with `.venv\Scripts\python -m core.perception.<module>`.
(The `[Stream 70] Invalid magic` stderr spam is a flaky USB link; harmless.)

| file | role |
|---|---|
| `kinect.py` | `Kinect` class (background runloop thread), `depth_to_meters`, `deproject` |
| `block_detect.py` | HSV blob detector + `--tune` trackbar tool |
| `calibrate.py` | camera -> robot transform (ArUco+depth, ArUco PnP, clicked points), `verify` |
| `block_pose.py` | `get_block_xyz_robot()` -> `BlockPose(valid, xyz, ...)` in robot frame |
| `success.py` | `Region.rect/circle`, `check_success(region)` |
| `config.json` | marker pose, block HSV, block size (see below) |
| `calib.json` | output of calibration: `T_robot_cam` 4x4 (+ method, fit RMS) |

Frames: robot = x forward, y left, z up, origin at the arm base on the table.
Camera = optical (x right, y down, z forward). Depth is `DEPTH_REGISTERED` (uint16 mm,
aligned to RGB, 0 = invalid), RGB intrinsics fx=529.2 fy=525.6 cx=328.9 cy=267.5.

## Calibration procedure (do this in person)

Marker: `markers/aruco_4x4_50_id0_80mm.png` (DICT_4X4_50, id 0, black square 80 mm).

1. Print at 100% / "actual size" (no fit-to-page). **Measure the black square with a
   ruler**; if it is not 80.0 mm, set `marker.size_mm` in `config.json` to the measured value.
   Glue it flat on stiff card; it must be flat and matte (no glare).
2. Decide the robot frame: origin = point on the table directly below the arm's base
   axis (shoulder-pan axis); +x = the direction the arm faces at pan=0 (forward), +y = left
   of that when looking along +x from above, +z = up.
3. Lay the marker flat on the table within the camera view, **not under the arm's
   sweep**, with its top edge (as printed) pointing along a robot axis. Pick yaw:
   - top edge points robot +y (left): `yaw_deg = 0`
   - top edge points robot +x (forward): `yaw_deg = -90`
   - top edge points robot -y (right): `yaw_deg = 180`
   - top edge points robot -x (back): `yaw_deg = 90`
   (Equivalently the print's right-hand direction is robot (cos yaw, sin yaw).) Small misalignment is the main error source; use
   a square/ruler to align the marker edge with the axis, or measure and enter the real yaw.
4. Measure the **centre of the marker** (diagonal intersection) in robot coordinates with a
   tape measure: forward distance x and left distance y from the base axis, in metres.
   z = 0 if the table is the robot base plane (marker thickness ~0 ok); if the base sits on a
   plate/raised, use the marker's height above the base plane.
5. Edit `config.json`:
   ```json
   "marker": {"id": 0, "size_mm": 80.0, "center_xyz_m": [0.20, -0.10, 0.0], "yaw_deg": -90}
   ```
6. Make sure nothing else holds the Kinect (close any viewer), then
   `python -m core.perception.calibrate aruco`. Check the printed fit RMS (expect < ~5 mm) and
   "camera origin" (z ~ camera height above table, ~0.8-1.2 m). Writes `calib.json`.
   Fallbacks: `calibrate pnp` (RGB only; noisy, expect cm-level / a few degrees tilt error,
   use only if depth fails) or `calibrate points --points X1,Y1 X2,Y2 X3,Y3 X4,Y4` (click table
   points with known robot-frame coordinates, in order; Enter to finish).
7. Verify: `python -m core.perception.calibrate verify`, hover over the base centre / a point
   measured by tape; the readout must match within ~1 cm and z ~ 0 on the table. Repeat step 6 if
   not. **Re-run whenever the Kinect or the arm base moves.**
8. Remove or leave the marker (keep it out of the block/target areas).

## Block detection

`python -m core.perception.block_detect --tune` : trackbars for HSV (hue wraps when h_lo > h_hi,
default red/orange 170..12), click image to read HSV, `s` saves to `config.json["block"]`,
`q` quits. Use your real block under real lighting. `python -m core.perception.block_detect` runs
10 live detections. Detection is rejected if no blob of plausible size/fill, or depth invalid
(<30% valid pixels in blob) or outside 0.4-3 m. Optional `roi` [x0,y0,x1,y1] limits the search
(good for excluding red things elsewhere). Table/robot colours that match the hue will confuse it.

## Usage in code

```python
from core.perception.block_pose import get_block_xyz_robot
from core.perception.success import Region, check_success
r = get_block_xyz_robot()          # median over ~10 frames; r.valid False if unseen/moving
ok, info = check_success(Region.rect(0.15, 0.30, -0.20, -0.05))   # in region AND on table
```
`xyz` is the cube *centre* (top-face measurement minus `block_pose.block_size_m`/2, default
0.03 m); success requires centre z within 1.5 cm of block_size/2. Tunables in `config.json`
under `block_pose`: `block_size_m`, `n_frames`, `min_detections`, `max_spread_m`.
An undetected block (e.g. still in gripper/occluded) is never a success.
