# Real-robot interface (SO-101 follower, COM4)

Files: `arm.py` (SafeArm), `mapping.py` (sim rad <-> real units), `check.py` (read-only diagnostic).
Nothing here has ever enabled torque or commanded the arm; the first power-on test is yours.

Units: arm joints are LeRobot calibrated DEGREES (zero = middle of calibrated range).
The gripper is a 0-100 RANGE value (0 closed, 100 open), not degrees.

## 0. Before you touch anything
1. `.venv\Scripts\python -m core.robot.check` - should show 6 servos, ~5 V, temps < 40 C, `Torque_Enable` all 0.
2. READ THE CALIBRATION WARNING it prints. The motors' stored homing offsets / limits differ from
   `DYLAN-FOLLOWER-1.json` (pan by 111 ticks, wrist_flex by 101, etc). SafeArm compensates by shifting
   readings into the file's frame (`calibration_source="file"`), and intersects the motors' own limit
   registers into the soft limits. Both interpretations agree within ~1 deg except shoulder_pan (~4 deg).
   Cleanest fix is to re-run `lerobot-calibrate` in person so file and motors agree, then rerun check.py
   (no warning expected).
3. Support the arm / clear the workspace. Motor supply is ~5.2 V (low for STS3215: weak torque, may sag
   under load; a 6-7.4 V supply is the normal rating).

## 1. First torque-on (you present, hand on the power switch)
Put the arm in a pose it can rest in (folded / resting on its stop). Then:

```python
import sys; sys.path.insert(0, "robot")
import logging; logging.basicConfig(level=logging.INFO)
from arm import SafeArm, JOINTS
with SafeArm(allow_writes=True) as arm:      # exit/exception/Ctrl+C => torque disabled (arm goes limp!)
    print(arm.read_positions())
    arm.enable_torque()    # goal:=present, gentle P/accel/velocity, then torque. Arm should NOT move.
    input("holding? ENTER to continue")
```
`enable_torque` writes Goal_Position = Present_Position first because the motors' goal register
currently reads 0 (a plain torque enable would swing every joint toward tick 0). If anything twitches,
cut the power switch.

## 2. Single-joint jogs (+-5 deg, slow)
For each joint in turn, with all others held:
```python
p = arm.read_positions()
for d in (+5, -5):
    t = dict(p); t["shoulder_pan"] = p["shoulder_pan"] + d
    arm.move_to(t, max_speed_deg_s=10, dry_run=True)   # inspect first
    arm.move_to(t, max_speed_deg_s=10)                 # then for real
```
Order: wrist_roll, wrist_flex, elbow_flex, shoulder_lift, shoulder_pan (loaded joints last), gripper
(+-5 units). Keep a hand near the power switch. `move_to` aborts (hold) on load > 80%, tracking
error > 20 deg, voltage < 4 V; limps on temperature >= 60 C.

## 3. Verify direction vs sim, per joint
Start the sim viewer with the same joint, give it a + step (`+0.1 rad`) and compare to the real
+5 deg jog. Positive real motion must look like positive sim motion (same rotation sense as seen from
the same viewpoint). Use `mapping.real_to_sim(arm.read_positions())` to feed real poses to the sim.
Also compare a static pose, e.g. arm pointing straight up / forearm horizontal: the sim and real
values should agree within a few degrees.

## 4. Fixing `mapping.py`
`real = sign * scale * degrees(sim) + offset` in `TABLE`:
* Joint moves the opposite way: set `sign` to `-1`.
* Same direction but constant error: set `offset` (real units) = real_reading - sign*sim_degrees at a
  pose you can set accurately in both.
* Gripper: `scale` = real units per sim degree (1.0 now; sim opens to 100 deg, real 0-100).
Check `python -m core.robot.mapping` (round trip) and re-verify with section 3. Flipping a sign does not
change which joint limits `SafeArm` enforces (those are in real units).

## 5. Full moves
Only after all six joints are verified: `move_to(real_targets, max_speed_deg_s=20)` in small steps
(default max jump 60 deg; larger needs `allow_big_jump=True`). Targets outside the soft limits are
clamped with a warning. `arm.stop()` holds position; `arm.disable_torque()` goes limp. Ending the
`with` block goes limp, so move to a rest pose first.
