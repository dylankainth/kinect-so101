"""SafeArm: guarded interface to the SO-101 follower (STS3215 x6) using LeRobot's calibration.

Safety model
------------
* Constructed with ``allow_writes=False`` (default) the object can ONLY read registers. Every
  register write goes through ``_write``/``_sync_write`` which raise unless ``allow_writes=True``.
  ``robot/check.py`` and all offline tests use the read-only mode.
* ``enable_torque()`` first writes Goal_Position := Present_Position (the motors' Goal_Position
  register currently reads 0, i.e. the arm would lunge toward tick 0 otherwise), then the
  conservative gains/accel/velocity, then Torque_Enable=1.
* ``move_to`` interpolates at a fixed rate with a speed cap, clamps to soft limits, refuses large
  jumps, aborts on tracking error / load / temperature / voltage.

Units (LeRobot convention, MotorNormMode)
-----------------------------------------
* shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll: DEGREES,
  deg = (ticks - mid) * 360 / 4095, mid = (range_min + range_max) / 2  (lerobot motors_bus._normalize).
* gripper: RANGE_0_100 (NOT degrees): 0 = fully closed, 100 = fully open,
  pct = (ticks - range_min) / (range_max - range_min) * 100.
All "ticks" here are in the frame of the *calibration* (see ``calibration_source``).
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

log = logging.getLogger("safearm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
RES = 4096
MAX_RES = RES - 1
DEFAULT_CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/DYLAN-FOLLOWER-1.json"

# Conservative settings written ONLY by enable_torque() (lerobot configure() uses P=16, I=0, D=32,
# Acceleration=254; we keep P/I/D and use a much gentler acceleration + a velocity cap).
GAINS = dict(P_Coefficient=16, I_Coefficient=0, D_Coefficient=32)
ACCELERATION = 30          # units of 100 ticks/s^2 (254 = lerobot's aggressive default)
GOAL_VELOCITY = 400        # ticks/s ~ 35 deg/s firmware cap (software cap is max_speed_deg_s)
# lerobot sets the gripper's overload threshold to 25%: any real grip trips it and 2 s later the servo
# drops to Protective_Torque (20%), so a grasped block slipped out on the lift. Use the arm joints'
# 80% threshold with a 50% fallback, and keep the squeeze proportional (run_pick_place.GRIP_SQUEEZE).
GRIPPER_REGS = dict(Max_Torque_Limit=800, Protection_Current=400, Overload_Torque=80,
                    Protective_Torque=50, P_Coefficient=32,
                    Goal_Velocity=2000, Acceleration=100)  # the arm's slow-speed cap made the jaws lag ~2 s

SOFT_MARGIN = {j: 3.0 for j in ARM_JOINTS} | {"gripper": 1.0}      # deg (gripper: 0-100 units)
LOAD_LIMIT = {j: 800 for j in ARM_JOINTS} | {"gripper": None}      # |Present_Load| 0..1000; gripper stalls on objects by design
TEMP_WARN_C, TEMP_ABORT_C = 50, 60
VOLT_ABORT_V = 3.4         # immediate stop below this
VOLT_LOW_V = 4.0           # ...or after VOLT_LOW_READS consecutive status reads below this
VOLT_LOW_READS = 3         # (the 5 V supply dips to ~3.7 V for moments when the arm is extended)
SAG_COMP_GAIN = 0.15       # per step (25 Hz): fraction of the tracking error added to the offset
SAG_COMP_MAX = 8.0         # deg, cap on the gravity-sag offset per joint
BUS_RETRIES = 3           # the bus occasionally drops a status packet; retry single-register access
TRACKING_ABORT = {j: 20.0 for j in ARM_JOINTS} | {"gripper": 1e9}  # gripper stalls on the block by design


class SafetyAbort(RuntimeError):
    pass


class SafeArm:
    def __init__(self, port="COM4", calibration_path=DEFAULT_CAL, allow_writes=False,
                 calibration_source="file", rate_hz=30.0, max_jump_deg=60.0):
        """calibration_source:
        'file'  - ranges/homing offsets from the LeRobot JSON; motor-reported ticks are shifted by
                  (motor Homing_Offset - file homing_offset) so results equal what lerobot would
                  report after writing the file to the motors. The motors' own Min/Max_Position_Limit
                  registers (hardware enforced) are intersected into the soft limits.
        'motor' - use the calibration actually stored in the motors.
        """
        self.port, self.allow_writes, self.rate_hz = port, allow_writes, rate_hz
        self.max_jump_deg, self.source = max_jump_deg, calibration_source
        self.cal = json.loads(Path(calibration_path).read_text())
        self.bus = None
        self.torque_on = False
        self.corr = {j: 0 for j in JOINTS}          # ticks added to motor-reported position
        self.lo, self.hi = {}, {}                   # soft limits in user units
        self.mismatch = {}
        self.stop_requested = None                  # optional callable -> True to abort move_to and hold
        self.low_volt_reads = 0
        self.trace = []                             # (t, goal, measured) per follow() step

    # ------------------------------------------------------------------ connection
    def connect(self):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        self.bus = FeetechMotorsBus(
            port=self.port,
            motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(JOINTS)})
        self.bus.connect(handshake=False)           # no register writes, no torque change
        found = self.bus.broadcast_ping()
        if not found or len(found) < 6:
            raise RuntimeError(f"expected 6 servos, ping found {found}")
        self._load_calibration()
        if any(self._read1("Torque_Enable", j) for j in JOINTS):
            self.torque_on = True
            log.warning("torque is ALREADY enabled on at least one joint")
        return self

    def _load_calibration(self):
        self.mot = {j: dict(off=self._read1("Homing_Offset", j),
                            lo=self._read1("Min_Position_Limit", j),
                            hi=self._read1("Max_Position_Limit", j)) for j in JOINTS}
        for j in JOINTS:
            f, m = self.cal[j], self.mot[j]
            if (f["homing_offset"], f["range_min"], f["range_max"]) != (m["off"], m["lo"], m["hi"]):
                self.mismatch[j] = dict(file=(f["homing_offset"], f["range_min"], f["range_max"]),
                                        motor=(m["off"], m["lo"], m["hi"]))
        if self.source == "file":
            self.rmin = {j: self.cal[j]["range_min"] for j in JOINTS}
            self.rmax = {j: self.cal[j]["range_max"] for j in JOINTS}
            self.corr = {j: self.mot[j]["off"] - self.cal[j]["homing_offset"] for j in JOINTS}
        else:
            self.rmin = {j: self.mot[j]["lo"] for j in JOINTS}
            self.rmax = {j: self.mot[j]["hi"] for j in JOINTS}
            self.corr = {j: 0 for j in JOINTS}
        for j in JOINTS:
            m = SOFT_MARGIN[j]
            if j == "wrist_roll":                   # full-turn joint: stay away from the +-180 wrap seam
                lo, hi = -170.0, 170.0
            else:
                lo = self.ticks_to_user(j, self.rmin[j]) + m
                hi = self.ticks_to_user(j, self.rmax[j]) - m
                # stay inside the hardware position-limit registers too (converted to calibration frame)
                hl = self.ticks_to_user(j, self.mot[j]["lo"] + self.corr[j]) + m
                hh = self.ticks_to_user(j, self.mot[j]["hi"] + self.corr[j]) - m
                lo, hi = max(lo, hl), min(hi, hh)
            self.lo[j], self.hi[j] = lo, hi

    def disconnect(self):
        if self.bus is not None:
            try:
                self.bus.disconnect(disable_torque=False)
            finally:
                self.bus = None

    # ------------------------------------------------------------------ context manager
    def __enter__(self):
        return self.connect()

    def __exit__(self, et, ev, tb):
        try:
            if self.torque_on and self.allow_writes:
                log.warning("exiting: disabling torque (arm goes limp - support it!)")
                self.disable_torque()
        finally:
            self.disconnect()
        return False   # never swallow exceptions / KeyboardInterrupt

    # ------------------------------------------------------------------ raw access
    def _unstick(self):
        """A transaction that died mid-way (USB glitch) leaves the SDK's busy flag set, after which every
        call fails with 'Port is in use!'. This code never talks to the bus from two threads, so the flag
        can safely be cleared before each call."""
        try:
            self.bus.port_handler.is_using = False
        except AttributeError:
            pass

    def _read1(self, reg, joint):
        self._unstick()
        return self.bus.read(reg, joint, normalize=False, num_retry=BUS_RETRIES)

    def _write(self, reg, joint, val):
        if not self.allow_writes:
            raise PermissionError(f"write {reg}={val} to {joint} blocked: SafeArm(allow_writes=False)")
        self._unstick()
        self.bus.write(reg, joint, int(val), normalize=False, num_retry=BUS_RETRIES)

    def _sync_write(self, reg, vals):
        if not self.allow_writes:
            raise PermissionError(f"sync_write {reg} blocked: SafeArm(allow_writes=False)")
        self._unstick()
        self.bus.sync_write(reg, {k: int(v) for k, v in vals.items()}, normalize=False)

    # ------------------------------------------------------------------ unit conversion
    def ticks_to_user(self, j, t):
        """Calibration-frame ticks -> degrees (arm joints) or 0-100 (gripper)."""
        lo, hi = self.rmin[j], self.rmax[j]
        if j == "gripper":
            return (min(hi, max(lo, t)) - lo) / (hi - lo) * 100.0
        return (t - (lo + hi) / 2) * 360.0 / MAX_RES

    def user_to_ticks(self, j, v):
        lo, hi = self.rmin[j], self.rmax[j]
        if j == "gripper":
            return lo + v / 100.0 * (hi - lo)
        return v * MAX_RES / 360.0 + (lo + hi) / 2

    def motor_to_frame(self, j, raw):
        t = raw + self.corr[j]
        return t % RES if j == "wrist_roll" else t

    def frame_to_motor(self, j, t):
        m = int(round(t)) - self.corr[j]
        return m % RES if j == "wrist_roll" else max(0, min(MAX_RES, m))

    # ------------------------------------------------------------------ reads
    def read_raw(self):
        self._unstick()
        return self.bus.sync_read("Present_Position", normalize=False, num_retry=BUS_RETRIES)

    def read_positions(self):
        """dict joint -> deg (pct for gripper), LeRobot-calibrated convention."""
        raw = self.read_raw()
        return {j: self.ticks_to_user(j, self.motor_to_frame(j, raw[j])) for j in JOINTS}

    def read_status(self, full=True):
        s = {"load": self.bus.sync_read("Present_Load", normalize=False, num_retry=BUS_RETRIES)}
        if full:
            s["temp"] = self.bus.sync_read("Present_Temperature", normalize=False, num_retry=BUS_RETRIES)
            s["volt"] = {j: v / 10 for j, v in self.bus.sync_read("Present_Voltage", normalize=False, num_retry=BUS_RETRIES).items()}
            s["torque"] = self.bus.sync_read("Torque_Enable", normalize=False, num_retry=BUS_RETRIES)
            s["current"] = self.bus.sync_read("Present_Current", normalize=False, num_retry=BUS_RETRIES)
        return s

    def _check_status(self, st):
        for j, t in st.get("temp", {}).items():
            if t >= TEMP_ABORT_C:
                self.disable_torque()
                raise SafetyAbort(f"{j} temperature {t}C >= {TEMP_ABORT_C}C: torque disabled (limp)")
            if t >= TEMP_WARN_C:
                log.warning("%s temperature %dC", j, t)
        if "volt" in st:
            vmin = min(st["volt"].values())
            self.low_volt_reads = self.low_volt_reads + 1 if vmin < VOLT_LOW_V else 0
            if vmin < VOLT_ABORT_V or self.low_volt_reads >= VOLT_LOW_READS:
                self.stop()
                raise SafetyAbort(f"supply voltage {vmin}V (low for {self.low_volt_reads} reads): holding")
            if vmin < VOLT_LOW_V:
                log.warning("supply dipped to %.1f V", vmin)
        for j, l in st["load"].items():
            lim = LOAD_LIMIT[j]
            if lim is not None and abs(l) > lim:
                self.stop()
                raise SafetyAbort(f"{j} load {l} > {lim} (of 1000): holding position")

    # ------------------------------------------------------------------ torque
    def enable_torque(self):
        """Hold-in-place torque enable. Writes goal=present FIRST, then gentle gains, then torque."""
        if not self.allow_writes:
            raise PermissionError("enable_torque blocked: SafeArm(allow_writes=False)")
        # STS servos switch torque on by themselves when they receive a Goal_Position, so the
        # dynamics are written first (definitely limp) and any failure part-way limps everything.
        try:
            for j in JOINTS:                        # 1. gentle dynamics (torque off)
                self._write("Torque_Enable", j, 0)
                self._write("Operating_Mode", j, 0)
                for r, v in GAINS.items():
                    self._write(r, j, v)
                self._write("Acceleration", j, ACCELERATION)
                self._write("Goal_Velocity", j, GOAL_VELOCITY)
                if j == "gripper":
                    for r, v in GRIPPER_REGS.items():
                        self._write(r, j, v)
            raw = self.read_raw()
            for j in JOINTS:                        # 2. goal := present (this engages torque per joint)
                self._write("Goal_Position", j, raw[j])
            for j in JOINTS:                        # 3. verify
                if abs(self._read1("Goal_Position", j) - raw[j]) > 2:
                    raise SafetyAbort(f"{j}: Goal_Position readback mismatch, not enabling torque")
            for j in JOINTS:                        # 4. torque on
                self._write("Torque_Enable", j, 1)
        except BaseException:
            log.error("enable_torque failed part-way: going limp")
            self.disable_torque()
            raise
        self.torque_on = True
        log.info("torque enabled (holding current pose)")

    def disable_torque(self):
        """Limp. Attempts all joints even if one write fails."""
        err = None
        for j in JOINTS:
            try:
                self._write("Torque_Enable", j, 0)
            except Exception as e:  # noqa: BLE001
                err = e
        self.torque_on = False
        if err:
            raise err

    def stop(self):
        """Hold current position with torque on (goal := present)."""
        if not self.allow_writes:
            return
        raw = self.read_raw()
        self._sync_write("Goal_Position", {j: raw[j] for j in JOINTS})

    # ------------------------------------------------------------------ motion
    def clamp(self, target):
        out = {}
        for j in JOINTS:
            v = float(target[j])
            if not math.isfinite(v):
                raise ValueError(f"{j}: non-finite target")
            c = min(self.hi[j], max(self.lo[j], v))
            if abs(c - v) > 1e-6:
                log.warning("%s target %.1f clamped to %.1f (soft limits %.1f..%.1f)",
                            j, v, c, self.lo[j], self.hi[j])
            out[j] = c
        return out

    def settle_gripper(self, target, timeout=2.5, still=0.4):
        """Command the gripper to `target` (0-100) and wait until the jaws stop moving (blocked by
        an object or at the target). Returns the position they settled at."""
        self._write("Goal_Position", "gripper",
                    self.frame_to_motor("gripper", self.user_to_ticks("gripper", target)))
        t0, hist = time.perf_counter(), []
        while time.perf_counter() - t0 < timeout:
            if self.stop_requested is not None and self.stop_requested():
                self.stop()
                raise SafetyAbort("STOP requested: holding position")
            hist.append(self.read_positions()["gripper"])
            if len(hist) >= 4 and max(hist[-4:]) - min(hist[-4:]) < still:
                break
            time.sleep(0.05)
        return hist[-1]

    def follow(self, trajectory, rate_hz, max_step_deg=3.0, max_start_gap_deg=10.0):
        """Stream an already-smooth trajectory (list of joint dicts) at a fixed rate, no per-point
        stopping. Same per-step checks as move_to (STOP flag, load/temp/voltage, tracking error).
        The first point must be close to the current pose; consecutive points must differ by at
        most max_step_deg (arm joints), so the caller controls speed via rate_hz."""
        if not (self.allow_writes and self.torque_on):
            raise PermissionError("follow needs allow_writes=True and enable_torque() first")
        traj = [self.clamp(p) for p in trajectory]
        cur = self.read_positions()
        # The arm sags a few degrees under gravity, so the start may be off by more than one step.
        start_gap = max(abs(traj[0][j] - cur[j]) for j in ARM_JOINTS)
        if start_gap > max_start_gap_deg:
            raise ValueError(f"trajectory starts {start_gap:.1f} deg from the arm: move there first")
        step = max(max(abs(b[j] - a[j]) for j in ARM_JOINTS) for a, b in zip(traj, traj[1:]))
        if step > max_step_deg:
            raise ValueError(f"trajectory step {step:.1f} deg > {max_step_deg}: resample or slow down")
        dt = 1.0 / rate_hz
        status_every = max(1, int(rate_hz // 2))
        t_next = time.perf_counter()
        bias = {j: 0.0 for j in ARM_JOINTS}  # learned sag compensation, deg
        pos = cur
        try:
            for k, goal in enumerate(traj):
                if self.stop_requested is not None and self.stop_requested():
                    self.stop()
                    raise SafetyAbort("STOP requested: holding position")
                # The low-gain servos settle short of the goal under gravity; integrate the error
                # into a bounded offset so the joint ends up where it was commanded.
                if k:
                    for j in ARM_JOINTS:
                        bias[j] = max(-SAG_COMP_MAX, min(SAG_COMP_MAX,
                                      bias[j] + SAG_COMP_GAIN * (goal_prev[j] - pos[j])))
                sent = dict(goal)
                for j in ARM_JOINTS:
                    sent[j] = min(self.hi[j], max(self.lo[j], goal[j] + bias[j]))
                goal_prev = goal
                self._sync_write("Goal_Position",
                                 {j: self.frame_to_motor(j, self.user_to_ticks(j, sent[j])) for j in JOINTS})
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
                pos = self.read_positions()
                self.trace.append((time.perf_counter(), goal, pos))
                self._check_status(self.read_status(full=(k % status_every == 0)))
                for j in JOINTS:
                    if abs(goal[j] - pos[j]) > TRACKING_ABORT[j]:
                        self.stop()
                        raise SafetyAbort(f"{j} tracking error {goal[j] - pos[j]:.1f} (stall/collision?): holding")
        except SafetyAbort:
            raise
        except BaseException:
            self.stop()
            raise

    def move_to(self, joint_targets, max_speed_deg_s=30.0, dry_run=False, allow_big_jump=False):
        """Move to targets (dict by joint name, or 6-sequence in JOINTS order; deg, gripper 0-100).

        Interpolates (smoothstep, peak speed == max_speed_deg_s) at self.rate_hz. Blocking.
        dry_run=True: reads the arm, logs the waypoints it WOULD send, sends nothing (works with
        allow_writes=False and torque off).
        """
        if not isinstance(joint_targets, dict):
            joint_targets = dict(zip(JOINTS, joint_targets))
        if set(joint_targets) != set(JOINTS):
            raise ValueError(f"need all joints {JOINTS}")
        if not 0 < max_speed_deg_s <= 60:
            raise ValueError("max_speed_deg_s must be in (0, 60]")
        if not dry_run and not (self.allow_writes and self.torque_on):
            raise PermissionError("move_to needs allow_writes=True and enable_torque() first (or dry_run=True)")
        cur = self.read_positions()
        tgt = self.clamp(joint_targets)
        delta = {j: tgt[j] - cur[j] for j in JOINTS}
        big = max(abs(d) for d in delta.values())
        if big > self.max_jump_deg and not allow_big_jump:
            raise ValueError(f"refusing jump of {big:.1f} > max_jump_deg={self.max_jump_deg}; "
                             "split it or pass allow_big_jump=True")
        dur = max(1.5 * big / max_speed_deg_s, 1.0 / self.rate_hz)
        n = max(1, int(math.ceil(dur * self.rate_hz)))
        log.info("%smove: max delta %.1f, %d steps over %.2fs", "[DRY] " if dry_run else "", big, n, dur)
        dt = 1.0 / self.rate_hz
        t_next = time.perf_counter()
        status_every = max(1, int(self.rate_hz // 2))
        try:
            for k in range(1, n + 1):
                s = k / n
                s = s * s * (3 - 2 * s)
                goal = {j: cur[j] + delta[j] * s for j in JOINTS}
                ticks = {j: self.frame_to_motor(j, self.user_to_ticks(j, goal[j])) for j in JOINTS}
                if dry_run:
                    if k == 1 or k == n or k % max(1, n // 5) == 0:
                        log.info("[DRY] step %d/%d goal=%s ticks=%s", k, n,
                                 {j: round(v, 1) for j, v in goal.items()}, ticks)
                    continue
                if self.stop_requested is not None and self.stop_requested():
                    self.stop()
                    raise SafetyAbort("STOP requested: holding position")
                self._sync_write("Goal_Position", ticks)
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
                pos = self.read_positions()
                self._check_status(self.read_status(full=(k % status_every == 0)))
                for j in JOINTS:
                    if abs(goal[j] - pos[j]) > TRACKING_ABORT[j]:
                        self.stop()
                        raise SafetyAbort(f"{j} tracking error {goal[j] - pos[j]:.1f} (stall/collision?): holding")
        except SafetyAbort:
            raise
        except BaseException:                      # includes KeyboardInterrupt
            self.stop()
            raise
        return tgt
