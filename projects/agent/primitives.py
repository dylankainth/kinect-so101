"""The arm's primitives, and the guard rails around skills that are written on the fly.

A *skill* is a short Python function written by the planner (Claude, through the MCP server) that calls a handful of
primitives on `arm`:

    def run(arm, x=0.28, y=0.0):
        arm.jaws(1.0)                       # open fully
        arm.move(x, y, 0.06)                # grasp point, metres, robot frame (x forward, y left, z up)
        arm.move(x, y, 0.012)
        arm.grasp()                         # close on something; the run stops if nothing is there
        arm.move(x, y, 0.08)

It runs in two stages. `Recorder` is the `arm` a skill sees while it is being *rehearsed*: it solves IK for every
waypoint and checks the workspace, reach, the table, joint steps and the number of steps, and records what the arm would
do. Nothing moves. Only a recording that passes can be *executed* (`execute`), which streams it through SafeArm the way
core.grasp.run_arm does (STOP/PARK/LIMP honoured, grasp check from the jaw reading, folds back to rest at the end).

The skill code is checked before it runs (no imports, no file or attribute tricks, no `while`), and the arm API is the
only way it can cause motion. That keeps mistakes out; it is not a security boundary against hostile code. The physical
limits (workspace box, table height, speed, reach) live in the Recorder and apply to every skill.
"""
import ast
import builtins
import hashlib
import inspect
import json
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from core import control
from core.motion import (GRIP_CLOSED_CMD, REAL_GRIPPER_OPEN, REST_POSE, STREAM_HZ, TIME_SCALE, hold_until_user, keyframes,
                         segmented_move, status, wait_to_cool)
from core.robot.arm import SafeArm
from core.sim.kinematics import JOINTS
from core.sim.pick_place import PLAN_DT, W_JOINT, _kin, _min_jerk, home_pose
from projects.agent.skills import SkillError   # a request that cannot be carried out: the text goes back to the planner

log = logging.getLogger("agent.primitives")

OPEN_SIM = 1.2                       # rad in the plan that the real arm's 70/100 opening corresponds to (see skills.py)
X_RANGE, Y_RANGE, Z_RANGE = (0.12, 0.36), (-0.25, 0.25), (0.006, 0.20)   # m: where the grasp point may go
MAX_TILT = 0.6                       # rad from vertical
MAX_SPEED = 0.15                     # m/s of the grasp point (the proven pick-and-place uses 0.05 down, 0.12 across)
MIN_SPEED = 0.01
MAX_STEPS = 200                      # primitive calls in one skill
MAX_SECONDS = 150                    # estimated duration of one skill on the real arm
MAX_WAIT_S = 5.0
IK_TOL = 2e-3                        # m
TILT_TOL = 0.05                      # rad
LINK_CLEARANCE = -0.01               # m: arm links (not the tool, which is limited by Z_RANGE) must stay this far above the table
GRASP_MIN = 12.0                     # jaw reading (0-100) above which something is between the jaws (as core.grasp)
GRIP_SQUEEZE = 8.0                   # commanded inside the contact reading while carrying (as core.grasp)
CLOSE_TIME_S = 1.5
MAX_FRAME_STEP_DEG = 3.0             # the most any joint may move between two 25 Hz frames (SafeArm.follow refuses more)

ARM_JOINTS = JOINTS[:5]
_LINKS = ("upper_arm", "lower_arm", "wrist")


# ---------------------------------------------------------------- the events a rehearsal records
@dataclass
class Move:
    q: np.ndarray            # Nx5 joint angles (sim rad), one row per PLAN_DT
    xyz: np.ndarray          # Nx3 grasp point along the move (m)
    jaws_open: bool          # what the picture shows (False once the jaws have closed on something)


@dataclass
class Jaws:
    opening: float           # 0 closed .. 1 fully open (what 70/100 is)
    seconds: float


@dataclass
class Grasp:
    expect: bool             # False: closing is not required to catch something


@dataclass
class Wait:
    seconds: float


@dataclass
class Rehearsal:
    events: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    steps: int = 0


def move_seconds(dist, speed, d_roll=0.0, d_tilt=0.0):
    return max(0.3, 1.875 * dist / speed, 1.875 * abs(d_roll) / W_JOINT, 1.875 * abs(d_tilt) / W_JOINT)


class Recorder:
    """The `arm` a skill sees. Every call solves its IK now and records the result; nothing moves."""

    def __init__(self):
        self.k = _kin()
        h = home_pose()
        self.q = np.array([h[n] for n in ARM_JOINTS])
        self.xyz = self.k.fk(self.q)[0]
        self.tilt = 0.0
        self.roll = 0.0
        self.opening = 1.0
        self.holding = False
        self.rec = Rehearsal()
        self.rec.events.append(Jaws(1.0, 0.0))

    # -- read-only facts a skill may use -----------------------------------------------------------------------------
    @property
    def pos(self):
        """Where the grasp point is now: (x, y, z) in metres."""
        return tuple(float(v) for v in self.xyz)

    def reach_ok(self, x, y, z, tilt=None):
        """True if the grasp point can be put at (x, y, z) (at this tilt, or at some tilt when none is given) and is inside
        the allowed workspace."""
        try:
            self._check_box(float(x), float(y), float(z), 0.0 if tilt is None else float(tilt))
            self._pick_tilt(np.array([x, y, z], float), self.roll) if tilt is None else self._solve(np.array([x, y, z], float), float(tilt), self.roll)
            return True
        except (SkillError, TypeError, ValueError):
            return False

    def log(self, text):
        """A note that comes back to the planner with the result."""
        if len(self.rec.notes) < 50:
            self.rec.notes.append(str(text)[:200])

    # -- motion --------------------------------------------------------------------------------------------------------
    def _count(self, seconds):
        self.rec.steps += 1
        self.rec.seconds += seconds
        if self.rec.steps > MAX_STEPS:
            raise SkillError(f"too many arm calls ({MAX_STEPS} at most)")
        if self.rec.seconds > MAX_SECONDS:
            raise SkillError(f"the skill would take over {MAX_SECONDS} s on the real arm")

    def _check_box(self, x, y, z, tilt):
        for name, v, (lo, hi) in (("x", x, X_RANGE), ("y", y, Y_RANGE), ("z", z, Z_RANGE)):
            if not math.isfinite(v) or not lo <= v <= hi:
                raise SkillError(f"{name}={v:.3f} m is outside the allowed range {lo}..{hi} m")
        if not math.isfinite(tilt) or abs(tilt) > MAX_TILT:
            raise SkillError(f"tilt {tilt:.2f} rad is outside +-{MAX_TILT}")

    def move(self, x, y, z, tilt=None, roll=None, yaw=None, speed=0.08, tip="grasp"):
        """Move the grasp point (midway between the fingertips) in a straight line to (x, y, z) metres.

        tilt: approach angle in rad from straight down (positive leans the fingertips away from the base); by default the
        current tilt is kept if it reaches, else the least tilted one that does. roll: wrist roll in rad. yaw: instead of roll, the direction (rad, robot frame, 0 = along +x) the
        jaws close along. speed: m/s, 0.01..0.15. The tilt and roll change smoothly along the move.
        tip: which point of the hand goes to (x, y, z): "grasp" (the point midway between the open pads, the default),
        or "fixed" / "moving" for the centre of that pad at the current jaw opening. One jaw is fixed and the other swings,
        so the pads are centimetres away from the grasp point and the offset turns with the arm: use tip="fixed" or
        "moving" whenever a pad (or the closed jaws) must touch something, as when pushing."""
        if tip not in ("grasp", "fixed", "moving"):
            raise SkillError('tip must be "grasp", "fixed" or "moving"')
        x, y, z = float(x), float(y), float(z)
        speed = float(speed)
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise SkillError(f"speed {speed} m/s is outside {MIN_SPEED}..{MAX_SPEED}")
        target = np.array([x, y, z])
        self._check_box(x, y, z, 0.0 if tilt is None else float(tilt))
        auto = tilt is None
        if yaw is not None:
            q4 = self._solve(target, self.tilt if auto else float(tilt), self.roll)
            roll = self.k.roll_for_yaw(q4, float(yaw), math.pi)
        roll = self.roll if roll is None else float(roll)
        tilt = self._pick_tilt(target, roll) if auto else float(tilt)
        if tip != "grasp":      # aim the grasp point so that the chosen pad ends on (x, y, z)
            want = target
            for _ in range(3):
                target = want - self._pad_offset(target, self._pick_tilt(target, roll) if auto else tilt, roll, tip)
            if auto:
                tilt = self._pick_tilt(target, roll)
            x, y, z = (float(v) for v in target)
        self._check_box(x, y, z, tilt)
        if not self.k.lo[4] <= roll <= self.k.hi[4]:
            raise SkillError(f"roll {roll:.2f} rad is outside {self.k.lo[4]:.2f}..{self.k.hi[4]:.2f}")
        dist = float(np.linalg.norm(target - self.xyz))
        T = move_seconds(dist, speed, roll - self.roll, tilt - self.tilt)
        self._count(T)
        n = max(2, int(math.ceil(T / PLAN_DT)))
        q, out, xyz_out = self.q.copy(), [], []
        for i in range(n):
            s = _min_jerk((i + 1) / n)
            p = self.xyz + (target - self.xyz) * s
            t = self.tilt + (tilt - self.tilt) * s
            r = self.roll + (roll - self.roll) * s
            q4, pe, te = self.k.ik(p, t, q0=np.append(q[:4], 0.0), roll=r)
            if pe > IK_TOL or te > TILT_TOL:
                raise SkillError(f"cannot reach ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) at tilt {t:.2f} "
                                 f"(IK error {pe * 1000:.1f} mm) on the way to ({x:.3f}, {y:.3f}, {z:.3f})")
            q = np.append(q4, r)
            if i % 4 == 0 or i == n - 1:
                low = self._link_height(q)
                if low < LINK_CLEARANCE:
                    raise SkillError(f"an arm link would hit the table near ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) at tilt {t:.2f}")
            out.append(q.copy())
            xyz_out.append(p)
        Q = np.array(out)
        step = float(np.degrees(np.abs(np.diff(Q, axis=0)).max())) * (1.0 / STREAM_HZ) / (PLAN_DT * TIME_SCALE) if n > 1 else 0.0
        if step > MAX_FRAME_STEP_DEG:
            raise SkillError(f"the move to ({x:.3f}, {y:.3f}, {z:.3f}) turns a joint {step:.1f} deg per frame (limit "
                             f"{MAX_FRAME_STEP_DEG}); lower the speed or split the move")
        self.q, self.xyz, self.tilt, self.roll = Q[-1][:5].copy(), target, tilt, roll
        self.rec.events.append(Move(Q, np.array(xyz_out), not self.holding))

    def _pad_offset(self, gp, tilt, roll, which):
        """Vector (m) from the grasp point to the centre of a pad when the grasp point is at gp, at the current jaws."""
        import mujoco
        k = self.k
        q4 = self._solve(gp, tilt, roll)
        k.d.qpos[:] = 0
        k.d.qpos[k.qadr[:5]] = np.append(q4, roll)
        k.d.qpos[k.qadr[5]] = self.opening * OPEN_SIM
        mujoco.mj_kinematics(k.m, k.d)
        pad = k.d.geom_xpos[k.m.geom("fixed_pad" if which == "fixed" else "moving_pad").id]
        return pad - k.d.site_xpos[k.site]

    def pad_offset(self, which="fixed"):
        """(dx, dy, dz) in metres from the grasp point to the centre of the "fixed" or "moving" pad at the current pose and
        jaw opening: where that pad is relative to arm.pos."""
        return tuple(float(v) for v in self._pad_offset(self.xyz, self.tilt, self.roll, which))

    def _pick_tilt(self, xyz, roll):
        """The approach tilt for a target when the skill gives none: keep the current one if it reaches, else the least
        tilted one that does (the arm cannot hold the tool vertical high up or far out)."""
        for t in (self.tilt, 0.0, 0.15, 0.3, 0.45, 0.6):
            q, pe, te = self.k.ik(xyz, t, q0=self.q, roll=roll)
            if pe < 1.5e-3 and te < 0.02:
                return t
        raise SkillError(f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) is out of the arm's reach at any tilt")

    def _solve(self, xyz, tilt, roll):
        q, pe, te = self.k.ik(xyz, tilt, q0=self.q, roll=roll)
        if pe > IK_TOL:
            raise SkillError(f"cannot reach ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) at tilt {tilt:.2f}")
        return q

    def _link_height(self, q5):
        import mujoco
        k = self.k
        k.d.qpos[:] = 0
        k.d.qpos[k.qadr[:5]] = q5
        mujoco.mj_kinematics(k.m, k.d)
        low = 9.0
        for i in range(k.m.ngeom):
            if k.m.body(k.m.geom_bodyid[i]).name in _LINKS:
                low = min(low, float(k.d.geom_xpos[i][2] - k.m.geom_rbound[i]))
        return low

    # -- the gripper -----------------------------------------------------------------------------------------------
    def jaws(self, opening, seconds=0.8):
        """Set the jaw opening: 0 = closed, 1 = fully open (about 8 cm between the pads). Does not check for contact;
        use grasp() to close on something."""
        opening, seconds = float(opening), float(seconds)
        if not 0.0 <= opening <= 1.0:
            raise SkillError("jaws opening must be between 0 and 1")
        if not 0.2 <= seconds <= 5.0:
            raise SkillError("jaws seconds must be between 0.2 and 5")
        self._count(seconds)
        self.opening, self.holding = opening, False
        self.rec.events.append(Jaws(opening, seconds))

    def grasp(self, expect=True):
        """Close the jaws slowly on whatever is between them and hold it. With expect=True (the default) the skill stops,
        the jaws reopen and the arm goes to rest if the jaws close on nothing."""
        self._count(CLOSE_TIME_S + 1.0)
        self.holding = True
        self.rec.events.append(Grasp(bool(expect)))

    def wait(self, seconds):
        """Stand still for a moment (at most 5 s)."""
        seconds = float(seconds)
        if not 0.0 <= seconds <= MAX_WAIT_S:
            raise SkillError(f"wait must be between 0 and {MAX_WAIT_S} s")
        self._count(seconds)
        self.rec.events.append(Wait(seconds))

    def finish(self):
        """Back to the home pose with the jaws open (the arm then folds to rest). Called for the skill."""
        h = home_pose()
        hq = np.array([h[n] for n in ARM_JOINTS])
        dist = float(np.abs(hq - self.q).max())
        T = max(0.5, 1.875 * dist / W_JOINT)
        n = max(2, int(math.ceil(T / PLAN_DT)))
        Q = np.array([self.q + (hq - self.q) * _min_jerk((i + 1) / n) for i in range(n)])
        self.rec.seconds += T
        if self.holding:
            self.rec.notes.append("the skill ended while still holding something: the jaws were opened to let it go")
        if self.opening < 1.0 or self.holding:
            self.rec.events.append(Jaws(1.0, 0.8))
        self.rec.events.append(Move(Q, np.array([self.k.fk(q)[0] for q in Q[:: max(1, n // 20)]]), True))
        return self.rec


# ---------------------------------------------------------------- the code of a skill
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min", "pow", "range", "round",
    "sorted", "str", "sum", "tuple", "zip", "reversed", "isinstance", "ValueError", "Exception")}
BANNED_NAMES = {"eval", "exec", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input",
                "breakpoint", "__import__", "type", "object", "super", "memoryview", "bytearray", "bytes", "help", "dir"}
MAX_CODE_CHARS = 8000
MAX_RANGE = 2000


def check_source(src):
    """Raises SkillError unless `src` is a module that only defines run(arm, ...) (and helper functions / constants) with
    plain control flow and arithmetic."""
    if len(src) > MAX_CODE_CHARS:
        raise SkillError(f"the skill is longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SkillError(f"syntax error on line {e.lineno}: {e.msg}") from None
    for node in ast.walk(tree):
        bad = None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad = "imports are not allowed (math is already available as `math`)"
        elif isinstance(node, (ast.While, ast.AsyncFor, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.Try,
                               ast.With, ast.ClassDef, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Delete)):
            bad = f"{type(node).__name__} is not allowed (use for-loops over range(), plain functions and if/else)"
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in ("format", "format_map")):
            bad = f"attribute {node.attr!r} is not allowed"
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            bad = f"name {node.id!r} is not allowed"
        if bad:
            raise SkillError(f"line {getattr(node, 'lineno', '?')}: {bad}")
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if len(runs) != 1:
        raise SkillError("define exactly one top-level function `def run(arm, ...)`")
    if not runs[0].args.args or runs[0].args.args[0].arg != "arm":
        raise SkillError("run must take `arm` as its first argument")
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.Assign, ast.Expr)):
            raise SkillError(f"line {n.lineno}: only functions, constants and a docstring go at the top level")
        if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant):
            raise SkillError(f"line {n.lineno}: no code may run at the top level")
    return tree


def _bounded_range(*a):
    r = range(*a)
    if len(r) > MAX_RANGE:
        raise SkillError(f"range() over more than {MAX_RANGE} items")
    return r


def compile_skill(src):
    """(run function, parameter names with defaults). The code is checked first."""
    tree = check_source(src)
    env = {"__builtins__": dict(SAFE_BUILTINS, range=_bounded_range), "math": math, "__name__": "skill"}
    exec(compile(tree, "<skill>", "exec"), env)   # noqa: S102  (checked above; it can only reach the arm API and math)
    run = env["run"]
    sig = inspect.signature(run)
    return run, {n: p.default for n, p in list(sig.parameters.items())[1:]}


def rehearse(src, args=None):
    """Run the skill against a Recorder. Returns the Rehearsal; raises SkillError with what to fix."""
    run, params = compile_skill(src)
    args = dict(args or {})
    unknown = set(args) - set(params)
    if unknown:
        raise SkillError(f"unknown arguments {sorted(unknown)}; the skill takes {sorted(params) or 'none'}")
    missing = [n for n, d in params.items() if d is inspect.Parameter.empty and n not in args]
    if missing:
        raise SkillError(f"missing arguments {missing}")
    arm = Recorder()
    try:
        run(arm, **args)
    except SkillError:
        raise
    except Exception as e:  # noqa: BLE001  a bug in the skill: tell the planner where
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        raise SkillError(f"the skill failed in line {tb.tb_lineno}: {type(e).__name__}: {e}") from None
    if not any(isinstance(e, Move) for e in arm.rec.events):
        raise SkillError("the skill never moved the arm")
    return arm.finish()


def source_hash(src, args):
    return hashlib.sha256((src + "\0" + repr(sorted((args or {}).items()))).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- what a rehearsal looks like
def describe(rec):
    moves = [e for e in rec.events if isinstance(e, Move)]
    pts = np.vstack([m.xyz for m in moves])
    lo, hi = pts.min(0), pts.max(0)
    grasps = sum(isinstance(e, Grasp) for e in rec.events)
    return (f"{len(rec.events)} actions, about {rec.seconds * TIME_SCALE:.0f} s on the real arm, {grasps} grasp(s). "
            f"The grasp point stays inside x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f}, z {lo[2]:.3f}..{hi[2]:.3f} m."
            + ("\nNotes: " + " | ".join(rec.notes) if rec.notes else ""))


def preview(rec, size=360):
    """Top view (x up, y left) and side view (distance from the base, height) of the recorded path; thick where the jaws
    are shut on something."""
    import cv2
    img = np.full((size, 2 * size, 3), 255, np.uint8)

    def px_top(p):
        return int(size / 2 - p[1] / 0.5 * size), int(size - (p[0] - 0.0) / 0.45 * size)

    def px_side(p):
        r = math.hypot(p[0], p[1])
        return int(size + 20 + r / 0.45 * (size - 40)), int(size - 20 - p[2] / 0.25 * (size - 40))

    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (200, 200, 200), 1)
    cv2.line(img, (size + 20, size - 20), (2 * size - 20, size - 20), (120, 120, 120), 1)
    cv2.putText(img, "top (x up, y left)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "side (reach, height)", (size + 26, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.circle(img, px_top((0.0, 0.0)), 6, (0, 0, 0), -1)
    n = 0
    prev = None
    for e in rec.events:
        if isinstance(e, Move):
            col = (0, 140, 0) if e.jaws_open else (0, 0, 220)
            for p in e.xyz:
                cur = (px_top(p), px_side(p))
                if prev is not None:
                    cv2.line(img, prev[0], cur[0], col, 1)
                    cv2.line(img, prev[1], cur[1], col, 1)
                prev = cur
            n += 1
            cv2.putText(img, str(n), px_top(e.xyz[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return img


# ---------------------------------------------------------------- doing it on the real arm
@dataclass
class Outcome:
    ok: bool
    notes: list


def real_frames(q5, grip_cmd):
    """25 Hz real-arm frames (slowed by TIME_SCALE) for a block of sim joint rows, with the jaw command held at grip_cmd."""
    plan = [dict(zip(JOINTS, list(q) + [0.0])) for q in q5]
    frames = keyframes(plan, OPEN_SIM)
    for f in frames:
        f["gripper"] = grip_cmd
    return frames


def save_trace(arm):
    """Commanded vs measured joints of the run, for working out why a skill did not do what the rehearsal showed."""
    if arm.trace:
        t0 = arm.trace[0][0]
        (control.DIR / "skill_trace.json").write_text(json.dumps([dict(t=t - t0, goal=g, pos=p) for t, g, p in arm.trace]))


def execute(rec, cool=wait_to_cool):
    """Stream a passed rehearsal through the real arm. Returns an Outcome; 'aborted' means the arm is held or parked."""
    notes, ok = [], True
    cool()
    with SafeArm(allow_writes=True) as arm:
        def stop_requested():
            control.beat()
            return control.is_set(control.STOP) or control.is_set(control.LIMP)
        arm.stop_requested = stop_requested
        arm.enable_torque()
        rest = dict(REST_POSE)
        try:
            h = home_pose()
            grip = REAL_GRIPPER_OPEN
            status("moving to the start pose")
            segmented_move(arm, real_frames([[h[n] for n in ARM_JOINTS]], grip)[0], dry_run=False)
            for e in rec.events:
                if isinstance(e, Move):
                    arm.follow(real_frames(e.q, grip), rate_hz=STREAM_HZ)
                elif isinstance(e, Jaws):
                    target = GRIP_CLOSED_CMD + e.opening * (REAL_GRIPPER_OPEN - GRIP_CLOSED_CMD)
                    n = max(1, int(e.seconds * STREAM_HZ))
                    pose = arm.read_positions()
                    if n > 1:
                        arm.follow([pose | {"gripper": grip + (target - grip) * _min_jerk((i + 1) / n)} for i in range(n)],
                                   rate_hz=STREAM_HZ)
                    grip = target
                elif isinstance(e, Grasp):
                    pose = arm.read_positions()
                    n = int(CLOSE_TIME_S * STREAM_HZ)
                    arm.follow([pose | {"gripper": grip + (GRIP_CLOSED_CMD - grip) * (i + 1) / n} for i in range(n)],
                               rate_hz=STREAM_HZ)
                    reading = arm.settle_gripper(GRIP_CLOSED_CMD)
                    if reading < GRASP_MIN and e.expect:
                        notes.append(f"grasp: the jaws closed to {reading:.0f}/100, on nothing: stopped, the skill did not finish")
                        arm.follow([pose | {"gripper": REAL_GRIPPER_OPEN}], rate_hz=STREAM_HZ)
                        ok = False
                        break
                    grip = max(GRIP_CLOSED_CMD, reading - GRIP_SQUEEZE) if e.expect else GRIP_CLOSED_CMD
                    notes.append(f"grasp: the jaws stopped at {reading:.0f}/100 ({'holding something' if reading >= GRASP_MIN else 'closed'})")
                elif isinstance(e, Wait):
                    time.sleep(e.seconds)
            status("folding back to rest")
            segmented_move(arm, rest, dry_run=False)
            time.sleep(0.5)
            arm.disable_torque()
            save_trace(arm)
        except BaseException as ex:  # noqa: BLE001  STOP, safety abort, anything
            log.error("skill aborted: %r", ex)
            try:
                arm.stop()
            except Exception:  # noqa: BLE001
                pass
            save_trace(arm)
            hold_until_user(arm, rest, str(ex) or type(ex).__name__)
            return Outcome(False, notes + [f"aborted: {ex or type(ex).__name__}. The arm is held or parked: do not continue."])
    return Outcome(ok, notes)
