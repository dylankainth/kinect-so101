"""The arm's skills, as the planner sees them: a named action with a JSON schema that returns text (and a picture).

Every skill starts and ends with the arm folded at rest, checks its own outcome with the camera, and says plainly
what happened, so the planner can retry or change tack. The planner never sets joints: it picks skills and objects.

move_object uses what the stack project (projects/stack_blocks) found on the real arm: the jaws open to 1.2 rad in the
plan (what 70/100 really is), the grasp point sits 8 mm above the block's centre, the held block is centred on the jaw
gap (held_width), a block going onto another is turned to match it (place_yaw, with a learned trim), the open gripper
slides clear before lifting, and the camera judges the result. What each attempt did is logged and learned from
(experience.py): the placement bias, the release strategy and the grasp offset.
"""
import logging
import re
import time
from dataclasses import dataclass

import numpy as np

from core import control, grasp
from core.motion import aim, fixed_jaw_dir, keyframes, wait_to_cool
from core.sim.pick_place import plan_pick_place
from projects.agent import world as W
from projects.agent.experience import Experience

log = logging.getLogger("agent.skills")

OPEN_SIM = 1.2              # rad, jaw opening in the plan: what the real arm's 70/100 opens to (about 8 cm between pads)
GRASP_RAISE = 0.008         # m above the object's centre for the grasp point
PLACE_HOVERS = (0.035, 0.025, 0.02, 0.015)   # approach heights above the place spot, the highest the arm can reach
MAX_GRASP_WIDTH = 0.06      # m: the short side of the top face must fit between the open jaws (the 5 cm boxes are gripped fine)
MAX_GRASP_HEIGHT = 0.15     # m: taller than this is not attempted
DROP_CLEAR = 0.03           # m above a container's rim that the carried object's bottom is released
SET_DOWN_TOL = 0.04         # m: set down within this of the target counts
MOVED_TOL = 0.03            # m: an object that moved at least this far from where it was has been taken
STACK_XY_OK = 0.015         # m: a block placed on another counts as on it within this
STACK_YAW_OK = 20.0         # deg
STACK_Z_BELOW, STACK_Z_ABOVE = 0.008, 0.010   # m: the new top may be this far below / above where a flat placement puts it
MEMORY_RADIUS = 0.03        # m: an object this close to a remembered spot is the same one
DROP_IN, SET_DOWN = "drop_in", "set_down"


@dataclass
class Result:
    ok: bool
    text: str
    image: object = None    # BGR array to show the planner, if any


class SkillError(Exception):
    """A request that cannot be carried out (out of reach, too big, unknown object): the text goes back to the planner."""


def close_yaw(o):
    """(direction the jaws close along, symmetry period of that direction) for an object: across the short side of an
    elongated top face, any face of a square one."""
    if o.flat:
        return 0.0, np.pi
    if o.length > 1.25 * max(o.width, 1e-3):
        return float((o.yaw + np.pi) % np.pi - np.pi / 2), np.pi
    return float((o.yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4), np.pi / 2


def placement_error(base, top):
    """Offset of `top` from `base`: (m along base's closing axis, m along its long axis, deg of yaw wrapped to +-period/2)."""
    cy, period = close_yaw(base)
    s = np.array([np.cos(cy), np.sin(cy)])
    e = top.xy - base.xy
    ty, _ = close_yaw(top)
    dyaw = (ty - cy + period / 2) % period - period / 2
    return float(e @ s), float(e @ np.array([-s[1], s[0]])), float(np.degrees(dyaw))


class SkillBox:
    def __init__(self, grounder, real=False, capture=W.capture, run_arm=None, confirm=None, cool=wait_to_cool, exp=None):
        self.cool = cool
        self.grounder, self.real, self.capture = grounder, real, capture
        self._run_arm = run_arm
        self.confirm = confirm or (lambda what: True)
        self.exp = exp or Experience()
        self.scene = None
        self.objs = {}
        self.support = []   # [(xy, height of what the object there rests on)]: stacked blocks the camera sees as one pile
        self.skills = {
            "look": (self.look, LOOK_SPEC),
            "move_object": (self.move_object, MOVE_SPEC),
        }

    # -- the planner's view ----------------------------------------------------------------------------------------
    def tool_specs(self):
        return [dict(name=n, **spec) for n, (_, spec) in self.skills.items()]

    def call(self, name, args):
        if name not in self.skills:
            return Result(False, f"no skill called {name!r}; available: {', '.join(self.skills)}")
        try:
            return self.skills[name][0](**args)
        except SkillError as e:
            return Result(False, str(e))
        except TypeError as e:
            return Result(False, f"bad arguments for {name}: {e}")

    # -- perception ------------------------------------------------------------------------------------------------
    def observe(self, queries=None):
        self.scene = self.capture()
        found = self.grounder.detect(self.scene.rgb, queries, scene=self.scene)
        pts = self.scene.points()
        self.objs, count = {}, {}
        for d in found:
            base = re.sub(r"\W+", "_", d["label"].lower()).strip("_") or "object"
            count[base] = count.get(base, 0) + 1
            o = W.measure(self.scene, d["label"], d["box"], f"{base}_{count[base]}", pts, d.get("mask"))
            if o is None:
                continue
            for xy, bz in self.support:
                if np.linalg.norm(o.xy - xy) < MEMORY_RADIUS and o.top_z > bz + 0.01:
                    o.base_z = bz
            self.objs[o.id] = o
        return list(self.objs.values())

    def view(self):
        return W.annotate(self.scene.rgb, list(self.objs.values()))

    def look(self, find=None):
        objs = self.observe(find)
        if not objs:
            return Result(True, "I see no objects" + (f" matching {find}." if find else "."), self.view())
        return Result(True, "Objects (robot frame: x forward, y left):\n" + "\n".join(o.describe() for o in objs),
                      self.view())

    def get(self, oid):
        if oid not in self.objs:
            raise SkillError(f"unknown object {oid!r}; known: {', '.join(self.objs) or 'none (call look first)'}")
        return self.objs[oid]

    def forget_support(self, xy):
        self.support = [(p, z) for p, z in self.support if np.linalg.norm(p - xy) >= MEMORY_RADIUS]

    # -- motion ----------------------------------------------------------------------------------------------------
    def move_object(self, object_id, to_object=None, to_xy=None, how=SET_DOWN):
        o = self.get(object_id)
        if how not in (SET_DOWN, DROP_IN):
            raise SkillError(f"how must be {SET_DOWN!r} or {DROP_IN!r}")
        if (to_object is None) == (to_xy is None):
            raise SkillError("give exactly one of to_object or to_xy")
        dest = self.get(to_object) if to_object else None
        if dest is not None and dest.id == o.id:
            raise SkillError("the destination is the object itself")
        frames, info = self.plan(o, dest, to_xy, how)
        what = f"move {o.id} to {to_object or tuple(to_xy)} ({how})"
        if not self.real:
            return Result(True, f"[dry run] planned: {what}; {len(frames)} frames, the arm was not moved.", self.view())
        if any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)):
            raise SkillError("a STOP/PARK/LIMP flag is set: the operator must clear it in stop_gui.py")
        if not self.confirm(what):
            raise SkillError("the operator declined this move")
        self.cool()
        result = self.run_arm(frames, info["pick_c"], grasp.GraspParams(tactile=info["tactile"]))
        kind = "stack" if info["stack"] else "move"
        base = dict(obj=o.id, pick=o.xy.tolist(), size=[o.length, o.width], thick=info["thick"], how=how,
                    dest=None if dest is None else dest.xy.tolist(), params=info["params"])
        if result == "aborted":
            self.exp.log(kind, "aborted", **base)
            raise SkillError("the motion was stopped or faulted; the arm is held or parked. Do not continue.")
        reward = {"done": 1.0, "slipped": 0.5, "missed": 0.0}[result]
        self.exp.grasp.record(o.xy, info["grasp_offset"], reward)
        if result != "done":
            self.exp.log(kind, result, **base)
            raise SkillError(f"the grasp {result} (the jaws closed on nothing or the object slipped out on the lift); "
                             f"nothing was moved. The grasp offset for this spot is being adjusted; trying again may work.")
        self.forget_support(o.xy)
        return self.verify(o, dest, to_xy, how, info, base)

    def plan_hovering(self, pick, place, o, yaw, period, place_yaw, held_width, clear):
        err = None
        for hover in PLACE_HOVERS:
            try:
                return plan_pick_place(pick, place, block_yaw=yaw, yaw_period=period, place_z=place[2],
                                       gripper_open=OPEN_SIM, held_width=held_width, place_hover=hover,
                                       place_yaw=place_yaw, release_clear=clear)
            except ValueError as e:
                err = e
        raise SkillError(f"not reachable by the arm: {err}")

    def plan(self, o, dest, to_xy, how):
        narrow = 0.0 if o.flat else o.width
        if narrow > MAX_GRASP_WIDTH:
            raise SkillError(f"{o.id} is {narrow * 100:.1f} cm across its short side; the jaws open about "
                             f"{MAX_GRASP_WIDTH * 100:.0f} cm. It cannot be grasped.")
        if o.top_z > MAX_GRASP_HEIGHT:
            raise SkillError(f"{o.id} is {o.top_z * 100:.0f} cm tall; the limit is {MAX_GRASP_HEIGHT * 100:.0f} cm.")
        thick = max(o.top_z - o.base_z, 0.006)
        trim, off_s, off_l = self.exp.correction()
        place_raise, clear = self.exp.choose_release()
        yaw, period = close_yaw(o)
        stack = dest is not None and how == SET_DOWN and not dest.flat
        txy, surface = (dest.xy, dest.top_z) if dest is not None else (np.asarray(to_xy, float), 0.0)
        place_yaw = None
        if stack:   # line the block up with the one under it; the trim and offset are what earlier placements taught
            dy, _ = close_yaw(dest)
            s = np.array([np.cos(dy), np.sin(dy)])
            txy = txy + off_s * s + off_l * np.array([-s[1], s[0]])
            place_yaw = dy + np.radians(trim)
        place_c = np.array([*txy, surface + thick / 2 + (DROP_CLEAR if how == DROP_IN else 0.0)])
        pick_c = np.array([*o.xy, o.top_z - thick / 2])
        pick_aim, place_aim = aim(pick_c, 0.0), aim(place_c, 0.0)
        du, da, dz = self.exp.grasp.best(o.xy)[0]
        pick_aim[2] += GRASP_RAISE + dz
        place_aim[2] += place_raise
        held = None if o.flat or narrow < 0.02 else narrow
        plan = self.plan_hovering(pick_aim, place_aim, o, yaw, period, place_yaw, held, clear)
        if du or da:   # a learned grasp offset in the gripper's own frame (along / across the jaws)
            u = fixed_jaw_dir(plan)
            shift = np.array([*(du * u + da * np.array([-u[1], u[0]])), 0.0])
            plan = self.plan_hovering(pick_aim + shift, place_aim, o, yaw, period, place_yaw, held, clear)
        info = dict(pick_c=pick_c, thick=thick, stack=stack, tactile=narrow >= 0.02, grasp_offset=[du, da, dz],
                    params=dict(trim_deg=trim, off_short=off_s, off_long=off_l, place_raise=place_raise, release_clear=clear))
        return keyframes(plan, OPEN_SIM), info

    def run_arm(self, frames, pick_c, params):
        if self._run_arm:
            return self._run_arm(frames, pick_c, params)
        return grasp.run_arm(frames, real=True, block_xyz=pick_c, params=params)

    # -- checking what happened ------------------------------------------------------------------------------------
    def verify(self, o, dest, to_xy, how, info, base):
        time.sleep(1.0)
        start = o.xy.copy()
        objs = self.observe([o.label] + ([dest.label] if dest is not None else []))
        seen = "\n".join(x.describe() for x in objs) or "nothing visible"
        p = info["params"]
        if info["stack"]:
            return self.verify_stack(o, dest, objs, info, base, seen)
        txy = dest.xy if dest is not None else np.asarray(to_xy, float)
        near_start = [x for x in objs if x.label == o.label and np.linalg.norm(x.xy - start) < MOVED_TOL]
        arrived = [x for x in objs if x.label == o.label and np.linalg.norm(x.xy - txy) < SET_DOWN_TOL]
        if how == SET_DOWN and arrived:
            x = min(arrived, key=lambda x: np.linalg.norm(x.xy - txy))
            self.exp.log("move", "ok", err_mm=float(np.linalg.norm(x.xy - txy) * 1000), **base)
            return Result(True, f"{o.label} is now {np.linalg.norm(x.xy - txy) * 100:.1f} cm from the target.", self.view())
        if how == DROP_IN and not near_start:
            self.exp.log("move", "ok", **base)
            return Result(True, f"{o.label} is no longer where it was; it should be in the container "
                                f"(it may be hidden from the camera).", self.view())
        self.exp.log("move", "not_there", **base)
        why = ("is still where it started" if near_start else "left its place but is not at the target")
        return Result(False, f"{o.label} {why}. Ids were reassigned; now seen:\n{seen}", self.view())

    def judge_stack(self, dest, objs, thick):
        """What the camera shows at the stack: dict(result = ok | off | toppled | not_on | lost, ...)."""
        expect = dest.top_z + thick
        top = [x for x in objs if np.linalg.norm(x.xy - dest.xy) < 0.04]
        if not top:
            return dict(result="lost")
        x = min(top, key=lambda x: np.linalg.norm(x.xy - dest.xy))
        dz = x.top_z - expect
        if dz > STACK_Z_ABOVE or dz < -STACK_Z_BELOW:
            return dict(result="toppled" if dz > 0 else "not_on", x=x, dz=dz)
        es, el, ey = placement_error(dest, x)
        good = abs(es) <= STACK_XY_OK and abs(el) <= STACK_XY_OK and abs(ey) <= STACK_YAW_OK
        return dict(result="ok" if good else "off", x=x, dz=dz, es=es, el=el, ey=ey)

    def verify_stack(self, o, dest, objs, info, base, seen):
        p = info["params"]
        j = self.judge_stack(dest, objs, info["thick"])
        if j["result"] in ("lost", "toppled", "not_on"):   # the Kinect sometimes drops out: look once more before believing it
            time.sleep(2.0)
            objs2 = self.observe([o.label, dest.label])
            j2 = self.judge_stack(dest, objs2, info["thick"])
            if j2["result"] in ("ok", "off"):
                j, objs, seen = j2, objs2, "\n".join(x.describe() for x in objs2)
        r = j["result"]
        if r == "lost":
            self.exp.log("stack", "lost", **base)
            return Result(False, f"the stack at {tuple(np.round(dest.xy, 3))} is not seen any more; now seen:\n{seen}", self.view())
        if r in ("toppled", "not_on"):
            self.exp.log("stack", r, dz_mm=j["dz"] * 1000, **base)
            self.exp.learn_release(p["place_raise"], p["release_clear"], False)
            return Result(False, (f"the block is not lying flat on the stack: its top is {j['dz'] * 100:+.1f} cm from where a flat "
                                  "placement puts it (tilted, standing on an edge, or off the stack)") if r == "toppled" else
                          f"the block is not on the stack: the pile's top is {-j['dz'] * 100:.1f} cm lower than expected",
                          self.view())
        es, el, ey, good = j["es"], j["el"], j["ey"], r == "ok"
        learned = self.exp.learn_placement(es, el, ey, (p["trim_deg"], p["off_short"], p["off_long"]))
        self.exp.learn_release(p["place_raise"], p["release_clear"], good)
        self.exp.log("stack", r, es_mm=es * 1000, el_mm=el * 1000, ey_deg=ey, dz_mm=j["dz"] * 1000, **base)
        if good:
            self.support.append((j["x"].xy.copy(), dest.top_z))
        t, s_, l_ = self.exp.correction()
        return Result(good, f"{o.label} is on the stack: {es * 1000:+.1f} mm across, {el * 1000:+.1f} mm along, "
                            f"{ey:+.1f} deg of yaw from the block under it"
                      + (f"; learned: yaw trim now {t:+.1f} deg, aim offset {s_ * 1000:+.1f}/{l_ * 1000:+.1f} mm" if learned else "")
                      + ("" if good else " (too far off to count)"), self.view())


LOOK_SPEC = dict(
    description="Take a fresh camera picture and list the objects on the table with their positions and sizes. Pass "
                "`find` to look for specific things (e.g. ['glass', 'the red box']); omit it to list every loose object. "
                "Object ids are reassigned on every look.",
    input_schema=dict(type="object", properties=dict(find=dict(type="array", items=dict(type="string"))),
                      required=[]))

MOVE_SPEC = dict(
    description="Grasp one object with the gripper and move it. Destination is another object (to_object) or a "
                "table position (to_xy = [x, y] metres in the robot frame). how='set_down' places it on the "
                "destination (on the table at to_xy, or stacked on top of to_object, lined up with it); how='drop_in' lets "
                "go over a container so the object falls in. The arm returns to rest afterwards and the camera checks the "
                "result, reporting the placement error for stacks. One object at a time; objects whose top face is wider "
                "than 6 cm or taller than 15 cm cannot be grasped.",
    input_schema=dict(type="object", properties=dict(
        object_id=dict(type="string"),
        to_object=dict(type="string"),
        to_xy=dict(type="array", items=dict(type="number"), minItems=2, maxItems=2),
        how=dict(type="string", enum=[SET_DOWN, DROP_IN])),
        required=["object_id"]))
