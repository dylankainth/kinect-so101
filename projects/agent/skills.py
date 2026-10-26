"""The arm's skills, as the planner sees them: a named action with a JSON schema that returns text (and a picture).

Every skill starts and ends with the arm folded at rest, checks its own outcome with the camera, and says plainly
what happened, so the planner can retry or change tack. The planner never sets joints: it picks skills and objects.
"""
import logging
import re
import time
from dataclasses import dataclass

import numpy as np

from core import control
from core.motion import aim, keyframes, wait_to_cool
from core.sim.pick_place import plan_pick_place
from projects.agent import world as W

log = logging.getLogger("agent.skills")

MAX_GRASP_WIDTH = 0.05      # m: the narrow side must fit between the open jaws (about 6 cm between the pads, less margin)
MAX_GRASP_HEIGHT = 0.15     # m: taller than this is not attempted
GRASP_FRACTION = 0.5        # grasp at this fraction of the object's height ...
GRASP_Z_MAX = 0.05          # ... but never higher than this (the arm cannot reach far above the table)
DROP_CLEAR = 0.03           # m above a container's rim that the carried object's bottom is released
SET_DOWN_TOL = 0.04         # m: set down within this of the target counts
MOVED_TOL = 0.03            # m: an object that moved at least this far from where it was has been taken
DROP_IN, SET_DOWN = "drop_in", "set_down"


@dataclass
class Result:
    ok: bool
    text: str
    image: object = None    # BGR array to show the planner, if any


class SkillError(Exception):
    """A request that cannot be carried out (out of reach, too big, unknown object): the text goes back to the planner."""


class SkillBox:
    def __init__(self, grounder, real=False, capture=W.capture, run_arm=None, confirm=None, cool=wait_to_cool):
        self.cool = cool
        self.grounder, self.real, self.capture = grounder, real, capture
        self._run_arm = run_arm
        self.confirm = confirm or (lambda what: True)
        self.scene = None
        self.objs = {}
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
        found = self.grounder.detect(self.scene.rgb, queries)
        pts = self.scene.points()
        self.objs, count = {}, {}
        for d in found:
            base = re.sub(r"\W+", "_", d["label"].lower()).strip("_") or "object"
            count[base] = count.get(base, 0) + 1
            o = W.measure(self.scene, d["label"], d["box"], f"{base}_{count[base]}", pts)
            if o is not None:
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
        frames = self.plan(o, dest, to_xy, how)
        what = f"move {o.id} to {to_object or tuple(to_xy)} ({how})"
        if not self.real:
            return Result(True, f"[dry run] planned: {what}; {len(frames)} frames, the arm was not moved.", self.view())
        if any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)):
            raise SkillError("a STOP/PARK/LIMP flag is set: the operator must clear it in stop_gui.py")
        if not self.confirm(what):
            raise SkillError("the operator declined this move")
        self.cool()
        ran = self.run_arm([(o.id, frames)])
        if "aborted" in ran or not ran:
            raise SkillError("the motion was stopped or faulted; the arm is held or parked. Do not continue.")
        return self.verify(o, dest, to_xy, how)

    def plan(self, o, dest, to_xy, how):
        narrow = 0.0 if o.flat else o.width
        if narrow > MAX_GRASP_WIDTH:
            raise SkillError(f"{o.id} is {narrow * 100:.1f} cm across its narrow side; the jaws open about "
                             f"{MAX_GRASP_WIDTH * 100:.0f} cm. It cannot be grasped.")
        if o.top_z > MAX_GRASP_HEIGHT:
            raise SkillError(f"{o.id} is {o.top_z * 100:.0f} cm tall; the limit is {MAX_GRASP_HEIGHT * 100:.0f} cm.")
        gz = float(np.clip(o.top_z * GRASP_FRACTION, 0.006, GRASP_Z_MAX))
        if dest is not None:
            txy, surface = dest.xy, dest.top_z
        else:
            txy, surface = np.asarray(to_xy, float), 0.0
        place_z = surface + gz + (DROP_CLEAR if how == DROP_IN else 0.0)
        elongated = not o.flat and o.length > 1.25 * max(o.width, 1e-3)
        close_yaw = (o.yaw + np.pi) % np.pi - np.pi / 2 if elongated else 0.0   # jaws close across the long side's normal
        pick = aim(np.array([*o.xy, gz]), 0.0)
        drop = aim(np.array([*txy, place_z]), 0.0)
        try:
            plan = plan_pick_place(pick, drop, block_yaw=close_yaw, yaw_period=np.pi, place_z=drop[2],
                                   held_width=None if o.flat or narrow < 0.02 else narrow)
        except ValueError as e:
            raise SkillError(f"not reachable by the arm: {e}")
        return keyframes(plan)

    def run_arm(self, batch):
        if self._run_arm:
            return self._run_arm(batch)
        from projects.put_screws_away.chain import run_chain
        return run_chain(batch)

    def verify(self, o, dest, to_xy, how):
        time.sleep(1.0)
        start = o.xy.copy()
        txy = dest.xy if dest is not None else np.asarray(to_xy, float)
        objs = self.observe([o.label] + ([dest.label] if dest is not None else []))
        same = [x for x in objs if x.label == o.label]
        near_start = [x for x in same if np.linalg.norm(x.xy - start) < MOVED_TOL]
        arrived = [x for x in same if np.linalg.norm(x.xy - txy) < SET_DOWN_TOL]
        if how == SET_DOWN and arrived:
            return Result(True, f"{o.label} is now at the target (within {SET_DOWN_TOL * 100:.0f} cm).", self.view())
        if how == DROP_IN and not near_start:
            return Result(True, f"{o.label} is no longer where it was; it should be in the container "
                                f"(it may be hidden from the camera).", self.view())
        seen = "\n".join(x.describe() for x in objs) or "nothing visible"
        if near_start:
            return Result(False, f"{o.label} is still where it started (missed grasp or it slipped). Ids were "
                                 f"reassigned; now seen:\n{seen}", self.view())
        return Result(False, f"{o.label} left its place but is not at the target. Ids were reassigned; now seen:\n{seen}",
                      self.view())


LOOK_SPEC = dict(
    description="Take a fresh camera picture and list the objects on the table with their positions and sizes. Pass "
                "`find` to look for specific things (e.g. ['glass', 'the red box']); omit it to list every loose object. "
                "Object ids are reassigned on every look.",
    input_schema=dict(type="object", properties=dict(find=dict(type="array", items=dict(type="string"))),
                      required=[]))

MOVE_SPEC = dict(
    description="Grasp one object with the gripper and move it. Destination is another object (to_object) or a "
                "table position (to_xy = [x, y] metres in the robot frame). how='set_down' places it on the "
                "destination (on the table at to_xy, or on top of to_object); how='drop_in' lets go over a "
                "container so the object falls in. The arm returns to rest afterwards and the camera checks the result. "
                "One object at a time; objects wider than 5 cm or taller than 15 cm cannot be grasped.",
    input_schema=dict(type="object", properties=dict(
        object_id=dict(type="string"),
        to_object=dict(type="string"),
        to_xy=dict(type="array", items=dict(type="number"), minItems=2, maxItems=2),
        how=dict(type="string", enum=[SET_DOWN, DROP_IN])),
        required=["object_id"]))
