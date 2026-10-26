"""What the arm has learned from its own attempts, kept in control/agent_experience.json between runs.

Every move is logged with what was tried and what the camera saw afterwards. Three things are learned from it:

* Placement bias (stacking): after a block lands on another, its offset from that one (along the stack's short and long
  axes, and in yaw) is measured. The next place subtracts a damped share of it: an aim offset in x/y and a place-yaw trim
  (the stack project fitted yaw error = 10.3 + 0.87 * trim deg, so the trim moves by error / 0.87). Only good-enough
  placements update it: a toppled block says nothing about the bias.
* Release strategy: how high above the block's centre the jaws let go (`place_raise`) and how far the open gripper slides
  clear before lifting (`release_clear`). A Beta bandit over a few values each; a placement that stays put is a win.
* Grasp offset (along / across the jaws, and height): `GraspLearner` (projects/pick_place/learner.py), rewarded 1 for a
  held block, 0.5 for a slip, 0 for a miss.
"""
import json
import time
from collections import Counter

import numpy as np

from core.paths import CONTROL_DIR
from projects.pick_place.learner import GraspLearner

PATH = CONTROL_DIR / "agent_experience.json"

PLACE_RAISES = (0.008, 0.004, 0.012, 0.0, 0.016)    # m above the block centre at release (stack project: 0.008)
RELEASE_CLEARS = (0.015, 0.02, 0.01, 0.025)         # m (stack project: 0.015)
START_TRIM_DEG = -11.8                               # stack project, fitted from 9 real trials
TRIM_PER_DEG = 0.87                                  # d(yaw error) / d(trim)
GAIN = 0.5                                           # share of a measured error that is corrected per placement
MAX_TRIM_DEG, MAX_OFFSET = 25.0, 0.012
LEARN_XY, LEARN_YAW = 0.015, 20.0                    # a placement worse than this (m, deg) is a failure, not bias


class Experience:
    def __init__(self, path=PATH, seed=None):
        self.path = path
        self.rng = np.random.default_rng(seed)
        saved = json.loads(path.read_text()) if path.exists() else {}
        self.state = dict(trim_deg=START_TRIM_DEG, off_short=0.0, off_long=0.0,
                          raise_counts={}, clear_counts={}) | saved.get("state", {})
        self.events = saved.get("events", [])
        self.grasp = GraspLearner("agent", seed=seed)
        self.grasp.path = path.with_name(path.stem + "_grasp.json")
        self.grasp.hist = json.loads(self.grasp.path.read_text()) if self.grasp.path.exists() else []

    def save(self):
        self.path.write_text(json.dumps(dict(state=self.state, events=self.events), indent=1))

    # -- placement bias ---------------------------------------------------------------------------------------------
    def correction(self):
        """(place-yaw trim in deg, aim offset along the stack's short axis in m, along its long axis in m)"""
        s = self.state
        return s["trim_deg"], s["off_short"], s["off_long"]

    def learn_placement(self, es, el, ey_deg):
        """es, el (m) and ey_deg: where the placed block ended up relative to the one under it, with the current
        correction already applied. Returns True if it was used."""
        if abs(es) > LEARN_XY or abs(el) > LEARN_XY or abs(ey_deg) > LEARN_YAW:
            return False
        s = self.state
        s["trim_deg"] = float(np.clip(s["trim_deg"] - GAIN * ey_deg / TRIM_PER_DEG,
                                      START_TRIM_DEG - MAX_TRIM_DEG, START_TRIM_DEG + MAX_TRIM_DEG))
        s["off_short"] = float(np.clip(s["off_short"] - GAIN * es, -MAX_OFFSET, MAX_OFFSET))
        s["off_long"] = float(np.clip(s["off_long"] - GAIN * el, -MAX_OFFSET, MAX_OFFSET))
        return True

    # -- release strategy -------------------------------------------------------------------------------------------
    def _sample(self, counts, values):
        best, pick = -1.0, values[0]
        for i, v in enumerate(values):
            w, l = counts.get(repr(v), [2, 1] if i == 0 else [1, 1])   # the stack project's value is the prior favourite
            draw = self.rng.beta(w, l)
            if draw > best:
                best, pick = draw, v
        return pick

    def choose_release(self, explore=True):
        """(place_raise, release_clear)"""
        if not explore:
            return PLACE_RAISES[0], RELEASE_CLEARS[0]
        return (self._sample(self.state["raise_counts"], PLACE_RAISES),
                self._sample(self.state["clear_counts"], RELEASE_CLEARS))

    def learn_release(self, place_raise, release_clear, stayed):
        for key, v, values in (("raise_counts", place_raise, PLACE_RAISES), ("clear_counts", release_clear, RELEASE_CLEARS)):
            c = self.state[key].setdefault(repr(v), [2, 1] if v == values[0] else [1, 1])
            c[0 if stayed else 1] += 1

    # -- the record -------------------------------------------------------------------------------------------------
    def log(self, kind, result, **info):
        self.events.append(dict(t=time.time(), kind=kind, result=result, **info))
        self.save()

    def summary(self):
        ev = [e for e in self.events if e["kind"] == "stack"]
        if not ev:
            return "No stacking experience yet."
        c = Counter(e["result"] for e in ev)
        trim, os_, ol = self.correction()
        last = [e["result"] for e in ev[-5:]]
        return (f"Stacking so far: {len(ev)} attempts, " + ", ".join(f"{n} {r}" for r, n in c.most_common())
                + f". Last five: {', '.join(last)}. Learned place-yaw trim {trim:+.1f} deg, aim offset "
                  f"{os_ * 1000:+.1f} / {ol * 1000:+.1f} mm.")
