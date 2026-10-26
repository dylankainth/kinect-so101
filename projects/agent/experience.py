"""What the arm has learned from its own attempts, kept in control/agent_experience.json between runs.

Every move is logged with what was tried and what the camera saw afterwards. Three things are learned from it:

* Placement bias (stacking): after a block lands on another, its offset from that one (along the stack's short and long
  axes, and in yaw) is measured. Each measurement is turned back into the raw bias with the correction that was applied
  taken out (error - offset used; the stack project fitted yaw error = 10.3 + 0.87 * trim deg, so a trim estimate is
  trim used - error / 0.87). The correction is the mean of the last WINDOW raw biases, pulled towards "no correction" by K
  pseudo-samples, so scatter averages out instead of being chased (a share-of-the-last-error update hunted: with 5 mm of
  scatter and almost no bias it walked the offset back and forth). A toppled block says nothing about the bias.
* Release strategy: how high above the block's centre the jaws let go (`place_raise`) and how far the open gripper slides
  clear before lifting (`release_clear`). A Beta bandit over a few values each: it keeps the best-looking value, and only
  tries others after a recent failure (exploring on every placement just adds scatter).
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
WINDOW, K = 12, 3                                    # placements averaged; pseudo-samples of 'no correction'
MAX_TRIM_DEG, MAX_OFFSET = 25.0, 0.012
LEARN_XY, LEARN_YAW = 0.015, 20.0                    # a placement worse than this (m, deg) is a failure, not bias


def placement_reward(es, el, ey_deg):
    """1 for a block exactly on the one below, falling linearly to 0 at 15 mm (or 20 deg) off."""
    return max(0.0, 1 - float(np.hypot(es, el)) / LEARN_XY) * max(0.0, 1 - abs(ey_deg) / LEARN_YAW)


class Experience:
    def __init__(self, path=PATH, seed=None):
        self.path = path
        self.rng = np.random.default_rng(seed)
        saved = json.loads(path.read_text()) if path.exists() else {}
        self.state = dict(trim_deg=START_TRIM_DEG, off_short=0.0, off_long=0.0,
                          raise_counts={}, clear_counts={}, samples=[]) | saved.get("state", {})
        self.events = saved.get("events", [])
        if "samples" not in saved.get("state", {}):   # an older file: rebuild the bias estimate from what was logged
            self.state["trim_deg"], self.state["off_short"], self.state["off_long"] = START_TRIM_DEG, 0.0, 0.0
            for e in self.events:
                if e["kind"] == "stack" and e["result"] in ("ok", "off") and "es_mm" in e:
                    p = e["params"]
                    self.learn_placement(e["es_mm"] / 1000, e["el_mm"] / 1000, e["ey_deg"],
                                         (p["trim_deg"], p["off_short"], p["off_long"]))
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

    def learn_placement(self, es, el, ey_deg, used):
        """es, el (m) and ey_deg: where the placed block ended up relative to the one under it. used: the (trim deg, offset
        short m, offset long m) that was applied for that placement. Returns True if it was used."""
        if abs(es) > LEARN_XY or abs(el) > LEARN_XY or abs(ey_deg) > LEARN_YAW:
            return False
        trim, off_s, off_l = used
        smp = self.state["samples"]
        smp.append([es - off_s, el - off_l, trim - ey_deg / TRIM_PER_DEG])
        del smp[:-WINDOW]
        m = np.mean(smp, axis=0)
        w = len(smp) / (len(smp) + K)
        s = self.state
        s["off_short"] = float(np.clip(-w * m[0], -MAX_OFFSET, MAX_OFFSET))
        s["off_long"] = float(np.clip(-w * m[1], -MAX_OFFSET, MAX_OFFSET))
        s["trim_deg"] = float(np.clip((1 - w) * START_TRIM_DEG + w * m[2],
                                      START_TRIM_DEG - MAX_TRIM_DEG, START_TRIM_DEG + MAX_TRIM_DEG))
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

    def recent_trouble(self):
        """The last stacking attempt did not come out ok (a success right after puts it back on the best release)."""
        ev = [e for e in self.events if e["kind"] == "stack"]
        return bool(ev) and ev[-1]["result"] != "ok"

    def choose_release(self, explore=None):
        """(place_raise, release_clear). explore None: sample only after a recent failure; True: sample; False: the best so far."""
        if explore is None:
            explore = self.recent_trouble()
        if explore:
            return (self._sample(self.state["raise_counts"], PLACE_RAISES),
                    self._sample(self.state["clear_counts"], RELEASE_CLEARS))
        best = []
        for key, values in (("raise_counts", PLACE_RAISES), ("clear_counts", RELEASE_CLEARS)):
            def mean(i):
                w, l = self.state[key].get(repr(values[i]), [2, 1] if i == 0 else [1, 1])
                return w / (w + l)
            best.append(values[max(range(len(values)), key=mean)])
        return tuple(best)

    def learn_release(self, place_raise, release_clear, reward):
        """reward in 0..1: 0 = the block fell off or toppled, 1 = it sits exactly on the one below (see placement_reward)."""
        reward = float(reward)
        for key, v, values in (("raise_counts", place_raise, PLACE_RAISES), ("clear_counts", release_clear, RELEASE_CLEARS)):
            c = self.state[key].setdefault(repr(v), [2, 1] if v == values[0] else [1, 1])
            c[0] += reward
            c[1] += 1.0 - reward

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
