"""Learns a grasp correction from real attempts and proposes the next one.

Each attempt aims at the detected object plus an offset in the gripper's frame:
    du  along the closing axis (toward the fixed jaw), m
    da  across the closing axis (horizontal), m
    dz  grasp height relative to the profile's default, m
After the attempt we record the outcome: 1.0 grasped and held, 0.5 grasped but slipped, 0.0 missed.

The next offset comes from Thompson sampling over a grid of candidate offsets. Every past attempt
counts toward nearby candidates (Gaussian weight in offset space) and more strongly when it was
made near the same spot on the table, so the arm keeps what worked where it worked and moves
away from offsets that missed. Experience persists per object in control/learn_<object>.json.
"""
import json
import time
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent / "control"

GRID_UA = np.linspace(-0.015, 0.015, 7)        # 5 mm steps along / across the closing axis
GRID_Z = np.array([-0.004, 0.0, 0.004, 0.008])  # grasp height steps
SIGMA_OFF = np.array([0.006, 0.006, 0.003])   # how far one attempt's evidence reaches (du, da, dz)
SIGMA_LOC = 0.08                              # m on the table
PRIOR = 1.0                                   # pseudo-counts: success near 0 offset, failure elsewhere


class GraspLearner:
    def __init__(self, obj, seed=None):
        self.obj = obj or "default"
        self.path = DIR / f"learn_{self.obj}.json"
        self.rng = np.random.default_rng(seed)
        self.hist = json.loads(self.path.read_text()) if self.path.exists() else []
        g = np.array(np.meshgrid(GRID_UA, GRID_UA, GRID_Z, indexing="ij")).reshape(3, -1).T
        self.cands = g

    def _posterior(self, xy):
        # prior: the calibrated aim (0 offset) is the best guess
        d0 = np.sum((self.cands / SIGMA_OFF / 1.5) ** 2, axis=1)
        a = 0.3 + PRIOR * np.exp(-d0 / 2)
        b = 0.3 + PRIOR * (1 - np.exp(-d0 / 2))
        for h in self.hist:
            off = np.array(h["offset"])
            w = np.exp(-np.sum(((self.cands - off) / SIGMA_OFF) ** 2, axis=1) / 2)
            w *= np.exp(-np.sum((np.array(h["xy"]) - xy) ** 2) / (2 * SIGMA_LOC ** 2))
            a += w * h["reward"]
            b += w * (1 - h["reward"])
        return a, b

    def propose(self, xy, explore=True):
        """Offset (du, da, dz) to try for an object at table position xy."""
        a, b = self._posterior(np.asarray(xy, float))
        score = self.rng.beta(a, b) if explore else a / (a + b)
        return self.cands[int(np.argmax(score))].copy()

    def best(self, xy):
        a, b = self._posterior(np.asarray(xy, float))
        i = int(np.argmax(a / (a + b)))
        return self.cands[i].copy(), float(a[i] / (a[i] + b[i]))

    def record(self, xy, offset, reward, **info):
        self.hist.append(dict(t=time.time(), xy=[float(v) for v in xy], offset=[float(v) for v in offset],
                              reward=float(reward), **info))
        self.path.write_text(json.dumps(self.hist, indent=1))

    def summary(self, xy):
        off, p = self.best(xy)
        n = len(self.hist)
        wins = sum(h["reward"] >= 1 for h in self.hist)
        return (f"{n} attempts on record ({wins} held); best offset here: along {off[0] * 100:+.1f} cm, "
                f"across {off[1] * 100:+.1f} cm, height {off[2] * 100:+.1f} cm (p~{p:.2f})")
