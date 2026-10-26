"""Open-vocabulary grounding: a text description -> pixel boxes in the colour image.

The Grounder interface is one method, so a local detector (Grounding DINO + SAM, OWL-ViT) can replace the language-model
vision backend without touching the skills or the planner.
"""
import json
import re

import cv2
import numpy as np

from core.perception.multi_block import DEFAULT_ROI, MAX_HEIGHT, MIN_HEIGHT, WORKSPACE
from projects.agent.llm import DEFAULT_MODEL, data_url, make_client

PROMPT = """This is a {w}x{h} pixel colour image from a camera looking down at a robot's workspace (a table, the robot arm at the top).
{ask}
Reply with only a JSON list: [{{"label": "<short name>", "box": [left, top, right, bottom]}}, ...] in image pixel
coordinates, tight boxes around each whole object. An empty list if there are none. Do not box the robot arm, the
table or the mat."""


class Grounder:
    def detect(self, rgb, queries=None, scene=None):
        """-> [{"label": str, "box": (u0, v0, u1, v1), optional "mask": HxW bool}]. queries: descriptions to find, or None
        for every loose object. scene: the depth and calibration for backends that use them."""
        raise NotImplementedError


class VisionGrounder(Grounder):
    """Asks a vision-language model (through OpenRouter) for boxes."""
    def __init__(self, model=DEFAULT_MODEL, client=None):
        self.client, self.model = client or make_client(), model

    def detect(self, rgb, queries=None, scene=None):
        h, w = rgb.shape[:2]
        ask = ("Find: " + "; ".join(queries) + ". Include every match." if queries
               else "List every loose object on the table that the robot could pick up.")
        msg = self.client.chat.completions.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))}},
                {"type": "text", "text": PROMPT.format(w=w, h=h, ask=ask)}]}])
        return parse_boxes(msg.choices[0].message.content or "", w, h)


class DepthGrounder(Grounder):
    """Every loose object standing above the table inside the mat window, from depth alone (no network, no model): the
    same rule as core/perception/multi_block.py. All are labelled "block"; anything wider than MAX_EXTENT (a tray, a hand,
    the arm) is skipped. The query is ignored."""
    MAX_EXTENT = 0.12
    MIN_PIXELS = 60

    def detect(self, rgb, queries=None, scene=None):
        P = scene.points()
        u0, v0, u1, v1 = DEFAULT_ROI
        sub = P[v0:v1, u0:u1]
        x0, x1, y0, y1 = WORKSPACE
        z = sub[..., 2]
        m = (np.isfinite(sub).all(-1) & (z > MIN_HEIGHT) & (z < MAX_HEIGHT) & (sub[..., 0] > x0) & (sub[..., 0] < x1)
             & (sub[..., 1] > y0) & (sub[..., 1] < y1))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < self.MIN_PIXELS:
                continue
            xy = sub[lab == i][:, :2]
            if np.ptp(xy, axis=0).max() > self.MAX_EXTENT:
                continue
            full = np.zeros(P.shape[:2], bool)
            full[v0:v1, u0:u1] = lab == i
            l, t, w, h = (st[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            out.append(dict(label="block", box=(u0 + l, v0 + t, u0 + l + w, v0 + t + h), mask=full))
        return sorted(out, key=lambda d: (d["box"][0], d["box"][1]))


def parse_boxes(text, w, h):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for r in raw:
        try:
            u0, v0, u1, v1 = (float(x) for x in r["box"])
            label = str(r["label"]).strip() or "object"
        except (KeyError, TypeError, ValueError):
            continue
        u0, u1 = sorted((min(max(u0, 0), w), min(max(u1, 0), w)))
        v0, v1 = sorted((min(max(v0, 0), h), min(max(v1, 0), h)))
        if u1 - u0 >= 3 and v1 - v0 >= 3:
            out.append({"label": label, "box": (u0, v0, u1, v1)})
    return out
