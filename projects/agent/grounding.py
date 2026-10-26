"""Open-vocabulary grounding: a text description -> pixel boxes in the colour image.

The Grounder interface is one method, so a local detector (Grounding DINO + SAM, OWL-ViT) can replace the Claude
vision backend without touching the skills or the planner.
"""
import base64
import json
import re

import cv2

DEFAULT_MODEL = "claude-sonnet-5-5"

PROMPT = """This is a {w}x{h} pixel colour image from a camera looking down at a robot's workspace (a table, the robot arm at the top).
{ask}
Reply with only a JSON list: [{{"label": "<short name>", "box": [left, top, right, bottom]}}, ...] in image pixel
coordinates, tight boxes around each whole object. An empty list if there are none. Do not box the robot arm, the
table or the mat."""


class Grounder:
    def detect(self, rgb, queries=None):
        """-> [{"label": str, "box": (u0, v0, u1, v1)}]. queries: descriptions to find, or None for every loose object."""
        raise NotImplementedError


class ClaudeGrounder(Grounder):
    def __init__(self, model=DEFAULT_MODEL, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model = client, model

    def detect(self, rgb, queries=None):
        h, w = rgb.shape[:2]
        ask = ("Find: " + "; ".join(queries) + ". Include every match." if queries
               else "List every loose object on the table that the robot could pick up.")
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        img = base64.standard_b64encode(jpg.tobytes()).decode()
        msg = self.client.messages.create(
            model=self.model, max_tokens=1500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
                {"type": "text", "text": PROMPT.format(w=w, h=h, ask=ask)}]}])
        return parse_boxes("".join(b.text for b in msg.content if b.type == "text"), w, h)


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
