"""projects.agent: measuring an object from depth, the skills' checks and plan, and the planner loop with a scripted LLM.

    .venv\Scripts\python -m tests.test_agent
"""
import json
from types import SimpleNamespace

import numpy as np

from core.perception.calibrate import load_calib
from projects.agent import grounding, planner, skills, world as W

INTR = dict(fx=529.2, fy=525.6, cx=328.9, cy=267.5)
T = load_calib()


def ray_depth(h, footprint=None):
    """Depth image of a flat table (z = 0) with, optionally, a box of height h over footprint (x0, x1, y0, y1)."""
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    ray = np.stack([(u - INTR["cx"]) / INTR["fx"], (v - INTR["cy"]) / INTR["fy"], np.ones_like(u, float)], -1) @ T[:3, :3].T
    o = T[:3, 3]

    def hit(z):
        t = (z - o[2]) / ray[..., 2]
        return t, o + ray * t[..., None]
    t0, _ = hit(0.0)
    depth = t0.copy()
    if footprint:
        t1, p = hit(h)
        x0, x1, y0, y1 = footprint
        on = (p[..., 0] > x0) & (p[..., 0] < x1) & (p[..., 1] > y0) & (p[..., 1] < y1) & (t1 > 0)
        depth[on] = t1[on]
    return depth.astype(np.float32), ray


def scene_with(objs):
    """objs: [(h, footprint)]; later ones are drawn over earlier ones where they overlap."""
    depth, _ = ray_depth(0.0)
    for h, fp in objs:
        d, _ = ray_depth(h, fp)
        depth = np.minimum(depth, d)
    return W.Scene(np.full((480, 640, 3), 90, np.uint8), depth, INTR, T)


def box_for(scene, fp, h):
    """Pixel box around the footprint corners at height h."""
    inv = np.linalg.inv(T)
    x0, x1, y0, y1 = fp
    uv = []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (0.0, h):
                p = inv @ np.array([x, y, z, 1.0])
                uv.append((p[0] / p[2] * INTR["fx"] + INTR["cx"], p[1] / p[2] * INTR["fy"] + INTR["cy"]))
    uv = np.array(uv)
    return (*uv.min(0) - 3, *uv.max(0) + 3)


CUBE = (0.20, 0.23, 0.05, 0.08)     # a 3 cm cube, 3 cm tall
TRAY = (0.22, 0.30, -0.14, -0.06)   # an 8 cm square, 5 cm tall (the "container")


class FakeGrounder(grounding.Grounder):
    def __init__(self, scene, items):
        self.scene, self.items = scene, items   # [(label, footprint, height)]

    def detect(self, rgb, queries=None):
        return [dict(label=l, box=box_for(self.scene, fp, h)) for l, fp, h in self.items
                if not queries or any(q in l for q in queries)]


def test_measure():
    sc = scene_with([(0.03, CUBE)])
    o = W.measure(sc, "cube", box_for(sc, CUBE, 0.03))
    assert np.linalg.norm(o.xy - [0.215, 0.065]) < 0.01, o.xy
    assert abs(o.top_z - 0.03) < 0.004, o.top_z
    assert 0.025 < o.width < 0.04 and 0.025 < o.length < 0.04, (o.length, o.width)
    assert W.measure(sc, "nothing", (10, 10, 60, 60)) is None or W.measure(sc, "nothing", (10, 10, 60, 60)).flat


def make_box(real=False, ran=None, after=None):
    sc = scene_with([(0.03, CUBE), (0.05, TRAY)])
    g = FakeGrounder(sc, [("cube", CUBE, 0.03), ("tray", TRAY, 0.05)])
    captures = [sc] + ([after] if after else [])
    state = {"n": 0}

    def capture():
        s = captures[min(state["n"], len(captures) - 1)]
        state["n"] += 1
        g.scene = s
        return s

    def run_arm(batch):
        ran.append(batch)
        return [b[0] for b in batch]
    return skills.SkillBox(g, real=real, capture=capture, run_arm=run_arm, cool=lambda: None), g


def test_look_and_plan():
    box, _ = make_box()
    r = box.call("look", {})
    assert r.ok and "cube_1" in r.text and "tray_1" in r.text, r.text
    r = box.call("move_object", {"object_id": "cube_1", "to_object": "tray_1", "how": "drop_in"})
    assert r.ok and "dry run" in r.text, r.text
    assert not box.call("move_object", {"object_id": "nope", "to_xy": [0.2, 0.0]}).ok
    assert not box.call("move_object", {"object_id": "cube_1"}).ok
    assert not box.call("move_object", {"object_id": "cube_1", "to_xy": [0.9, 0.9]}).ok   # out of reach
    big = box.call("move_object", {"object_id": "tray_1", "to_xy": [0.2, 0.1]})
    assert not big.ok and "cannot be grasped" in big.text, big.text


def test_real_verify():
    ran = []
    # after the move the cube is gone from its start and sits on the tray's far side: drop_in is believed
    box, g = make_box(real=True, ran=ran, after=scene_with([(0.05, TRAY)]))
    box.call("look", {})
    g.items = [("tray", TRAY, 0.05)]
    r = box.call("move_object", {"object_id": "cube_1", "to_object": "tray_1", "how": "drop_in"})
    assert r.ok and ran and ran[0][0][0] == "cube_1", r.text
    # the cube is still where it was: the move is reported as failed
    ran.clear()
    box, g = make_box(real=True, ran=ran, after=scene_with([(0.03, CUBE), (0.05, TRAY)]))
    box.call("look", {})
    r = box.call("move_object", {"object_id": "cube_1", "to_object": "tray_1", "how": "drop_in"})
    assert not r.ok and "still where it started" in r.text, r.text


class ScriptedClient:
    """Plays a fixed list of (tool, input) calls, one per turn, and records the messages it was sent."""
    def __init__(self, script):
        self.script, self.i, self.seen = script, 0, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.seen.append(list(kw["messages"]))
        name, args = self.script[self.i]
        self.i += 1
        call = SimpleNamespace(id=f"t{self.i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])


def test_planner_loop():
    box, _ = make_box()
    client = ScriptedClient([("move_object", {"object_id": "cube_1", "to_object": "tray_1", "how": "drop_in"}),
                             ("finish", {"success": True, "summary": "cube in tray"})])
    ok, summary = planner.run_task("put the cube in the tray", box, client, max_steps=5)
    assert ok and summary == "cube in tray"
    tool_msg = [m for m in client.seen[1] if m["role"] == "tool"][0]
    assert tool_msg["content"].startswith("OK:") and "dry run" in tool_msg["content"]
    assert client.seen[1][-1]["role"] == "user" and client.seen[1][-1]["content"][1]["type"] == "image_url"
    ok, _ = planner.run_task("x", box, ScriptedClient([("look", {})] * 3), max_steps=3)
    assert not ok


def test_parse_boxes():
    txt = 'Sure:\n[{"label": "mug", "box": [10, 20, 100, 90]}, {"label": "bad"}, {"label": "tiny", "box": [1, 1, 2, 2]}]'
    out = grounding.parse_boxes(txt, 640, 480)
    assert out == [{"label": "mug", "box": (10.0, 20.0, 100.0, 90.0)}], out
    assert grounding.parse_boxes("no objects", 640, 480) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
