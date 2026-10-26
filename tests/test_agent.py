"""projects.agent: measuring from depth, the skills' checks and plan, stack verification and what is learned from it, and
the planner loop with a scripted LLM. No hardware, no network.

    .venv\Scripts\python -m tests.test_agent
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.perception.calibrate import load_calib
from projects.agent import experience, grounding, planner, skills, world as W
from projects.agent.practice import flat_boxes

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
    """objs: [(h, footprint)]; the nearer surface to the camera wins where they overlap."""
    depth, _ = ray_depth(0.0)
    for h, fp in objs:
        d, _ = ray_depth(h, fp)
        depth = np.minimum(depth, d)
    return W.Scene(np.full((480, 640, 3), 90, np.uint8), depth, INTR, T)


def box_for(fp, h):
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


def flat_box(cx, cy, dy=0.0):
    """An 8 x 5 cm top face, long side along x, centred (cx, cy + dy)."""
    return (cx - 0.04, cx + 0.04, cy + dy - 0.025, cy + dy + 0.025)


BASE_XY, LOOSE_XY = (0.26, -0.05), (0.26, -0.20)
CUBE = (0.20, 0.23, 0.05, 0.08)     # a 3 cm cube, 3 cm tall
TRAY = (0.22, 0.30, -0.14, -0.06)   # an 8 cm square, 5 cm tall (the "container")


class FakeGrounder(grounding.Grounder):
    def __init__(self, items):
        self.items = items   # [(label, footprint, height)]

    def detect(self, rgb, queries=None, scene=None):
        return [dict(label=l, box=box_for(fp, h)) for l, fp, h in self.items if not queries or any(q in l for q in queries)]


def make_exp():
    return experience.Experience(Path(tempfile.mkdtemp()) / "exp.json", seed=1)


def test_measure_uses_the_top_face():
    sc = scene_with([(0.03, CUBE)])
    o = W.measure(sc, "cube", box_for(CUBE, 0.03))
    assert np.linalg.norm(o.xy - [0.215, 0.065]) < 0.006, o.xy
    assert abs(o.top_z - 0.03) < 0.004, o.top_z
    assert 0.025 < o.width < 0.036 and 0.025 < o.length < 0.036, (o.length, o.width)
    sc = scene_with([(0.03, flat_box(*BASE_XY))])
    o = W.measure(sc, "box", box_for(flat_box(*BASE_XY), 0.03))
    assert abs(o.length - 0.08) < 0.008 and abs(o.width - 0.05) < 0.008, (o.length, o.width)
    assert np.linalg.norm(o.xy - BASE_XY) < 0.005, o.xy


def test_depth_grounder_finds_the_boxes():
    sc = scene_with([(0.03, flat_box(*BASE_XY)), (0.03, flat_box(*LOOSE_XY))])
    box = skills.SkillBox(grounding.DepthGrounder(), capture=lambda: sc, exp=make_exp())
    objs = box.observe()
    good, odd = flat_boxes(objs)
    assert len(good) == 2 and not odd, [o.describe() for o in objs]
    standing = scene_with([(0.05, (0.23, 0.26, 0.03, 0.11)), (0.03, flat_box(*LOOSE_XY))])   # one on its edge
    objs = skills.SkillBox(grounding.DepthGrounder(), capture=lambda: standing, exp=make_exp()).observe()
    good, odd = flat_boxes(objs)
    assert len(good) == 1 and len(odd) == 1


def test_close_yaw_and_error_frame():
    sc = scene_with([(0.03, flat_box(*BASE_XY))])
    o = W.measure(sc, "box", box_for(flat_box(*BASE_XY), 0.03))
    cy, period = skills.close_yaw(o)
    assert period == np.pi and abs(abs(cy) - np.pi / 2) < 0.08, cy   # long side along x: the jaws close along y
    top = W.measure(scene_with([(0.06, flat_box(*BASE_XY, dy=0.004))]), "box", box_for(flat_box(*BASE_XY, dy=0.004), 0.06))
    es, el, ey = skills.placement_error(o, top)
    assert abs(abs(es) - 0.004) < 0.003 and abs(el) < 0.003 and abs(ey) < 4, (es, el, ey)


def make_box(real=False, after=None, results=("done",), exp=None):
    sc = scene_with([(0.03, flat_box(*BASE_XY)), (0.03, flat_box(*LOOSE_XY))])
    caps = [sc] + (list(after) if isinstance(after, (list, tuple)) else [after] if after else [])
    state = dict(n=0, runs=[])

    def capture():
        s = caps[min(state["n"], len(caps) - 1)]
        state["n"] += 1
        return s

    def run_arm(frames, pick_c, params):
        state["runs"].append((len(frames), pick_c, params))
        return results[min(len(state["runs"]) - 1, len(results) - 1)]
    box = skills.SkillBox(grounding.DepthGrounder(), real=real, capture=capture, run_arm=run_arm, cool=lambda: None,
                          exp=exp or make_exp())
    return box, state


def test_stack_plan_uses_the_proven_recipe():
    box, _ = make_box()
    objs = {o.xy[1] > -0.1: o for o in box.observe()}
    base, loose = objs[True], objs[False]
    frames, info = box.plan(loose, base, None, skills.SET_DOWN)
    assert info["stack"] and info["tactile"] and abs(info["thick"] - 0.03) < 0.005, info
    assert abs(info["pick_c"][2] - 0.015) < 0.005, info["pick_c"]            # the box's centre height
    assert info["params"]["trim_deg"] == experience.START_TRIM_DEG
    assert frames[0]["gripper"] > 60                                          # opens to the real 70/100
    r = box.call("move_object", dict(object_id=loose.id, to_object=base.id))
    assert r.ok and "dry run" in r.text, r.text
    big = skills.SkillBox(FakeGrounder([("tray", TRAY, 0.05)]), capture=lambda: scene_with([(0.05, TRAY)]), exp=make_exp())
    big.look()
    r = big.call("move_object", dict(object_id="tray_1", to_xy=[0.2, 0.1]))
    assert not r.ok and "cannot be grasped" in r.text, r.text


def stacked_scene(dy=0.0, height=0.06):
    return scene_with([(0.03, flat_box(*BASE_XY)), (height, flat_box(*BASE_XY, dy=dy))])


def test_stack_success_is_measured_and_learned():
    box, st = make_box(real=True, after=stacked_scene(dy=0.008))
    objs = {o.xy[1] > -0.1: o for o in box.observe()}
    base, loose = objs[True], objs[False]
    trim0, s0, l0 = box.exp.correction()
    r = box.call("move_object", dict(object_id=loose.id, to_object=base.id, how=skills.SET_DOWN))
    assert r.ok, r.text
    assert len(st["runs"]) == 1 and st["runs"][0][2].tactile
    trim, s1, l1 = box.exp.correction()
    assert abs(s1 - s0) > 0.001, (s0, s1)          # the 8 mm miss across the stack moved the aim offset
    assert box.exp.events[-1]["kind"] == "stack" and box.exp.events[-1]["result"] == "ok"
    # the pile is now seen as one object; its thickness comes from what was remembered
    pile = box.observe()
    top = [o for o in pile if o.base_z > 0]
    assert len(top) == 1 and abs(top[0].base_z - 0.03) < 0.005 and abs(top[0].top_z - 0.06) < 0.006, [o.describe() for o in pile]
    frames, info = box.plan(top[0], None, [0.26, -0.2], skills.SET_DOWN)   # taking it off again
    assert abs(info["thick"] - 0.03) < 0.007 and abs(info["pick_c"][2] - 0.045) < 0.007, info


def test_a_second_look_rescues_a_camera_dropout():
    glitch = scene_with([(0.03, flat_box(*BASE_XY))])     # the loose box vanished: looks like "not on the stack"
    box, _ = make_box(real=True, after=[glitch, stacked_scene(dy=0.002)])
    objs = {o.xy[1] > -0.1: o for o in box.observe()}
    import time as _t
    real_sleep, _t.sleep = _t.sleep, lambda s_: None
    try:
        r = box.call("move_object", dict(object_id=objs[False].id, to_object=objs[True].id))
    finally:
        _t.sleep = real_sleep
    assert r.ok and box.exp.events[-1]["result"] == "ok", r.text


def test_toppled_block_is_reported_not_learned():
    box, _ = make_box(real=True, after=stacked_scene(height=0.11))
    objs = {o.xy[1] > -0.1: o for o in box.observe()}
    before = box.exp.correction()
    r = box.call("move_object", dict(object_id=objs[False].id, to_object=objs[True].id))
    assert not r.ok and "not lying flat" in r.text, r.text
    assert box.exp.correction() == before and box.exp.events[-1]["result"] == "toppled"


def test_missed_grasp_is_recorded_and_aborts_are_errors():
    box, _ = make_box(real=True, results=("missed",))
    objs = {o.xy[1] > -0.1: o for o in box.observe()}
    r = box.call("move_object", dict(object_id=objs[False].id, to_object=objs[True].id))
    assert not r.ok and "missed" in r.text and box.exp.grasp.hist[-1]["reward"] == 0.0, r.text
    box, _ = make_box(real=True, results=("aborted",))
    objs = {o.xy[1] > -0.1: o for o in box.observe()}
    r = box.call("move_object", dict(object_id=objs[False].id, to_object=objs[True].id))
    assert not r.ok and "Do not continue" in r.text


def test_experience_learns_and_persists():
    e = make_exp()
    # a robot that really lands 4 mm too far across and has yaw error = 5 + 0.87 * trim (best trim -5.7 deg)
    for _ in range(8):
        trim, off_s, off_l = e.correction()
        assert e.learn_placement(0.004 + off_s, 0.0 + off_l, 5.0 + 0.87 * trim, (trim, off_s, off_l))
    trim, s, l = e.correction()
    assert -8.5 < trim < -6.0 and -0.0036 < s < -0.0022 and abs(l) < 0.0005, (trim, s, l)
    for _ in range(40):   # pure scatter around no bias does not walk the correction away from where it settled
        before = e.correction()
        e.learn_placement(float(np.random.default_rng(0).normal(0, 0.005)), 0.0, 0.0, before)
    assert abs(e.correction()[1] - s) < 0.003
    assert not e.learn_placement(0.05, 0.0, 0.0, e.correction())    # a placement 5 cm off is a failure, not bias
    for _ in range(12):   # releasing 4 mm higher keeps working, the default keeps failing
        e.learn_release(0.008, 0.015, False)
        e.learn_release(0.004, 0.015, True)
    assert e.choose_release(explore=False)[0] == 0.004
    picks = [e.choose_release(explore=True)[0] for _ in range(40)]
    assert picks.count(0.004) > 25, picks.count(0.004)
    e.log("stack", "ok", es_mm=1.0)
    again = experience.Experience(e.path)
    assert again.correction() == e.correction() and len(again.events) == 1
    assert "1 attempts" in again.summary()
    assert e.choose_release()[0] == 0.004   # no recent trouble: keeps the best one rather than exploring
    # an older file without the sample list is rebuilt from its events
    old = json.loads(e.path.read_text())
    old["state"].pop("samples")
    old["events"] = [dict(kind="stack", result="ok", es_mm=4.0, el_mm=0.0, ey_deg=0.0,
                          params=dict(trim_deg=experience.START_TRIM_DEG, off_short=0.0, off_long=0.0))] * 6
    e.path.write_text(json.dumps(old))
    rebuilt = experience.Experience(e.path)
    assert rebuilt.correction()[1] < -0.0020, rebuilt.correction()


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
    client = ScriptedClient([("move_object", {"object_id": "block_1", "to_object": "block_2"}),
                             ("finish", {"success": True, "summary": "stacked"})])
    ok, summary = planner.run_task("stack the boxes", box, client, max_steps=5)
    assert ok and summary == "stacked"
    assert "learned so far" in client.seen[0][1]["content"][0]["text"]
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
