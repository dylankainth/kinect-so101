"""projects.agent.primitives / skill_library / mcp_server: the guard rails around skills written on the fly, the library, and
the MCP tools end to end through an in-process client. No hardware, no network.

    .venv\Scripts\python -m tests.test_mcp
"""
import tempfile
from pathlib import Path

import anyio
import numpy as np
from mcp import Client

from core.motion import STREAM_HZ
from projects.agent import grounding, mcp_server, primitives as P, skills
from projects.agent.skill_library import Library
from tests.test_agent import BASE_XY, LOOSE_XY, flat_box, make_exp, scene_with

PICK = '''
def run(arm, x=0.28, y=0.0, lift=0.08):
    arm.jaws(1.0)
    arm.move(x, y, 0.06)
    arm.move(x, y, 0.012)
    arm.grasp()
    arm.move(x, y, lift)
    arm.move(x, y + 0.1, lift)
    arm.move(x, y + 0.1, 0.014)
    arm.jaws(1.0)
    arm.move(x, y + 0.1, 0.07)
'''


def refused(src, args=None):
    try:
        P.rehearse(src, args)
    except P.SkillError as e:
        return str(e)
    raise AssertionError("was not refused:\n" + src)


def test_a_good_skill_rehearses_and_is_described():
    rec = P.rehearse(PICK, dict(x=0.28))
    assert sum(isinstance(e, P.Grasp) for e in rec.events) == 1
    assert 10 < rec.seconds < 60
    assert "z 0.012" in P.describe(rec) or "0.012" in P.describe(rec)
    img = P.preview(rec)
    assert img.ndim == 3 and img.shape[1] == 2 * img.shape[0]


def test_code_that_could_do_harm_is_refused():
    for src, why in [("import os\ndef run(arm):\n    pass", "imports"),
                     ("def run(arm):\n    while True:\n        arm.wait(1)", "While"),
                     ("def run(arm):\n    x = arm.__class__", "__class__"),
                     ("def run(arm):\n    open('x')", "open"),
                     ("def run(arm):\n    return '{0.__class__}'.format(arm)", "format"),
                     ("def run(arm):\n    pass\nrun(None)", "top level"),
                     ("def run(arm):\n    for i in range(10 ** 9):\n        pass", "range"),
                     ("def go(arm):\n    pass", "exactly one"),
                     ("def run(arm):\n    pass", "never moved")]:
        assert why in refused(src), (why, src)


def test_the_arm_limits_apply_to_every_skill():
    for body, why in [("arm.move(0.5, 0, 0.1)", "outside the allowed range"),
                      ("arm.move(0.28, 0, 0.002)", "outside the allowed range"),
                      ("arm.move(0.28, 0, 0.1, speed=0.5)", "speed"),
                      ("arm.move(0.28, 0, 0.1, tilt=1.2)", "tilt"),
                      ("arm.move(0.14, 0, 0.05, tilt=0.6)", "cannot reach"),
                      ("arm.jaws(2)", "between 0 and 1"),
                      ("arm.wait(60)", "wait"),
                      ("for i in range(150):\n        arm.move(0.28, 0, 0.1)\n        arm.move(0.30, 0, 0.1)", "too many")]:
        assert why in refused(f"def run(arm):\n    {body}\n"), body
    assert "unknown arguments" in refused(PICK, dict(speed=1))


def test_tilt_is_chosen_where_the_tool_cannot_stay_vertical():
    arm = P.Recorder()
    assert arm.reach_ok(0.28, 0.0, 0.012) and not arm.reach_ok(0.9, 0, 0.1) and not arm.reach_ok(0.28, 0, 0.001)
    arm.move(0.28, 0.1, 0.08)               # high and out: needs a tilt
    assert arm.tilt >= 0.0 and arm.pos[2] == 0.08


def test_a_rehearsal_streams_to_the_real_arm_in_small_steps():
    rec = P.rehearse(PICK)
    last = None
    for e in rec.events:
        if not isinstance(e, P.Move):
            continue
        frames = P.real_frames(e.q, 70.0)
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        arr = np.array([[f[j] for j in joints] for f in frames])
        assert np.abs(np.diff(arr, axis=0)).max() <= P.MAX_FRAME_STEP_DEG, np.abs(np.diff(arr, axis=0)).max()
        if last is not None:   # each move starts where the previous one ended (within one frame's travel)
            assert np.abs(arr[0] - last).max() <= P.MAX_FRAME_STEP_DEG + 1.0, np.abs(arr[0] - last).max()
        last = arr[-1]
    assert STREAM_HZ == 25.0


def test_library_saves_checks_and_remembers():
    lib = Library(Path(tempfile.mkdtemp()))
    assert lib.save("pick_carry", "carry a thing 10 cm to the left", PICK) == dict(x=0.28, y=0.0, lift=0.08)
    try:
        lib.save("Bad Name", "x", PICK)
        raise AssertionError("bad name accepted")
    except P.SkillError:
        pass
    try:
        lib.save("evil", "x", "import os\ndef run(arm):\n    pass")
        raise AssertionError("imports accepted")
    except P.SkillError:
        pass
    assert lib.names() == ["pick_carry"]
    lib.record_run("pick_carry", {}, "finished.")
    lib.record_verdict("pick_carry", True, "carried")
    assert "1/1 runs judged good" in lib.summary("pick_carry")
    lib.save("pick_carry", "v2", PICK.replace("0.08", "0.09"))     # a rewrite keeps the old code and starts a new record
    assert (lib.dir / "pick_carry.prev.py").exists() and "never judged" in lib.summary("pick_carry")
    assert lib.names() == ["pick_carry"]


def make_server(real=False):
    sc = scene_with([(0.03, flat_box(*BASE_XY)), (0.03, flat_box(*LOOSE_XY))])
    box = skills.SkillBox(grounding.DepthGrounder(), real=real, capture=lambda: sc, cool=lambda: None, exp=make_exp(),
                          run_arm=lambda *a: "done")
    ran = []

    def executor(rec):
        ran.append(rec)
        return P.Outcome(True, ["grasp: the jaws stopped at 28/100 (holding something)"])
    srv = mcp_server.ArmServer(box, Library(Path(tempfile.mkdtemp())), real=real, executor=executor, blocked=lambda: False)
    return srv, ran


def test_server_requires_a_rehearsal_and_a_dry_run_never_moves():
    srv, ran = make_server(real=False)
    srv.write_skill("pick_carry", "carry", PICK)
    try:
        srv.run_skill("pick_carry")
        raise AssertionError("ran without a rehearsal")
    except P.SkillError as e:
        assert "rehearse" in str(e)
    text, img = srv.rehearse_skill("pick_carry", dict(x=0.27))
    assert "rehearsal passed" in text and img.shape[0] > 100
    try:
        srv.run_skill("pick_carry", dict(x=0.30))     # other arguments than the rehearsed ones
        raise AssertionError("ran with other arguments")
    except P.SkillError:
        pass
    text, _ = srv.run_skill("pick_carry", dict(x=0.27))
    assert "dry run" in text and not ran
    srv.write_skill("pick_carry", "carry", PICK + "\n")        # changed code needs a new rehearsal
    try:
        srv.run_skill("pick_carry", dict(x=0.27))
        raise AssertionError("ran changed code")
    except P.SkillError:
        pass


def test_server_real_run_records_and_asks_for_a_look():
    srv, ran = make_server(real=True)
    srv.write_skill("pick_carry", "carry", PICK)
    srv.rehearse_skill("pick_carry")
    text, _ = srv.run_skill("pick_carry")
    assert len(ran) == 1 and text.startswith("finished") and "look" in text
    assert "pick_carry" in srv.skill_outcome("pick_carry", False, "dropped it")
    try:
        srv.run_skill("pick_carry")
        raise AssertionError("ran twice on one rehearsal")
    except P.SkillError:
        pass


def test_a_stop_flag_blocks_a_real_run():
    srv, ran = make_server(real=True)
    srv.blocked = lambda: True
    srv.write_skill("pick_carry", "carry", PICK)
    srv.rehearse_skill("pick_carry")
    try:
        srv.run_skill("pick_carry")
        raise AssertionError("ran with a STOP flag set")
    except skills.SkillError as e:
        assert "STOP" in str(e) and not ran


def test_locate_reads_a_pixel_in_robot_coordinates():
    srv, _ = make_server()
    try:
        srv.locate(10, 10)
        raise AssertionError("located before a look")
    except skills.SkillError:
        pass
    srv.look()
    o = [o for o in srv.box.objs.values() if abs(o.xy[0] - BASE_XY[0]) < 0.02 and abs(o.xy[1] - BASE_XY[1]) < 0.02][0]
    u, v = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
    assert "above the table" in srv.locate(u, v)
    x, y = (float(t.split("=")[1]) for t in srv.locate(u, v).split(" m,")[0].split()[-2:])
    assert abs(x - BASE_XY[0]) < 0.02 and abs(y - BASE_XY[1]) < 0.02, (x, y)


def test_the_mcp_tools_end_to_end():
    srv, ran = make_server(real=True)
    mcp = mcp_server.build_server(srv)

    async def go():
        async with Client(mcp) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"look", "locate", "move_object", "write_skill", "rehearse_skill", "run_skill", "skill_outcome",
                    "list_skills", "read_skill", "stop_arm", "arm_status"} <= names, names
            r = await c.call_tool("look", {})
            kinds = [b.type for b in r.content]
            assert kinds[0] == "text" and "image" in kinds, kinds
            assert "block_1" in r.content[0].text
            r = await c.call_tool("write_skill", dict(name="pick_carry", description="carry", code=PICK))
            assert "saved pick_carry" in r.content[0].text
            r = await c.call_tool("write_skill", dict(name="evil", description="x", code="import os\ndef run(arm):\n    pass"))
            assert "refused" in r.content[0].text and "imports" in r.content[0].text
            r = await c.call_tool("run_skill", dict(name="pick_carry"))
            assert "refused" in r.content[0].text and "rehearse" in r.content[0].text
            r = await c.call_tool("rehearse_skill", dict(name="pick_carry", args=dict(x=0.27)))
            assert "rehearsal passed" in r.content[0].text and any(b.type == "image" for b in r.content)
            r = await c.call_tool("run_skill", dict(name="pick_carry", args=dict(x=0.27)))
            assert r.content[0].text.startswith("finished"), r.content[0].text
            r = await c.call_tool("skill_outcome", dict(name="pick_carry", worked=True, notes="ok"))
            assert "1/1" in r.content[0].text
            r = await c.call_tool("list_skills", {})
            assert "pick_carry" in r.content[0].text
            r = await c.call_tool("move_object", dict(object_id="block_1", to_object="block_2"))
            assert r.content[0].text.startswith("OK") or r.content[0].text.startswith("FAILED"), r.content[0].text
            r = await c.call_tool("arm_status", {})
            assert "REAL" in r.content[0].text
    anyio.run(go)
    assert len(ran) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
