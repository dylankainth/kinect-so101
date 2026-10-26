"""The arm as an MCP server: plug it into Claude and give it tasks in plain language.

    claude mcp add kinect-arm -- C:\\Users\\dylan\\Documents\\GitHub\\kinect-so101\\.venv\\Scripts\\python.exe -m projects.agent.mcp_server --real

Claude is the planner and the vision model here: it looks through the Kinect (`look`, `locate`), moves rigid things with
`move_object`, and for anything else it writes a skill on the fly (`write_skill`), rehearses it (`rehearse_skill`: IK,
workspace, table and step checks, nothing moves), runs it (`run_skill`), looks again and records whether it worked
(`skill_outcome`). Skills are kept in projects/agent/skill_library/ and reused. No API key and no network are needed.

Without --real the arm is never moved (run_skill and move_object only plan). Keep core/tools/stop_gui.py open while the
arm moves; `stop_arm` sets the same STOP flag as its button.
"""
import argparse
import json
import logging
import sys
import threading

import anyio
import cv2
import numpy as np
from mcp.server.mcpserver import Image, MCPServer

from core import control
from projects.agent import grounding, primitives
from projects.agent import world as W
from projects.agent.skill_library import Library
from projects.agent.skills import SkillBox, SkillError

log = logging.getLogger("agent.mcp")

INSTRUCTIONS = """\
This server drives an SO-101 robot arm with a Kinect v1 looking down at a mat. Coordinates are the robot frame in metres: x
forward from the base, y to the left, z up from the table. The arm reaches roughly x 0.15..0.36, |y| < 0.25; the grasp point
is midway between the fingertips.

Workflow:
1. `look` first: a picture (with a grid of robot-frame x,y marks on the table) and the loose objects the depth camera finds
   (every loose object is called block; position, size, height). For anything it does not list (cloth, a glass, a flat
   thing) read the picture and use `locate(u, v)` on a pixel to get its robot-frame position and height.
2. Rigid things 6 cm or less across: `move_object` (grasp, carry, set down or stack, checked by the camera). It is the
   proven path (boxes stack to 5 mm), so prefer it.
3. Anything else: write a skill with `write_skill`: a Python function `def run(arm, x=0.28, y=0.0): ...` made of arm calls.
   `rehearse_skill` it first (nothing moves; it returns a picture of the path) and fix what it refuses. Then `run_skill`,
   then `look` to judge the result, then `skill_outcome` (worked or not, and why). If it failed, change the skill and
   try again: a different approach height, speed, jaw opening or grasp position. `list_skills` shows what already exists.
4. After every motion, look again before the next: the arm folds to rest at the end of each skill, so the table is
   visible. Never assume a move worked.

The arm object in a skill (units: metres, radians, seconds):
  arm.move(x, y, z, tilt=None, roll=None, yaw=None, speed=0.08, tip="grasp")  straight-line move of the grasp point, or
                                   with tip="fixed"/"moving" of the centre of that pad (see below)
  arm.jaws(opening, seconds=0.8)   0 shut .. 1 wide open (about 8 cm between the pads)
  arm.grasp(expect=True)           close slowly on what is between the jaws and hold; stops the skill if it closes on nothing
  arm.wait(seconds), arm.pos, arm.reach_ok(x, y, z), arm.pad_offset("fixed"), arm.log(text)
Only these, `math`, for-loops over range(), if/else, plain functions and arithmetic are allowed (no imports, no while).

The pads are NOT at the grasp point: one jaw is fixed, the other swings, and the pads sit centimetres to the side of it.
A skill that must touch something with the hand (pushing with the closed jaws, pressing, hooking a corner) has to use
tip="fixed" or tip="moving" so that pad, not the grasp point, goes where you say. (A first push aimed with the grasp point
passed the arm cleanly over a box without touching it.)

Things learned on this arm: for the boxes, the grasp point went 8 mm above the object's centre, coming straight down from
above with the jaws wide open. The grasp point is not the middle of the open jaws (one jaw is fixed, the other swings), so
a thing wider than about 4 cm centred on the grasp point lands against the fixed pad; judge by rehearsing, running and
looking rather than by a rule. z below 0.006 m and speeds above 0.15 m/s are refused. The gripper servo warms while it
holds something: the server waits for it to cool between runs. Only one call moves the arm at a time; `stop_arm` stops it
at once.
"""


class ArmServer:
    """The state behind the tools; the tool functions are thin wrappers so tests can call these directly."""

    def __init__(self, box, library=None, real=False, executor=primitives.execute, blocked=None):
        self.box, self.lib, self.real, self.executor = box, library or Library(), real, executor
        self.blocked = blocked or (lambda: any(control.is_set(f) for f in (control.STOP, control.PARK, control.LIMP)))
        self.rehearsed = {}     # skill name -> hash of the code and arguments that passed a rehearsal
        self.lock = threading.Lock()

    # -- seeing -----------------------------------------------------------------------------------------------------
    def view(self):
        img = W.annotate(self.box.scene.rgb, list(self.box.objs.values()))
        return grid_overlay(img, self.box.scene)

    def look(self, find=None):
        res = self.box.look(find)
        return res.text, self.view()

    def locate(self, u, v):
        sc = self.box.scene
        if sc is None:
            raise SkillError("call look first")
        h, w = sc.depth.shape
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < w and 0 <= v < h):
            raise SkillError(f"pixel ({u}, {v}) is outside the {w}x{h} picture")
        pts = sc.points()[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(-1)]
        if len(pts) < 5:
            return f"the Kinect has no depth at pixel ({u}, {v}) (too dark, shiny or out of range)"
        x, y, z = np.median(pts, axis=0)
        return f"pixel ({u}, {v}) is at x={x:.3f} y={y:+.3f} m, {z * 100:.1f} cm above the table"

    # -- skills -------------------------------------------------------------------------------------------------------
    def write_skill(self, name, description, code):
        params = self.lib.save(name, description, code)
        self.rehearsed.pop(name, None)
        return f"saved {name}({', '.join(f'{k}={v!r}' for k, v in params.items())}). Rehearse it before running it."

    def rehearse_skill(self, name, args=None):
        args = args or {}
        code = self.lib.code(name)
        rec = primitives.rehearse(code, args)
        self.rehearsed[name] = primitives.source_hash(code, args)
        return f"rehearsal passed: {primitives.describe(rec)}", primitives.preview(rec)

    def run_skill(self, name, args=None):
        args = args or {}
        code = self.lib.code(name)
        if self.rehearsed.get(name) != primitives.source_hash(code, args):
            raise SkillError(f"rehearse {name} with exactly these arguments first (rehearse_skill); a changed skill or "
                             f"changed arguments need a new rehearsal")
        rec = primitives.rehearse(code, args)
        if not self.real:
            return f"[dry run: the server was started without --real] {primitives.describe(rec)}", None
        if self.blocked():
            raise SkillError("a STOP/PARK/LIMP flag is set: the operator must clear it in stop_gui.py")
        out = self.executor(rec)
        text = ("finished. " if out.ok else "did NOT finish. ") + " | ".join(out.notes)
        self.lib.record_run(name, args, text)
        if not out.ok and any("aborted" in n for n in out.notes):
            return text, None
        self.rehearsed.pop(name, None)      # the next run needs a fresh rehearsal against a fresh look
        return text + " Now `look` to see what happened.", None

    def skill_outcome(self, name, worked, notes=""):
        self.lib.record_verdict(name, worked, notes)
        return self.lib.summary(name)

    def list_skills(self):
        names = self.lib.names()
        return "\n".join(self.lib.summary(n) for n in names) if names else "no skills yet; write one with write_skill"

    def status(self):
        flags = [f.name for f in (control.STOP, control.PARK, control.LIMP) if control.is_set(f)]
        try:
            last = control.STATUS.read_text()
        except OSError:
            last = ""
        return (f"{'REAL arm: it will move' if self.real else 'dry run: it will not move'}; flags set: "
                f"{', '.join(flags) or 'none'}; stop_gui running: {control.runner_alive(60)}; last status: {last or 'n/a'}")


def grid_overlay(img, scene):
    """Small marks at robot-frame (x, y) grid points on the table, so positions can be read off the picture."""
    R, t = scene.T[:3, :3], scene.T[:3, 3]
    fx, fy, cx, cy = (scene.intr[k] for k in ("fx", "fy", "cx", "cy"))
    out = img.copy()
    for x in np.arange(0.15, 0.41, 0.05):
        for y in np.arange(-0.20, 0.21, 0.10):
            c = (np.array([x, y, 0.0]) - t) @ R
            if c[2] <= 0:
                continue
            u, v = int(fx * c[0] / c[2] + cx), int(fy * c[1] / c[2] + cy)
            if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
                cv2.drawMarker(out, (u, v), (0, 200, 255), cv2.MARKER_CROSS, 8, 1)
                cv2.putText(out, f"{x:.2f},{y:+.1f}", (u + 3, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1)
    return out


def jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Image(data=buf.tobytes(), format="jpeg")


def build_server(srv):
    """The MCPServer with a tool per ArmServer method. A tool that errors returns the message instead of raising."""
    mcp = MCPServer("kinect-arm", instructions=INSTRUCTIONS)

    async def busy(fn, *a, **k):
        """Run fn in a worker thread; only one hardware call at a time (stop_arm and arm_status never wait)."""
        if not srv.lock.acquire(blocking=False):
            return ["the arm is busy with another call; wait for it, or call stop_arm"]

        def work():
            try:
                return fn(*a, **k)
            finally:
                srv.lock.release()
        try:
            out = await anyio.to_thread.run_sync(work)
        except SkillError as e:
            return [f"refused: {e}"]
        except Exception as e:  # noqa: BLE001  hardware and camera faults: report, do not kill the server
            log.exception("tool failed")
            return [f"failed: {type(e).__name__}: {e}"]
        return [out] if isinstance(out, str) else [out[0]] + ([jpeg(out[1])] if out[1] is not None else [])

    @mcp.tool()
    async def look(find: list[str] | None = None):
        """Take a fresh Kinect picture. Returns it (with robot-frame x,y marks on the table) and the loose objects the depth
        camera finds with positions, sizes and heights in the robot frame. `find` is ignored by the depth grounder."""
        return await busy(srv.look, find)

    @mcp.tool()
    async def locate(u: int, v: int):
        """Robot-frame position and height above the table of a pixel (u right, v down, in the 640x480 picture from the last
        look). Use it for things the depth camera does not list: read the picture, pick the pixel, ask where it is."""
        return await busy(lambda: srv.locate(u, v))

    @mcp.tool()
    async def move_object(object_id: str, to_object: str | None = None, to_xy: list[float] | None = None,
                          how: str = "set_down"):
        """Grasp one rigid object listed by the last look (top face up to 6 cm across, up to 15 cm tall) and set it down on
        another object (stacking, lined up) or at to_xy=[x, y]; how='drop_in' lets go over a container. The arm returns to
        rest and the camera checks the result. The proven path for boxes. Object ids change on every look."""
        def go():
            res = srv.box.call("move_object", dict(object_id=object_id, to_object=to_object, to_xy=to_xy, how=how))
            return ("OK: " if res.ok else "FAILED: ") + res.text, srv.view() if srv.box.scene is not None else None
        return await busy(go)

    @mcp.tool()
    async def write_skill(name: str, description: str, code: str):
        """Save a skill: Python source defining `def run(arm, ...)` with arguments that have defaults. See the server
        instructions for the arm calls. The code is checked (no imports or while loops, only the arm API) but not run."""
        return await busy(srv.write_skill, name, description, code)

    @mcp.tool()
    async def rehearse_skill(name: str, args: dict | None = None):
        """Dry-run a skill with these arguments: solves IK for every waypoint and checks the workspace, reach, table, speed and
        duration. Nothing moves. Returns what it would do and a top and side picture of the path (green = jaws free, red =
        holding). Required before run_skill, with the same arguments."""
        return await busy(srv.rehearse_skill, name, args)

    @mcp.tool()
    async def run_skill(name: str, args: dict | None = None):
        """Run a rehearsed skill on the real arm (the same name and arguments that passed rehearse_skill). The arm ends at
        rest. Then call look to judge the result."""
        return await busy(srv.run_skill, name, args)

    @mcp.tool()
    async def skill_outcome(name: str, worked: bool, notes: str = ""):
        """Record whether the last run of a skill did what was wanted (judged from a look), and why not if it did not."""
        return await busy(srv.skill_outcome, name, worked, notes)

    @mcp.tool()
    async def list_skills():
        """The saved skills with their arguments and how their runs were judged. Read one's code with read_skill."""
        return await busy(srv.list_skills)

    @mcp.tool()
    async def read_skill(name: str):
        """The source of a saved skill and the record of its runs."""
        return await busy(lambda: srv.lib.code(name) + "\n# runs: " + json.dumps(srv.lib.meta(name)["runs"][-5:]))

    @mcp.tool()
    def stop_arm() -> str:
        """Emergency stop: the arm holds where it is. The operator clears it with PARK or LIMP in stop_gui.py."""
        control.request(control.STOP)
        return "STOP flag set: the arm holds. The operator must press PARK or LIMP in stop_gui.py."

    @mcp.tool()
    def arm_status() -> str:
        """Whether the arm is real or dry-run, which stop flags are set and the last status line."""
        return srv.status()

    return mcp


def make(real=False, grounder=None):
    box = SkillBox(grounder or grounding.DepthGrounder(), real=real)
    srv = ArmServer(box, real=real)
    return srv, build_server(srv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="move the real arm (default: dry run)")
    ap.add_argument("--http", type=int, metavar="PORT", help="serve on http://127.0.0.1:PORT/mcp instead of stdio")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    srv, mcp = make(real=a.real)
    log.info("kinect-arm MCP server: %s", srv.status())
    if a.http:
        mcp.settings.host, mcp.settings.port = "127.0.0.1", a.http
        mcp.run("streamable-http")
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
