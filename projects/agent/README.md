# agent

Type a task, the arm does it. An LLM plans over a small library of skills; the Kinect checks every step.

```
.venv\Scripts\python -m projects.agent.main                                   # prompt: type tasks (plan only)
.venv\Scripts\python -m projects.agent.main --look                             # what does the grounder see?
.venv\Scripts\python -m projects.agent.main "put the cube in the tray"        # plan only, arm not touched
.venv\Scripts\python -m projects.agent.main --real                            # prompt with the real arm (stop_gui.py open)
.venv\Scripts\python -m tests.test_agent                                      # no hardware, no network
```

Needs `OPENROUTER_API_KEY` and a network connection (planner and vision grounder). `OPENROUTER_MODEL` or `--model` picks
the model (any OpenRouter model that takes images and tool calls; the default slug in `llm.py` is unverified). The arm and camera are local.

Layers:

* **Grounding** (`grounding.py`): description -> pixel box. `VisionGrounder` asks a vision model for boxes; the
  `Grounder` interface is one method, so Grounding DINO / OWL-ViT can replace it.
* **Measuring** (`world.py`): the Kinect depth under a box, through the calibration, gives the object's position,
  height, footprint and direction in the robot frame.
* **Skills** (`skills.py`): `look` and `move_object` (set down on a spot/object, or drop into a container). Each ends with the arm
  at rest and a camera check. It reuses `core.sim.pick_place` for IK and `put_screws_away.chain.run_chain` for execution
  and the STOP/PARK/LIMP handling. The planner never sets joints.
* **Planner** (`planner.py`): tool-use loop; one skill per turn, result plus a fresh annotated image fed back, ends with `finish`.

Not tried on the real arm yet. Limits: objects up to 5 cm across and 15 cm tall (`MAX_GRASP_WIDTH` is a guess from the
sim's ~6 cm open jaws), top-down-ish grasps only, so "flip the box" is refused rather than attempted. New skills (push,
rotate, regrasp) go in `SkillBox.skills` with a JSON schema.

## As an MCP server (plug the arm into Claude)

`mcp_server.py` makes Claude the planner and the vision model; no API key or network is needed on this side.

```
claude mcp add kinect-arm -- C:\Users\dylan\Documents\GitHub\kinect-so101\.venv\Scripts\python.exe -m projects.agent.mcp_server          # dry run: nothing moves
claude mcp add kinect-arm -- C:\Users\dylan\Documents\GitHub\kinect-so101\.venv\Scripts\python.exe -m projects.agent.mcp_server --real   # moves the real arm (keep core/tools/stop_gui.py open)
```

Tools: `look` (picture with robot-frame marks, loose objects from depth), `locate(u, v)` (pixel -> robot position and height),
`move_object` (the proven grasp/stack path), `write_skill` / `rehearse_skill` / `run_skill` / `skill_outcome` / `list_skills` /
`read_skill` (skills written on the fly), `stop_arm`, `arm_status`.

A skill is a short Python function over a few arm calls (`primitives.py`): `arm.move(x, y, z, tilt, roll, yaw, speed)`,
`arm.jaws(opening)`, `arm.grasp()`, `arm.wait(s)`, `arm.reach_ok(...)`. The flow is write -> rehearse -> run -> look -> record the verdict:

* **Checked before it exists.** The code may not import, loop with `while`, or reach into attributes. It cannot move the arm any way except the arm calls. (This stops mistakes; it is not a sandbox against hostile code.)
* **Rehearsed before it moves.** `rehearse_skill` solves IK for every waypoint and checks the workspace box (x 0.12..0.36, |y| <= 0.25, z 0.006..0.20), reach, link-to-table clearance, speed (<= 0.15 m/s), per-frame joint steps, step count and duration, and returns a picture of the path. `run_skill` refuses anything that was not rehearsed with the same code and arguments, and needs a fresh rehearsal after every run.
* **Run like the proven path.** It streams through SafeArm (STOP/PARK/LIMP honoured, load/temperature/voltage aborts), a `grasp()` that closes on nothing stops the skill, and the arm folds to rest at the end.
* **Kept and improved.** Skills live in `skill_library/<name>.py` with a record of each run and the planner's verdict; rewriting one keeps the old code as `.prev.py`.

`python -m tests.test_mcp` runs all of this without hardware, including the tools through an in-process MCP client.
