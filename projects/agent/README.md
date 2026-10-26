# agent

Type a task, the arm does it. An LLM plans over a small library of skills; the Kinect checks every step.

```
.venv\Scripts\python -m projects.agent.main                                   # what does the grounder see?
.venv\Scripts\python -m projects.agent.main "put the cube in the tray"        # plan only, arm not touched
.venv\Scripts\python -m projects.agent.main "put the cube in the tray" --real # real arm (stop_gui.py open)
.venv\Scripts\python -m tests.test_agent                                      # no hardware, no network
```

Needs `ANTHROPIC_API_KEY` and a network connection (planner and vision grounder). The arm and camera are local.

Layers:

* **Grounding** (`grounding.py`): description -> pixel box. `ClaudeGrounder` asks a vision model for boxes; the
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
