# agent: progress log

Where `projects/agent` got to, what was measured on the real arm, and what is still open.

## What exists

* **Planner + skills** (`planner.py`, `skills.py`): an LLM (OpenRouter) picks camera-verified skills, `look` and `move_object`.
* **Depth grounder** (`grounding.DepthGrounder`): finds loose objects from depth alone, no network.
* **Learning** (`experience.py`): placement bias (windowed mean with shrinkage), release strategy (Beta bandit, explores only after a failure), grasp offset.
* **Practice loop** (`practice.py`): stacks and unstacks the two boxes forever, with no LLM, learning each time.
* **MCP server** (`mcp_server.py`): Claude is the planner and vision model. Tools: `look`, `locate`, `move_object`, `write_skill`,
  `rehearse_skill`, `run_skill`, `skill_outcome`, `list_skills`, `read_skill`, `stop_arm`, `arm_status`.
* **Skills written on the fly** (`primitives.py`, `skill_library.py`): Python over `arm.move/jaws/grasp/wait`, checked before
  they exist, rehearsed (IK, workspace, table, speed, step size, duration) before they run, kept with a verdict per run.

## Measured on the real arm

* **Stack/unstack practice, run 3:** 23 trials, 38 of 40 stacks ok overall, the last five ok. Both failures were Kinect dropouts
  judged as "not on the stack" (the second look added since). Placement scatter about 5 mm across, 3 to 7 deg of yaw.
* **Learned values settled at:** place-yaw trim about -5 deg, aim offsets about +6 / +10 mm.
* **Re-check after the base was re-clamped:** a proven stack move landed within 6 mm, so the calibration holds.
* **First on-the-fly skill that worked:** `pull_in` (closed jaws, the fixed pad's outer face) pulled a loose box 9.6 cm toward the
  robot. It overshot the pad's end by about 2 cm, slid 3 cm sideways and turned a little. Saved with a "good" verdict.

## Things found the hard way

* **`core.calibration.handeye_auto` is harmful here.** It fits a z offset of about 89 mm (it sees the gripper body, not the jaw
  tips) and `aim()` then drives the tool into the table. `control/handeye.json` must not exist. Use `calib_joints` instead.
* **The grasp point is not where the pads are.** One jaw is fixed, the other swings: the fixed pad sits about 2 cm from the grasp
  point and the offset turns with the arm. Skills that must touch something use `arm.move(..., tip="fixed"|"moving")`.
* **A loose base looks like a software bug.** The arm's base came loose from the table mid-session. The encoders kept agreeing
  with the commands while the real arm drifted about 40 deg, and the parked arm crept from y 0 to y -0.11. Check the parked
  arm's position in `look` (about x 0.16, y 0.00) before blaming IK or calibration.
* **The Kinect drops off USB** now and then; `world.capture` waits up to 20 minutes for it to return.
* **Measure the top face only.** Side faces seen at an angle inflate the size and shift the centre.
* **The gripper servo overheats.** Every real run waits for the servos to cool below 42 C; it is most of the wall-clock time.

## Not done

* The MCP server has been exercised in-process and over stdio, and its skills were run by script, but it has not been registered in
  Claude Code and driven by Claude through the tools yet (`claude mcp add kinect-arm -- <venv python> -m projects.agent.mcp_server --real`).
* Only one on-the-fly skill (`pull_in`) has been run on the real arm. Flipping a box, pushing to a target and cloth folding are untried.
* The vision grounder and the OpenRouter planner path were not run end to end on the real arm after the stack recipe was fixed.
* The rehearsal checks reach, workspace and table clearance in simulation. It does not model contact, so a push or grasp can still
  fail; the camera check after each run is what catches that.
* `core/tools/view_kinect.py` holds the Kinect while open: close it before the arm tools run.
