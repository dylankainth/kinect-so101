"""Type a task, the arm does it: an LLM plans over the arm's skills, the Kinect checks every step.

    .venv\Scripts\python -m projects.agent.main                                   # what does the grounder see?
    .venv\Scripts\python -m projects.agent.main "put the glass next to the tray"  # plan with the arm untouched
    .venv\Scripts\python -m projects.agent.main "..." --real                      # real arm (keep core/tools/stop_gui open)

Needs ANTHROPIC_API_KEY and a network connection for the planner and the vision grounder (the arm and camera are local).
"""
import argparse
import logging

from core import control
from projects.agent import grounding, planner
from projects.agent.skills import SkillBox
from projects.agent.world import annotate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="what to do, in plain language; without it, only look")
    ap.add_argument("--real", action="store_true", help="move the real arm (default: plan only)")
    ap.add_argument("--yes", action="store_true", help="do not ask GO before each move")
    ap.add_argument("--model", default=planner.DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=12, help="most skill calls before giving up")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    import anthropic
    client = anthropic.Anthropic()
    confirm = (lambda what: True) if a.yes else (lambda what: input(f"{what}\nType GO to start: ").strip() == "GO")
    box = SkillBox(grounding.ClaudeGrounder(a.model, client), real=a.real, confirm=confirm)
    if not a.task:
        res = box.look()
        print(res.text)
        annotate(box.scene.rgb, list(box.objs.values()), control.DIR / "agent_look.jpg")
        print("picture in control/agent_look.jpg")
        return
    ok, summary = planner.run_task(a.task, box, client, a.model, a.steps)
    print(("DONE: " if ok else "NOT DONE: ") + summary)


if __name__ == "__main__":
    main()
