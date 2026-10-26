"""Type a task, the arm does it: an LLM plans over the arm's skills, the Kinect checks every step.

    .venv\Scripts\python -m projects.agent.main                                   # prompt: type tasks one after another
    .venv\Scripts\python -m projects.agent.main --real                            # same, with the real arm (keep core/tools/stop_gui open)
    .venv\Scripts\python -m projects.agent.main "put the glass next to the tray"  # one task, then exit
    .venv\Scripts\python -m projects.agent.main --look                            # only: what does the grounder see?

At the prompt: a task in plain language, `look` to see what the camera finds, `quit` to leave. Without --real the arm
is never moved (moves are only planned). Each real move asks for GO unless --yes.

Needs OPENROUTER_API_KEY and a network connection for the planner and the vision grounder (the arm and camera are local).
OPENROUTER_MODEL (or --model) picks the model; it must accept images and tool calls.
"""
import argparse
import logging

from core import control
from projects.agent import grounding, llm, planner
from projects.agent.skills import SkillBox
from projects.agent.world import annotate


def look(box):
    res = box.look()
    print(res.text)
    annotate(box.scene.rgb, list(box.objs.values()), control.DIR / "agent_look.jpg")
    print("picture in control/agent_look.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="one task, then exit; without it, an interactive prompt")
    ap.add_argument("--look", action="store_true", help="only detect and measure, then exit")
    ap.add_argument("--real", action="store_true", help="move the real arm (default: plan only)")
    ap.add_argument("--yes", action="store_true", help="do not ask GO before each move")
    ap.add_argument("--grounder", choices=("vision", "depth"), default="vision",
                    help="vision: a model names and boxes objects (any object, needs the network); depth: every loose "
                         "object standing above the table, from depth alone (no network, objects are all called block)")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=12, help="most skill calls before giving up")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    client = None if (a.look and a.grounder == "depth") else llm.make_client()   # --look with depth needs no network
    confirm = (lambda what: True) if a.yes else (lambda what: input(f"{what}\nType GO to start: ").strip() == "GO")
    grounder = grounding.VisionGrounder(a.model, client) if a.grounder == "vision" else grounding.DepthGrounder()
    box = SkillBox(grounder, real=a.real, confirm=confirm)
    if a.look:
        return look(box)

    def run(task):
        ok, summary = planner.run_task(task, box, client, a.model, a.steps)
        print(("DONE: " if ok else "NOT DONE: ") + summary)

    if a.task:
        return run(a.task)
    print(f"Arm: {'REAL, it will move' if a.real else 'dry run, it will not move'}. Type a task, 'look', or 'quit'.")
    while True:
        try:
            task = input("\ntask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if task.lower() in ("quit", "exit", "q"):
            return
        if task.lower() == "look":
            look(box)
        elif task:
            run(task)


if __name__ == "__main__":
    main()
