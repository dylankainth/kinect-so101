# projects/

One folder per thing you build on the arm. A project imports from `core/` and never from another
project; anything two projects need goes into `core/` instead.

```
projects/
  pick_place/   scripted Kinect pick-and-place (run_pick_place.py, learner.py, RESULTS.md)
  stack_blocks/ stack three blocks, colour-agnostic, re-looks between moves
  template/     copy this to start a new project
```

New project:

1. `cp -r projects/template projects/<name>` (keep the `__init__.py`).
2. Edit `main.py`; put results and notes in the project folder.
3. Run from the repo root: `.venv\Scripts\python -m projects.<name>.main`.

Per-project data that should not be committed (logs, learned offsets) goes in `control/` (gitignored).
Per-object colour/size profiles currently live in `core/perception/config.json` under `objects`.
