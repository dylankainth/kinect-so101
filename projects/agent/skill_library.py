"""Skills written on the fly, kept on disk so they can be reused and improved.

projects/agent/skill_library/<name>.py is the code (see primitives.py for what it may do) and <name>.json the record that
goes with it: the description, the arguments it takes, and every run with the camera's verdict. A skill that keeps
working shows a good record; one that does not is rewritten (write_skill with the same name keeps the old code as
<name>.prev.py and starts a new record).
"""
import json
import re
import time
from pathlib import Path

from projects.agent import primitives

DIR = Path(__file__).with_name("skill_library")
NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_RUNS = 30


class Library:
    def __init__(self, directory=DIR):
        self.dir = Path(directory)
        self.dir.mkdir(exist_ok=True)

    def _paths(self, name):
        if not NAME.match(name or ""):
            raise primitives.SkillError("a skill name is lowercase letters, digits and _, starting with a letter (40 at most)")
        return self.dir / f"{name}.py", self.dir / f"{name}.json"

    def save(self, name, description, code):
        """Check and store a skill. Returns its arguments (name -> default). Raises SkillError if the code is not allowed."""
        py, js = self._paths(name)
        _, params = primitives.compile_skill(code)
        if py.exists():
            py.with_suffix(".prev.py").write_text(py.read_text(encoding="utf-8"), encoding="utf-8")
        py.write_text(code, encoding="utf-8")
        meta = dict(description=description.strip(), params={k: repr(v) for k, v in params.items()},
                    created=time.strftime("%Y-%m-%d %H:%M"), runs=[])
        js.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return params

    def code(self, name):
        py, _ = self._paths(name)
        if not py.exists():
            raise primitives.SkillError(f"no skill called {name!r}; known: {', '.join(self.names()) or 'none yet'}")
        return py.read_text(encoding="utf-8")

    def meta(self, name):
        _, js = self._paths(name)
        return json.loads(js.read_text(encoding="utf-8")) if js.exists() else dict(description="", params={}, runs=[])

    def names(self):
        return sorted(p.stem for p in self.dir.glob("*.py") if not p.stem.endswith(".prev"))

    def record_run(self, name, args, result):
        _, js = self._paths(name)
        meta = self.meta(name)
        meta["runs"] = (meta["runs"] + [dict(at=time.strftime("%Y-%m-%d %H:%M"), args=args, result=result[:300],
                                              worked=None)])[-MAX_RUNS:]
        js.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def record_verdict(self, name, worked, notes=""):
        """The planner's judgement of the most recent run, after looking at the table."""
        _, js = self._paths(name)
        meta = self.meta(name)
        if not meta["runs"]:
            raise primitives.SkillError(f"{name} has not been run on the arm yet")
        meta["runs"][-1].update(worked=bool(worked), notes=notes[:300])
        js.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def summary(self, name):
        m = self.meta(name)
        judged = [r for r in m["runs"] if r.get("worked") is not None]
        good = sum(1 for r in judged if r["worked"])
        args = ", ".join(f"{k}={v}" for k, v in m["params"].items())
        record = f"{good}/{len(judged)} runs judged good" if judged else "never judged"
        return f"{name}({args}): {m['description']} [{record}]"
