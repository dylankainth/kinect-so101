"""The task planner: an LLM with the skills as tools. It looks, picks a skill, reads the camera-verified result, and
repeats until it calls `finish`. Only the first tool call of each turn is run: the arm does one thing at a time.
"""
import json
import logging

from projects.agent.llm import DEFAULT_MODEL, data_url

log = logging.getLogger("agent.planner")


SYSTEM = """You control a small 5-joint robot arm with a two-jaw gripper (SO-101) that works on a table seen by an overhead
Kinect camera. You are given a task in plain language. Achieve it by calling the arm's skills.

Rules:
- Robot frame: x forward from the arm base, y to the left, metres. The arm reaches roughly x 0.12-0.30 m and |y| < 0.20 m.
- Object ids change on every look; always use ids from the most recent list.
- One skill call per turn. Read the result: it is checked by the camera. If a move failed, look again, then retry
  (a different approach is fine) at most twice before giving up on that object.
- The gripper can only grasp objects up to 5 cm across and 15 cm tall, and can only pick up and put down. If the
  task needs something the skills cannot do (flipping, pushing, pouring), say so with `finish` rather than trying
  something unsafe.
- Never reposition things the task does not mention. Call `finish` as soon as the task is done or cannot be done."""

FINISH = dict(
    name="finish",
    description="End the task. success=true only if the last camera check confirms the task is done.",
    input_schema=dict(type="object", properties=dict(success=dict(type="boolean"), summary=dict(type="string")),
                      required=["success", "summary"]))


def tool(spec):
    return {"type": "function", "function": {"name": spec["name"], "description": spec["description"],
                                             "parameters": spec["input_schema"]}}


def image_part(bgr):
    return {"type": "image_url", "image_url": {"url": data_url(bgr)}}


def parse_args(raw):
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def run_task(task, box, client, model=DEFAULT_MODEL, max_steps=12):
    """-> (success, summary). box is a SkillBox; client is an OpenAI-style client (OpenRouter)."""
    tools = [tool(s) for s in box.tool_specs() + [FINISH]]
    first = box.call("look", {})
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": f"Task: {task}\n\nCurrent scene:\n{first.text}"}]
                 + ([image_part(first.image)] if first.image is not None else [])}]
    for step in range(max_steps):
        resp = client.chat.completions.create(model=model, max_tokens=2000, tools=tools, messages=messages)
        msg = resp.choices[0].message
        calls = list(msg.tool_calls or [])
        messages.append({"role": "assistant", "content": msg.content or "",
                         **({"tool_calls": [{"id": c.id, "type": "function", "function": {
                             "name": c.function.name, "arguments": c.function.arguments}} for c in calls]} if calls else {})})
        if msg.content and msg.content.strip():
            log.info("planner: %s", msg.content.strip())
        if not calls:
            return False, "the planner stopped without calling finish"
        ran, image = False, None
        for c in calls:
            args = parse_args(c.function.arguments)
            if c.function.name == "finish" and args is not None:
                return bool(args.get("success")), str(args.get("summary", ""))
            if args is None:
                text = "FAILED: the arguments were not valid JSON"
            elif ran:
                text = "not run: one skill at a time"
            else:
                ran = True
                log.info("step %d: %s %s", step + 1, c.function.name, args)
                res = box.call(c.function.name, args)
                log.info("  -> %s %s", "ok" if res.ok else "FAILED", res.text.splitlines()[0])
                text, image = ("OK: " if res.ok else "FAILED: ") + res.text, res.image
            messages.append({"role": "tool", "tool_call_id": c.id, "content": text})
        if image is not None:   # a tool message carries only text: the picture follows as a user message
            messages.append({"role": "user", "content": [{"type": "text", "text": "Camera view after that:"},
                                                         image_part(image)]})
    return False, f"gave up after {max_steps} steps"
