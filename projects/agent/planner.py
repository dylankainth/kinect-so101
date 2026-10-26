"""The task planner: an LLM with the skills as tools. It looks, picks a skill, reads the camera-verified result, and
repeats until it calls `finish`. Only the first tool call of each turn is run: the arm does one thing at a time.
"""
import base64
import logging

import cv2

log = logging.getLogger("agent.planner")

DEFAULT_MODEL = "claude-sonnet-5-5"

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


def image_block(bgr):
    ok, jpg = cv2.imencode(".jpg", bgr)
    data = base64.standard_b64encode(jpg.tobytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def result_content(res):
    content = [{"type": "text", "text": ("OK: " if res.ok else "FAILED: ") + res.text}]
    if res.image is not None:
        content.append(image_block(res.image))
    return content


def run_task(task, box, client, model=DEFAULT_MODEL, max_steps=12):
    """-> (success, summary). box is a SkillBox."""
    tools = box.tool_specs() + [FINISH]
    first = box.call("look", {})
    messages = [{"role": "user", "content": [{"type": "text", "text": f"Task: {task}\n\nCurrent scene:\n{first.text}"}]
                 + ([image_block(first.image)] if first.image is not None else [])}]
    for step in range(max_steps):
        resp = client.messages.create(model=model, max_tokens=2000, system=SYSTEM, tools=tools, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                log.info("planner: %s", b.text.strip())
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            return False, "the planner stopped without calling finish"
        out, ran = [], False
        for c in calls:
            if c.name == "finish":
                return bool(c.input.get("success")), str(c.input.get("summary", ""))
            if ran:
                out.append({"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                            "content": "not run: one skill at a time"})
                continue
            ran = True
            log.info("step %d: %s %s", step + 1, c.name, c.input)
            res = box.call(c.name, dict(c.input))
            log.info("  -> %s %s", "ok" if res.ok else "FAILED", res.text.splitlines()[0])
            out.append({"type": "tool_result", "tool_use_id": c.id, "is_error": not res.ok,
                        "content": result_content(res)})
        messages.append({"role": "user", "content": out})
    return False, f"gave up after {max_steps} steps"
