"""The model connection: OpenRouter's OpenAI-compatible API. Set OPENROUTER_API_KEY; OPENROUTER_MODEL picks the model."""
import base64
import os

import cv2

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5")


def make_client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY (https://openrouter.ai/keys)")
    return OpenAI(base_url=BASE_URL, api_key=key)


def data_url(bgr):
    ok, jpg = cv2.imencode(".jpg", bgr)
    return "data:image/jpeg;base64," + base64.standard_b64encode(jpg.tobytes()).decode()
