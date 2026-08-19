"""
Client for Hugging Face's current "Inference Providers" API
(https://router.huggingface.co/v1/chat/completions - OpenAI-compatible).

The old https://api-inference.huggingface.co endpoint used by earlier
versions of this file is DEPRECATED and no longer resolves - this file
uses the new router endpoint.

Needs HUGGINGFACE_LOGIN_TOKEN in .env - create a FINE-GRAINED token at
https://huggingface.co/settings/tokens with the
"Make calls to Inference Providers" permission checked (a plain "Read"
token is NOT enough for this new API).

Two things are used, both through the same chat-completions endpoint:
1. Vision model -> looks at an uploaded leaf photo and identifies the
   crop + disease + treatment directly (not limited to a fixed class list,
   so it can handle crops/diseases beyond any local dataset).
2. Text model -> open-ended farming chatbot.

Both fail gracefully (return None) if the token is missing/invalid, the
request times out, or there's no internet - the app then falls back to
the local model / rule-based chatbot so the site never crashes.
"""
import base64
import json
import re
import requests

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CHAT_MODEL = "openai/gpt-oss-120b:fastest"

DISEASE_PROMPT = (
    "You are an expert plant pathologist. Look at this leaf photo carefully.\n"
    "Identify the crop/plant species and whether it shows a disease.\n"
    "Reply with ONLY a compact JSON object (no markdown, no extra text) in exactly this shape:\n"
    '{"crop": "<crop name>", "healthy": true|false, "disease": "<disease name or empty '
    'string if healthy>", "confidence": <0-100 integer, your best estimate>, '
    '"cause": "<1 sentence cause>", "symptoms": "<1-2 sentence symptom description>", '
    '"treatment": "<2-3 sentence actionable treatment/prevention advice>"}'
)


def _headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _extract_json(text: str):
    """Pulls the first {...} JSON object out of a model response, tolerant of stray text/markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def diagnose_leaf_image(image_bytes: bytes, token: str, timeout: int = 45):
    """
    Returns a dict: {crop, healthy, disease, confidence, cause, symptoms, treatment}
    or None if the call failed for any reason.
    """
    if not token:
        return None
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DISEASE_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 500,
    }

    try:
        resp = requests.post(ROUTER_URL, headers=_headers(token), json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not parsed:
            return None
        parsed.setdefault("confidence", 80)
        return parsed
    except (requests.RequestException, KeyError, IndexError, TypeError):
        return None


def ask_chatbot(message: str, token: str, timeout: int = 30):
    """Returns a free-form answer string, or None if the API call failed."""
    if not token:
        return None

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are ArogyaKrishi's helpful farming assistant for Indian farmers. "
                "Answer clearly and directly in 2-4 sentences about crops, soil, fertilizer, "
                "plant disease, weather or general agriculture questions. If the question is "
                "unrelated to farming, still answer it briefly and helpfully."
            )},
            {"role": "user", "content": message},
        ],
        "stream": False,
        "max_tokens": 300,
    }

    try:
        resp = requests.post(ROUTER_URL, headers=_headers(token), json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, TypeError):
        return None
