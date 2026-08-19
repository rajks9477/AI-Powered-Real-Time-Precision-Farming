"""
Client for Hugging Face's current "Inference Providers" API
(https://router.huggingface.co/v1/chat/completions - OpenAI-compatible).

Needs HUGGINGFACE_LOGIN_TOKEN in .env - a FINE-GRAINED token with the
"Make calls to Inference Providers" permission checked.
"""
import base64
import json
import re
import requests

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

VISION_MODEL = "Qwen/Qwen2-VL-72B-Instruct"
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
            print("HF disease detection error:", resp.status_code, resp.text[:500])
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not parsed:
            print("HF disease detection: could not parse JSON from:", content[:300])
            return None
        parsed.setdefault("confidence", 80)
        return parsed
    except (requests.RequestException, KeyError, IndexError, TypeError) as e:
        print("HF disease detection exception:", e)
        return None


def ask_chatbot(message: str, token: str, timeout: int = 30):
    if not token:
        print("HF chatbot: no token provided")
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
            print("HF chatbot error:", resp.status_code, resp.text[:500])
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, TypeError) as e:
        print("HF chatbot exception:", e)
        return None