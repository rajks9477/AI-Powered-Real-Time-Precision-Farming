"""
Best-effort SOIL TYPE estimator from a photo.

Important honesty note: there is no reliable free public AI model that
identifies Indian soil types from a photo the way plant-disease models exist.
This module uses colour + brightness analysis (a real, working heuristic -
not a placeholder) to bucket the photo into one of the major soil categories
used in Indian agriculture, then maps that to crop suggestions. Treat the
result as a helpful starting point, not a lab soil test.
"""
from PIL import Image
import io
import colorsys

# Reference: ICAR soil classification broad categories
SOIL_PROFILES = {
    "black": {
        "label": "Black soil (Regur)",
        "why": "Very dark, high clay content - retains moisture well, rich in iron, magnesium and lime.",
        "good_for": ["cotton", "jowar (sorghum)", "sugarcane", "wheat", "sunflower", "citrus"],
        "care": "Drains slowly - avoid overwatering. Deep ploughing after monsoon improves aeration.",
    },
    "red": {
        "label": "Red / laterite soil",
        "why": "Reddish colour from iron oxide, generally low in nitrogen, phosphorus and organic matter.",
        "good_for": ["groundnut", "millets (ragi/bajra)", "potato", "pulses", "cashew", "tea"],
        "care": "Add organic compost and nitrogen-fixing legumes to improve fertility.",
    },
    "alluvial": {
        "label": "Alluvial soil",
        "why": "Light brown/grey, fertile river-deposited soil, good for most staple crops.",
        "good_for": ["rice", "wheat", "sugarcane", "maize", "pulses", "vegetables"],
        "care": "Generally fertile - focus on balanced NPK and regular crop rotation.",
    },
    "sandy": {
        "label": "Sandy / arid soil",
        "why": "Light, pale, coarse-looking with low water retention.",
        "good_for": ["bajra (pearl millet)", "moth beans", "watermelon", "muskmelon", "groundnut"],
        "care": "Add organic matter to improve water retention; use drip irrigation.",
    },
    "loamy_dark": {
        "label": "Loamy / organic-rich soil",
        "why": "Dark brown, crumbly-looking, generally rich in organic matter.",
        "good_for": ["vegetables", "banana", "maize", "pulses", "most horticultural crops"],
        "care": "Well-suited to most crops - maintain organic matter with compost/green manure.",
    },
}


def _avg_hsv(image_bytes: bytes, sample=40):
    """Downsamples the image and returns average (hue, saturation, value) in 0-1 range."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((sample, sample))
    pixels = list(img.getdata())
    h_sum = s_sum = v_sum = 0.0
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h_sum += h
        s_sum += s
        v_sum += v
    n = len(pixels)
    return h_sum / n, s_sum / n, v_sum / n


def estimate_soil_type(image_bytes: bytes):
    """
    Rule-of-thumb bucketing by brightness (value), saturation and hue,
    tuned against typical photos of Indian soil types.
    Returns a dict with the matched profile plus the raw HSV reading (for transparency).
    """
    h, s, v = _avg_hsv(image_bytes)

    if v < 0.28:
        key = "black"
    elif v > 0.70 and s < 0.25:
        key = "sandy"
    elif 0.02 <= h <= 0.08 and s > 0.35:
        key = "red"
    elif v < 0.45 and s > 0.25:
        key = "loamy_dark"
    else:
        key = "alluvial"

    profile = dict(SOIL_PROFILES[key])
    profile["confidence_note"] = (
        "Estimated from photo colour/brightness only - for a precise reading, "
        "get a lab soil test from your nearest Krishi Vigyan Kendra."
    )
    profile["raw_reading"] = {"hue": round(h, 3), "saturation": round(s, 3), "brightness": round(v, 3)}
    return profile
