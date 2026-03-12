"""
Scene classifier based on CLIP (zero-shot prompts).
Labels: day/night and indoor/outdoor.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict

from PIL import Image

# Reuse the CLIP model loader from knowledge_base to avoid double-loading.
from knowledge_base import get_clip_model


_DAY_PROMPT = "a photo taken during the day"
_NIGHT_PROMPT = "a photo taken at night"
_INDOOR_PROMPT = "an indoor photo"
_OUTDOOR_PROMPT = "an outdoor photo"


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _softmax_pair(a: float, b: float) -> tuple[float, float]:
    # Stable softmax for two numbers
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


@lru_cache(maxsize=1)
def _get_text_embeddings() -> Dict[str, list]:
    model = get_clip_model()
    prompts = [_DAY_PROMPT, _NIGHT_PROMPT, _INDOOR_PROMPT, _OUTDOOR_PROMPT]
    embs = model.encode(prompts)
    return {
        "day": embs[0].tolist(),
        "night": embs[1].tolist(),
        "indoor": embs[2].tolist(),
        "outdoor": embs[3].tolist(),
    }


def classify_scene(image_path: str) -> Dict[str, object]:
    """
    Classify scene into day/night and indoor/outdoor using CLIP zero-shot prompts.

    Returns a dict with:
      - scene: combined label, e.g. "day_outdoor"
      - day_night: "day" | "night"
      - indoor_outdoor: "indoor" | "outdoor"
      - confidence: min(day_prob, indoor_prob) as a conservative score
      - scores: raw cosine scores
      - probs: softmax probs for each pair
    """
    model = get_clip_model()

    img = Image.open(image_path).convert("RGB")
    img_emb = model.encode(img).tolist()

    text_embs = _get_text_embeddings()

    day_score = _cosine_similarity(img_emb, text_embs["day"])
    night_score = _cosine_similarity(img_emb, text_embs["night"])
    indoor_score = _cosine_similarity(img_emb, text_embs["indoor"])
    outdoor_score = _cosine_similarity(img_emb, text_embs["outdoor"])

    day_prob, night_prob = _softmax_pair(day_score, night_score)
    indoor_prob, outdoor_prob = _softmax_pair(indoor_score, outdoor_score)

    day_night = "day" if day_prob >= night_prob else "night"
    indoor_outdoor = "indoor" if indoor_prob >= outdoor_prob else "outdoor"

    scene = f"{day_night}_{indoor_outdoor}"
    confidence = min(
        day_prob if day_night == "day" else night_prob,
        indoor_prob if indoor_outdoor == "indoor" else outdoor_prob,
    )

    return {
        "scene": scene,
        "day_night": day_night,
        "indoor_outdoor": indoor_outdoor,
        "confidence": round(float(confidence), 4),
        "scores": {
            "day": round(float(day_score), 4),
            "night": round(float(night_score), 4),
            "indoor": round(float(indoor_score), 4),
            "outdoor": round(float(outdoor_score), 4),
        },
        "probs": {
            "day": round(float(day_prob), 4),
            "night": round(float(night_prob), 4),
            "indoor": round(float(indoor_prob), 4),
            "outdoor": round(float(outdoor_prob), 4),
        },
    }

