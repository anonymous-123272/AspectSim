from __future__ import print_function

import json
import re


def extract_json_object(text):
    """First JSON object in text with discusses_aspect and score, or None."""
    if not text or not isinstance(text, str):
        return None
    # Try slightly balanced `{ ... }` first (one level of nesting)
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        chunk = match.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "score" in obj:
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    # Broader non-greedy span; may work when nesting regex above is too strict
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        chunk = match.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "score" in obj:
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def parse_similarity_output(raw_text):
    """
    Return (score_float_or_none, error_message_or_none).
    score is clamped to [0, 1] when valid.
    """
    if raw_text is None:
        return None, "empty_output"
    obj = extract_json_object(raw_text.strip())
    if not obj:
        return None, "json_parse_failed"
    try:
        if str(obj.get("discusses_aspect", "")).strip().lower() == "no":
            return 0.0, None  # prompt contract: no aspect discussion => score 0
        s = obj["score"]
        if isinstance(s, (int, float)):
            v = float(s)
        else:
            v = float(str(s).strip())
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return round(v, 4), None
    except (TypeError, ValueError, KeyError):
        return None, "invalid_score_field"
