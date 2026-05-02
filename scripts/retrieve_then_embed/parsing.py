from __future__ import print_function

import json
import re


def normalize_model_output(text):
    """Strip common wrappers so JSON search runs on cleaner text."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    fence = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE
    )
    if fence:
        cleaned = fence.group(1).strip()
    cleaned = re.sub(
        r"<think>[\s\S]*?</think>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<reasoning>[\s\S]*?</reasoning>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def balanced_delimited_slices(text, open_ch, close_ch):
    """
    Yield balanced outer slices starting at open_ch, respecting double-quoted
    strings and backslashes (valid JSON-style strings).
    """
    length = len(text)
    i = 0
    while i < length:
        if text[i] != open_ch:
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escaped = False
        j = i
        while j < length:
            c = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
                j += 1
                continue
            if c == open_ch:
                depth += 1
                j += 1
                continue
            if c == close_ch:
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    j += 1
                    i = j
                    break
                j += 1
                continue
            j += 1
        else:
            i += 1


def balanced_json_object_strings(text):
    return list(balanced_delimited_slices(text, "{", "}"))


def normalize_yes_no(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    low = text.lower()
    if low in ("yes", "y", "true", "1"):
        return "Yes"
    if low in ("no", "n", "false", "0"):
        return "No"
    if text in ("Yes", "No"):
        return text
    return None


def coerce_narrative_1(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def parse_aspect_response_dict(output_text):
    """
    Extract the intended {"discusses_aspect", "narrative_1"} object from raw
    model text.
    """
    cleaned = normalize_model_output(output_text)
    if not cleaned:
        return None
    candidates = balanced_json_object_strings(cleaned)
    for obj in reversed(candidates):
        try:
            data = json.loads(obj)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        normalized = normalize_yes_no(data.get("discusses_aspect"))
        if normalized is None:
            continue
        result = dict(data)
        result["discusses_aspect"] = normalized
        result["narrative_1"] = coerce_narrative_1(data.get("narrative_1"))
        return result
    return None


def extract_narrative_loose(output_text):
    """
    If full JSON parse fails, find "narrative_1" key and slice its array value
    using bracket balancing.
    """
    cleaned = normalize_model_output(output_text)
    if not cleaned:
        return None
    match = re.search(r'(?i)"narrative_1"\s*:\s*', cleaned)
    if not match:
        match = re.search(r"(?i)'narrative_1'\s*:\s*", cleaned)
    if not match:
        return None
    pos = match.end()
    while pos < len(cleaned) and cleaned[pos] in " \t\n\r":
        pos += 1
    if pos >= len(cleaned) or cleaned[pos] != "[":
        return None
    for arr in balanced_delimited_slices(cleaned[pos:], "[", "]"):
        try:
            return json.loads(arr)
        except (ValueError, TypeError, json.JSONDecodeError):
            return arr
    return None


def extract_relevant_json(output_text):
    """Wraps parse_aspect_response_dict."""
    return parse_aspect_response_dict(output_text)


def extract_first_dictionary(output_text):
    """First balanced {...} slice after normalization, or None."""
    cleaned = normalize_model_output(output_text)
    objects = balanced_json_object_strings(cleaned)
    return objects[0] if objects else None


def extract_list(json_str):
    """Fallback parser: first [...] slice with mild trailing fix."""
    match = re.search(r"\[", json_str)
    if match:
        start_index = match.start()
        closing_index = json_str.find("]", start_index)
        if closing_index != -1:
            return json_str[start_index : closing_index + 1]
        list_part = json_str[start_index:]
        if list_part.count('"') % 2 != 0:
            list_part += '"'
        list_part += "]"
        return list_part
    return json_str


def extract_narrative_for_storage(output_text):
    """Value to store in pair1/pair2: narrative_1 list or best fallback."""
    data = parse_aspect_response_dict(output_text)
    if data is not None:
        return data.get("narrative_1", [])
    loose = extract_narrative_loose(output_text)
    if loose is not None:
        return loose
    raw = extract_list(output_text)
    try:
        return json.loads(raw)
    except Exception:
        return raw

