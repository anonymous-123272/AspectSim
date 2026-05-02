from __future__ import print_function

import os

baselinePromptsCache = {}


def load_llm_score_prompt(yaml_path):
    """Load YAML with string key llm_score; cache by mtime."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError("Prompt YAML not found: %s" % yaml_path)
    mtime = os.path.getmtime(yaml_path)
    hit = baselinePromptsCache.get(yaml_path)
    if hit is not None and hit[0] == mtime:
        return hit[1]  # file unchanged since last read
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "llm_score" not in data:
        raise ValueError("YAML must define a string key 'llm_score' in %s" % yaml_path)
    if not isinstance(data["llm_score"], str):
        raise ValueError("Key 'llm_score' must be a string in %s" % yaml_path)
    template = data["llm_score"]
    baselinePromptsCache[yaml_path] = (mtime, template)
    return template


def fill_llm_score_template(template, document1, document2, aspect):
    """Replace placeholders; avoids clashes with literal braces in document text."""
    return (
        str(template)
        .replace("__DOC1__", str(document1))
        .replace("__DOC2__", str(document2))
        .replace("__ASPECT__", str(aspect))
    )


def build_user_prompt(yaml_path, document1, document2, aspect):
    template = load_llm_score_prompt(yaml_path)
    return fill_llm_score_template(template, document1, document2, aspect)
