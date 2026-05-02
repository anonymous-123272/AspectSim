from __future__ import print_function

import os

promptsYamlCache = {}


def load_prompt_templates(yaml_path):
    """Load and cache prompt strings (invalidates on file mtime change)."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load prompts. Install with: pip install pyyaml"
        )
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError("Prompt YAML not found: %s" % yaml_path)
    mtime = os.path.getmtime(yaml_path)
    cached = promptsYamlCache.get(yaml_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with open(yaml_path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    for key in ("single_extraction", "multi_extraction", "summarize"):
        if key not in data or not isinstance(data[key], str):
            raise ValueError(
                "Invalid prompts YAML: missing or non-string key %r in %s"
                % (key, yaml_path)
            )
    promptsYamlCache[yaml_path] = (mtime, data)
    return data


def fill_prompt_template(template, document, aspect):
    """Substitute placeholders (safe for document text containing { braces })."""
    return (
        str(template)
        .replace("__DOCUMENT__", str(document))
        .replace("__ASPECT__", str(aspect))
    )


def ce_prompts_single_multi(document, aspect, prompts_yaml):
    """Return (single_prompt, multi_prompt) for sentence extraction."""
    data = load_prompt_templates(prompts_yaml)
    single = fill_prompt_template(data["single_extraction"], document, aspect)
    multi = fill_prompt_template(data["multi_extraction"], document, aspect)
    return single, multi


def prompt_summarize(document, aspect, prompts_yaml):
    data = load_prompt_templates(prompts_yaml)
    return fill_prompt_template(data["summarize"], document, aspect)

