#!/usr/bin/env python3
"""
Compute embedding similarity for response files and write back in place.

Reads an Excel response file (typically from results/retrieve-then-embed/<method>/...), parses
`pairs` (or falls back to `pair1`/`pair2`), computes cosine similarities using
SentenceTransformer models, and stores scores as new columns in the same file.
"""

from __future__ import print_function

import argparse
import ast
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_REGISTRY = {
    "mpnet": "all-mpnet-base-v2",
    "bilingual": "Lajavaness/bilingual-embedding-large",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    "jinja": "jinaai/jina-embeddings-v3",
    "mistral": "Salesforce/SFR-Embedding-Mistral",
    "ling": "Linq-AI-Research/Linq-Embed-Mistral",
    "qwen3": "Qwen/Qwen3-Embedding-8B",
    "e5": "intfloat/multilingual-e5-large",
    "gemma": "google/embeddinggemma-300m",
}

PRESETS = {
    "first": ["mpnet", "bilingual", "mxbai", "jinja", "mistral", "ling"],
    "second": ["qwen3", "e5", "gemma"],
    "all": [
        "mpnet",
        "bilingual",
        "mxbai",
        "jinja",
        "mistral",
        "ling",
        "qwen3",
        "e5",
        "gemma",
    ],
}


def flatten_text_fragments(value):
    """Recursively flatten nested list/stringified-list values into text fragments."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(flatten_text_fragments(item))
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if (text.startswith("[") and text.endswith("]")) or (
            text.startswith("(") and text.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return flatten_text_fragments(parsed)
            except Exception:
                pass
        return [text]
    return [str(value)]


def merge_to_string(value):
    return ", ".join(flatten_text_fragments(value))


def parse_pair_cell(value):
    """
    Parse one row's `pairs` cell into (left_text, right_text).
    Returns empty strings when unavailable/unparseable.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "", ""

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return merge_to_string(value[0]), merge_to_string(value[1])

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "", ""
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return merge_to_string(parsed[0]), merge_to_string(parsed[1])
        except Exception:
            return "", ""

    return "", ""


def build_pair_lists(frame):
    """
    Build aligned pair-text lists (left/right) for every row.
    Priority:
      1) `pairs` column
      2) `pair1` and `pair2` columns
    """
    left = []
    right = []

    if "pairs" in frame.columns:
        for value in frame["pairs"].tolist():
            a_text, b_text = parse_pair_cell(value)
            left.append(a_text)
            right.append(b_text)
        return left, right

    if "pair1" in frame.columns and "pair2" in frame.columns:
        for a_val, b_val in zip(frame["pair1"].tolist(), frame["pair2"].tolist()):
            left.append(merge_to_string(a_val))
            right.append(merge_to_string(b_val))
        return left, right

    raise ValueError("Input file must contain `pairs` or both `pair1` and `pair2` columns.")


def cosine_diag(a_embeddings, b_embeddings):
    """Row-wise cosine similarity for already-normalized embeddings."""
    return np.sum(a_embeddings * b_embeddings, axis=1)


def compute_similarity_column(model, left_texts, right_texts, batch_size):
    """
    Compute similarity per row. Empty side -> score 0.
    Returns list[float].
    """
    scores = np.zeros(len(left_texts), dtype=np.float32)
    valid_indices = [
        i for i, (a, b) in enumerate(zip(left_texts, right_texts)) if a and b
    ]
    if not valid_indices:
        return scores.tolist()

    valid_left = [left_texts[i] for i in valid_indices]
    valid_right = [right_texts[i] for i in valid_indices]

    left_emb = model.encode(
        valid_left,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    right_emb = model.encode(
        valid_right,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    sims = cosine_diag(left_emb, right_emb)
    for idx, val in zip(valid_indices, sims):
        scores[idx] = float(np.round(val, 4))
    return scores.tolist()


def parse_models_arg(models_arg, preset):
    if models_arg:
        names = [x.strip() for x in models_arg.split(",") if x.strip()]
    else:
        names = PRESETS[preset]
    unknown = [name for name in names if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError("Unknown model aliases: %s" % ", ".join(unknown))
    return names


def main():
    parser = argparse.ArgumentParser(description="Compute embedding similarities in-place.")
    parser.add_argument("--file", required=True, help="Path to response xlsx file")
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Excel sheet name (default: first sheet)",
    )
    parser.add_argument(
        "--preset",
        choices=["first", "second", "all"],
        default="first",
        help="Model preset when --models is not provided",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model aliases overriding preset (e.g. mpnet,e5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding encode batch size",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File not found: %s" % file_path)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    workbook = pd.ExcelFile(file_path)
    sheet_name = args.sheet_name or workbook.sheet_names[0]
    frame = pd.read_excel(file_path, sheet_name=sheet_name)

    left_texts, right_texts = build_pair_lists(frame)
    model_names = parse_models_arg(args.models, args.preset)

    print("File:", file_path)
    print("Sheet:", sheet_name)
    print("Rows:", len(frame))
    print("Models:", ", ".join(model_names))

    for model_name in tqdm(model_names, desc="Models", dynamic_ncols=True):
        model_id = MODEL_REGISTRY[model_name]
        print("Loading model:", model_name, "->", model_id)
        if model_name in ("bilingual", "jinja"):
            model = SentenceTransformer(model_id, trust_remote_code=True)
        else:
            model = SentenceTransformer(model_id)
        col_name = model_name
        frame[col_name] = compute_similarity_column(
            model, left_texts, right_texts, batch_size=args.batch_size
        )
        del model

    with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)

    print("Saved similarities to:", file_path)


if __name__ == "__main__":
    main()

