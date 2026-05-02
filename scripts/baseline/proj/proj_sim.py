#!/usr/bin/env python3
"""
Projection-based similarity on the consolidated label table.

For each row, embed article1 (A), article2 (B), and aspect (C) with SentenceTransformer
(L2-normalized). Per model, score = dot(A,C) - dot(B,C) (same geometry as wiki_proj_sim.py).

Model lists match retrieve_then_embed/embed.py: --preset first|second|all or --models mpnet,e5,...

Default workbook: results/baseline/proj/proj_sim.xlsx (sheet proj_sim). Re-running with a
different preset adds new embedding columns to the same file; columns that already exist
are skipped. Use the same row set (dataset / domain / indices) for sensible merges.

Scores are stored with 4 decimal places.
"""

from __future__ import print_function

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import set_seed

scriptDir = os.path.dirname(os.path.abspath(__file__))
scriptsDir = os.path.dirname(os.path.dirname(scriptDir))
projectRoot = os.path.dirname(scriptsDir)
if scriptsDir not in sys.path:
    sys.path.insert(0, scriptsDir)

from retrieve_then_embed.data_io import default_aspect_dataset_path, load_aspect_frame
from retrieve_then_embed.embed import MODEL_REGISTRY, parse_models_arg

warnings.filterwarnings("ignore")
set_seed(42)

defaultDataset = default_aspect_dataset_path(projectRoot)

PROJ_OUT_DIR = os.path.join(projectRoot, "results", "baseline", "proj")
DEFAULT_WORKBOOK = "proj_sim.xlsx"
SHEET_NAME = "proj_sim"
SCORE_DECIMALS = 4


def resolve_output_path(args):
    os.makedirs(PROJ_OUT_DIR, exist_ok=True)
    if args.output:
        return os.path.abspath(args.output)
    return os.path.join(PROJ_OUT_DIR, DEFAULT_WORKBOOK)


def cell_to_str(value):
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def base_columns(frame):
    cols = ["doc_n", "aspect"]
    for name in ("label", "domain", "pair", "aspect_type"):
        if name in frame.columns:
            cols.append(name)
    return cols


def merge_key_columns(frame):
    """Row identity for merging incremental runs (exclude label)."""
    order = ("domain", "pair", "doc_n", "aspect", "aspect_type")
    return [c for c in order if c in frame.columns]


def pick_sheet_name(path):
    xl = pd.ExcelFile(path)
    if SHEET_NAME in xl.sheet_names:
        return SHEET_NAME
    return xl.sheet_names[0]


def round_registry_score_columns(frame):
    """Round embedding score columns to SCORE_DECIMALS."""
    out = frame.copy()
    for c in MODEL_REGISTRY:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(SCORE_DECIMALS)
    return out


def scores_rounded(raw_scores):
    out = np.array(raw_scores, dtype=np.float64)
    fin = np.isfinite(out)
    out[fin] = np.round(out[fin], SCORE_DECIMALS)
    return out


def compute_projection_scores(model, doc1_list, doc2_list, aspect_list, encode_batch_size, row_chunk):
    """Batched projection difference dot(A,C)-dot(B,C) for unit-normalized embeddings."""
    n = len(doc1_list)
    scores = np.full(n, np.nan, dtype=np.float64)
    for start in tqdm(range(0, n, row_chunk), desc="Chunks", dynamic_ncols=True, leave=False):
        end = min(start + row_chunk, n)
        texts = []
        idx_map = []
        for i in range(start, end):
            a, b, c = doc1_list[i], doc2_list[i], aspect_list[i]
            if not a or not b or not c:
                continue
            texts.extend([a, b, c])
            idx_map.append(i)
        if not texts:
            continue
        emb = model.encode(
            texts,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        k = len(idx_map)
        emb = emb.reshape(k, 3, -1)
        A, B, C = emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]
        block_scores = np.sum(A * C, axis=1) - np.sum(B * C, axis=1)
        for i, val in zip(idx_map, block_scores):
            scores[i] = float(val)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Projection similarity baseline (embedding).")
    parser.add_argument("--dataset", default=defaultDataset, help="Input label table (.csv or .xlsx)")
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain filter; omit for all rows",
    )
    parser.add_argument(
        "--preset",
        choices=["first", "second", "all"],
        default="first",
        help="Model bundle when --models is omitted (same as embed.py)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model aliases overriding --preset (e.g. mpnet,e5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .xlsx (default: results/baseline/proj/proj_sim.xlsx; merges new columns into existing file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoder batch size passed to SentenceTransformer.encode",
    )
    parser.add_argument(
        "--row-chunk",
        type=int,
        default=512,
        help="Rows per chunk (3 texts per valid row encoded together per chunk)",
    )
    parser.add_argument("--start-index", type=int, default=None, help="Inclusive start row index")
    parser.add_argument("--end-index", type=int, default=None, help="Inclusive end row index")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.row_chunk < 1:
        parser.error("--row-chunk must be >= 1")

    model_names = parse_models_arg(args.models, args.preset)

    frame = load_aspect_frame(args.dataset, args.domain)
    if args.start_index is not None or args.end_index is not None:
        start_i = args.start_index if args.start_index is not None else 0
        end_i = args.end_index if args.end_index is not None else len(frame) - 1
        if start_i < 0 or end_i < start_i or start_i >= len(frame):
            parser.error("Invalid --start-index / --end-index")
        end_i = min(end_i, len(frame) - 1)
        frame = frame.iloc[start_i : end_i + 1].reset_index(drop=True)

    keys = merge_key_columns(frame)
    if len(keys) < 2 or "doc_n" not in keys or "aspect" not in keys:
        parser.error("Need doc_n and aspect (and merge keys) in frame for incremental proj_sim.xlsx")

    doc1_list = [cell_to_str(x) for x in frame["article1"].tolist()]
    doc2_list = [cell_to_str(x) for x in frame["article2"].tolist()]
    aspect_list = [cell_to_str(x) for x in frame["aspect"].tolist()]

    out_file = resolve_output_path(args)
    base = frame[base_columns(frame)].copy()

    existing = None
    if os.path.isfile(out_file):
        sheet_read = pick_sheet_name(out_file)
        existing = pd.read_excel(out_file, sheet_name=sheet_read)
        print("Loaded existing %s (%d rows, sheet=%s)" % (out_file, len(existing), sheet_read))

    models_to_run = []
    for m in model_names:
        if existing is not None and m in existing.columns:
            print("Skip %s (already in workbook)" % m)
            continue
        models_to_run.append(m)

    if not models_to_run:
        print("No new models to compute; exiting.")
        return

    print("Models to run:", ", ".join(models_to_run))
    print("Output:", out_file)

    out = base.copy()
    for model_key in tqdm(models_to_run, desc="Models", dynamic_ncols=True):
        model_id = MODEL_REGISTRY[model_key]
        trust = model_key in ("bilingual", "jinja")
        print("Loading %s -> %s" % (model_key, model_id))
        model = SentenceTransformer(model_id, trust_remote_code=trust) if trust else SentenceTransformer(
            model_id
        )

        raw = compute_projection_scores(
            model,
            doc1_list,
            doc2_list,
            aspect_list,
            encode_batch_size=args.batch_size,
            row_chunk=args.row_chunk,
        )
        out[model_key] = scores_rounded(raw)
        del model

    sheet_write = SHEET_NAME[:31]
    if existing is None:
        final = round_registry_score_columns(out)
        final.to_excel(out_file, index=False, sheet_name=sheet_write)
        print(
            "Wrote %s (%d rows, %d new score columns)"
            % (out_file, len(final), len(models_to_run))
        )
        return

    keys_common = [k for k in keys if k in existing.columns and k in out.columns]
    if len(keys_common) < len(keys):
        print(
            "Warning: merge keys restricted to intersection: %s (existing columns: %s)"
            % (keys_common, list(existing.columns)),
            file=sys.stderr,
        )
    if not keys_common:
        print("Cannot merge: no common key columns between existing file and this run.", file=sys.stderr)
        sys.exit(1)

    add_block = out[keys_common + models_to_run].copy()
    final = existing.merge(add_block, on=keys_common, how="outer")
    final = round_registry_score_columns(final)
    final.to_excel(out_file, index=False, sheet_name=sheet_write)
    print(
        "Updated %s (%d rows, added columns: %s)"
        % (out_file, len(final), ", ".join(models_to_run))
    )


if __name__ == "__main__":
    main()
