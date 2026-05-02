#!/usr/bin/env python3
"""
LLM-based direct document similarity (0–1) conditioned on aspects.

Reads rows with article1, article2, aspect, doc_n; writes sim_score (+ optional error column).
"""

from __future__ import print_function

import argparse
import os
import sys
import warnings

from tqdm import tqdm
from transformers import set_seed

# This file is scripts/baseline/lbs/… ; repo root is two levels above scripts/
scriptDir = os.path.dirname(os.path.abspath(__file__))
scriptsDir = os.path.dirname(os.path.dirname(scriptDir))
projectRoot = os.path.dirname(scriptsDir)
if scriptsDir not in sys.path:
    sys.path.insert(0, scriptsDir)  # so baseline.* and retrieve_then_embed.* resolve

from retrieve_then_embed.data_io import default_aspect_dataset_path, load_aspect_frame
from retrieve_then_embed.generation import generate_with_vllm_batch
from retrieve_then_embed.modeling import build_llm

from baseline.lbs.parsing import parse_similarity_output
from baseline.lbs.prompts import build_user_prompt

warnings.filterwarnings("ignore")
set_seed(42)

defaultDataset = default_aspect_dataset_path(projectRoot)
defaultPromptsYaml = os.path.join(projectRoot, "prompts", "lbs_prompts.yaml")


# Output filename suffix for this baseline (outputs live in results/baseline/lbs/).
BASELINE_METHOD = "lbs"


def output_path(llm_name):
    out_dir = os.path.join(projectRoot, "results", "baseline", "lbs")
    filename = "%s_%s.xlsx" % (llm_name, BASELINE_METHOD)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def choose_message_list(llm_name, user_prompt):
    # Some chat templates behave better with a single user turn (matches retrieve runner style)
    low = llm_name.lower()
    if any(n in low for n in ("gemma", "deepseek")):
        return [{"role": "user", "content": user_prompt}]
    return [
        {
            "role": "system",
            "content": "You are a precise similarity scoring model for document pairs.",
        },
        {"role": "user", "content": user_prompt},
    ]


def base_columns(frame):
    # Always keep identifiers; add optional label columns when present for downstream correlation
    cols = ["doc_n", "aspect"]
    for name in ("label", "domain", "pair", "aspect_type"):
        if name in frame.columns:
            cols.append(name)
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based direct document similarity on aspects."
    )
    parser.add_argument("--dataset", default=defaultDataset, help="Input label table (.csv or .xlsx)")
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain filter; omit for all rows",
    )
    parser.add_argument("--model_id", required=True, help="HF / vLLM model id")
    parser.add_argument("--llm_name", required=True, help="Short tag for sheet name and filename")
    parser.add_argument("--prompts-yaml", default=defaultPromptsYaml, help="YAML with llm_score template")
    parser.add_argument("--batch-size", type=int, default=1, help="Rows per vLLM generate() call")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override tensor_parallel_size",
    )
    parser.add_argument("--max-model-len", type=int, default=None, help="Override max_model_len")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override gpu_memory_utilization (0–1)",
    )
    parser.add_argument("--start-index", type=int, default=None, help="Inclusive start row index")
    parser.add_argument("--end-index", type=int, default=None, help="Inclusive end row index")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write Excel every N completed rows (plus final)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    frame = load_aspect_frame(args.dataset, args.domain)
    if args.start_index is not None or args.end_index is not None:
        start_index = args.start_index if args.start_index is not None else 0
        end_index = args.end_index if args.end_index is not None else len(frame) - 1
        if start_index < 0 or end_index < start_index or start_index >= len(frame):
            parser.error("Invalid --start-index / --end-index")
        end_index = min(end_index, len(frame) - 1)
        frame = frame.iloc[start_index : end_index + 1].reset_index(drop=True)

    tokenizer, sampling_params, llm = build_llm(
        args.model_id,
        args.llm_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    out_file = output_path(args.llm_name)
    base_cols = base_columns(frame)
    n = len(frame)
    scores = [None] * n  # parsed floats or None on parse failure
    errors = [None] * n  # short reason string from parsing.py, or None when ok

    def flush_up_to(end_exclusive):
        # Persist rows [0, end_exclusive) so checkpoints never require a full re-run
        slice_frame = frame.iloc[:end_exclusive][base_cols].copy()
        slice_frame["sim_score"] = scores[:end_exclusive]
        slice_frame["score_error"] = errors[:end_exclusive]
        slice_frame.to_excel(out_file, index=False, sheet_name=args.llm_name)

    batches = []
    i = 0
    while i < n:
        batches.append(list(range(i, min(i + args.batch_size, n))))
        i += args.batch_size

    pbar = tqdm(batches, desc="LBS", dynamic_ncols=True)
    completed = 0
    for batch_idx in pbar:
        messages_list = []
        for row_i in batch_idx:
            row = frame.iloc[row_i]
            doc1 = row["article1"]
            doc2 = row["article2"]
            if hasattr(doc1, "item"):
                doc1 = doc1.item()  # unwrap numpy / scalar cells from Excel
            if hasattr(doc2, "item"):
                doc2 = doc2.item()
            aspect = row["aspect"]
            user_prompt = build_user_prompt(
                args.prompts_yaml,
                doc1,
                doc2,
                aspect,
            )
            messages_list.append(choose_message_list(args.llm_name, user_prompt))

        raw_list = generate_with_vllm_batch(
            tokenizer,
            llm,
            args.llm_name,
            messages_list,
            sampling_params,
        )

        for row_i, raw in zip(batch_idx, raw_list):
            val, err = parse_similarity_output(raw)
            scores[row_i] = val
            errors[row_i] = err
            completed += 1

        if (
            args.checkpoint_every > 0
            and completed > 0
            and completed % args.checkpoint_every == 0
        ):
            flush_up_to(completed)
        pbar.set_postfix(wrote=completed)

    flush_up_to(n)
    print("Wrote %s" % out_file)


if __name__ == "__main__":
    main()
