#!/usr/bin/env python3
"""
vLLM runner over the consolidated aspect-similarity label table.

  --method single      Single-sentence extraction (CE-style prompt).
  --method multi       Multi-sentence extraction (CE-style prompt).
  --method summarize   Aspect summarization prompt.

Rows use the full table by default; optional --domain can filter to one domain.
The --method flag only selects the prompt style; it does not filter rows.

Outputs:
  <project>/results/retrieve-then-embed/<method>/<llm_name>_<method>_<suffix>.xlsx
  Sheet name = llm_name. Columns doc_n, aspect, pair1 / pair2.

Optional row slicing:
  Use --start-index / --end-index (inclusive) to run only a subset.
"""

from __future__ import print_function

import argparse
import os
import sys
import warnings

packageDir = os.path.dirname(os.path.abspath(__file__))
scriptsDir = os.path.dirname(packageDir)
if scriptsDir not in sys.path:
    sys.path.insert(0, scriptsDir)

from transformers import set_seed

from retrieve_then_embed.data_io import default_aspect_dataset_path, load_aspect_frame, output_path
from retrieve_then_embed.modeling import build_llm
from retrieve_then_embed.runner import response_generation

warnings.filterwarnings("ignore")
set_seed(42)

projectRoot = os.path.abspath(os.path.join(scriptsDir, os.pardir))
defaultDataset = default_aspect_dataset_path(projectRoot)
resultsRoot = os.path.join(projectRoot, "results")
defaultPromptsYaml = os.path.join(projectRoot, "prompts", "aspectsim_prompts.yaml")


def build_parser():
    parser = argparse.ArgumentParser(description="vLLM extraction/summarization over aspect-similarity labels")
    parser.add_argument("--dataset", default=defaultDataset, help="Input table (.csv or .xlsx)")
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain filter (wiki, allside, hotel, mslr, peer). Use all rows if omitted.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["single", "multi", "summarize"],
        help="single / multi = extraction; summarize = aspect summary",
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--llm_name", required=True)
    parser.add_argument("--column_name", required=True, choices=["article1", "article2"])
    parser.add_argument("--pair_name", required=True, choices=["pair1", "pair2"])
    parser.add_argument(
        "--prompts-yaml",
        default=defaultPromptsYaml,
        help="YAML with keys single_extraction, multi_extraction, summarize",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override tensor_parallel_size (e.g. 2 for 70B+)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override max_model_len context window",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Override vLLM gpu_memory_utilization (0-1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of rows per vLLM generate() call (default 1)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional inclusive start index within filtered dataframe",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Optional inclusive end index within filtered dataframe",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    frame = load_aspect_frame(args.dataset, args.domain)
    if args.start_index is not None or args.end_index is not None:
        start_index = args.start_index if args.start_index is not None else 0
        end_index = args.end_index if args.end_index is not None else len(frame) - 1
        if start_index < 0:
            parser.error("--start-index must be >= 0")
        if end_index < start_index:
            parser.error("--end-index must be >= --start-index")
        if start_index >= len(frame):
            parser.error("--start-index is out of range for filtered dataframe")
        end_index = min(end_index, len(frame) - 1)
        print("Using slice start=%d end=%d (inclusive)" % (start_index, end_index))
        frame = frame.iloc[start_index : end_index + 1].reset_index(drop=True)

    tokenizer, sampling_params, llm = build_llm(
        args.model_id,
        args.llm_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    llm_info = {
        "tokenizer": tokenizer,
        "llm": llm,
        "sampling_params": sampling_params,
        "llm_name": args.llm_name,
    }

    output_file = output_path(results_root=resultsRoot, llm_name=args.llm_name, method=args.method)
    out = response_generation(
        aspect_data=frame,
        column_name=args.column_name,
        pair_name=args.pair_name,
        method=args.method,
        llm_info=llm_info,
        prompts_yaml=args.prompts_yaml,
        output_file=output_file,
        batch_size=args.batch_size,
    )
    print("Done. Wrote %s" % out)


if __name__ == "__main__":
    main()

