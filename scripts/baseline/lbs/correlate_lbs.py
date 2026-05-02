#!/usr/bin/env python3
"""
Spearman correlation between LBS outputs (default column sim_score) and human labels.
python scripts/baseline/lbs/correlate_lbs.py --file results/baseline/lbs/qwen2.5-14b_lbs.xlsx
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

scriptDir = os.path.dirname(os.path.abspath(__file__))
scriptsDir = os.path.dirname(os.path.dirname(scriptDir))
if scriptsDir not in sys.path:
    sys.path.insert(0, scriptsDir)

from retrieve_then_embed.correlate import encode_labels, safe_corr


def main():
    parser = argparse.ArgumentParser(description="Spearman: LBS sim_score vs labels (one xlsx).")
    parser.add_argument("--file", required=True, help="Path to LBS results Excel (e.g. *_lbs.xlsx)")
    parser.add_argument("--sheet-name", default=None, help="Sheet name (default: first sheet)")
    parser.add_argument("--label-column", default="label", help="Human label column")
    parser.add_argument(
        "--score-column",
        default="sim_score",
        help="LBS prediction column (default: sim_score)",
    )
    parser.add_argument("--output", default=None, help="Optional CSV path for one-row result")
    args = parser.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        print("File not found: %s" % path, file=sys.stderr)
        sys.exit(1)

    wb = pd.ExcelFile(path)
    sheet = args.sheet_name or wb.sheet_names[0]
    frame = pd.read_excel(path, sheet_name=sheet)

    if args.label_column not in frame.columns:
        print("Missing label column %r" % args.label_column, file=sys.stderr)
        sys.exit(1)
    if args.score_column not in frame.columns:
        print("Missing score column %r" % args.score_column, file=sys.stderr)
        sys.exit(1)

    enc = encode_labels(frame[args.label_column])
    frame = frame.copy()
    frame["enc_label"] = enc

    unmapped = frame[args.label_column].notna() & frame["enc_label"].isna()
    if unmapped.any():
        bad = frame.loc[unmapped, args.label_column].dropna().unique()[:10]
        print(
            "Warning: %d labels could not be encoded (sample): %s"
            % (unmapped.sum(), list(bad)),
            file=sys.stderr,
        )

    s = pd.to_numeric(frame[args.score_column], errors="coerce")
    spr, spr_p = safe_corr(s, frame["enc_label"], spearmanr)
    n_valid = int(
        np.sum(np.isfinite(s.to_numpy(dtype=float)) & np.isfinite(frame["enc_label"].to_numpy(dtype=float)))
    )

    row = {
        "file": path,
        "sheet": sheet,
        "score_column": args.score_column,
        "label_column": args.label_column,
        "spearman": round(spr, 4) if np.isfinite(spr) else spr,
        "spearman_p": spr_p,
        "n_valid": n_valid,
    }

    out = pd.DataFrame([row])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(out.to_string(index=False))

    if args.output:
        out.to_csv(os.path.abspath(args.output), index=False)
        print("Wrote:", args.output)


if __name__ == "__main__":
    main()
