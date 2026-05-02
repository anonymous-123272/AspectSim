#!/usr/bin/env python3
"""
Spearman correlation between proj_sim outputs (embedding score columns) and human labels.

For each numeric score column (all embedding models in the sheet, not only a fixed list),
computes Spearman vs encoded labels. Known model aliases are correlated first (mpnet, …);
any other numeric columns follow.
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

from retrieve_then_embed.correlate import (
    KNOWN_SCORE_COLUMNS,
    RESERVED_COLUMNS,
    encode_labels,
    safe_corr,
)


def infer_proj_score_columns(frame, label_column, explicit=None):
    """
    Every numeric column that is not metadata is treated as an embedding score column.

    Columns listed in KNOWN_SCORE_COLUMNS (that are present) are listed first, in that
    order; any other numeric score columns follow, sorted alphabetically.
    """
    if explicit:
        cols = [c.strip() for c in explicit.split(",") if c.strip()]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError("Requested score columns not in file: %s" % ", ".join(missing))
        return cols

    numeric_scores = []
    for c in frame.columns:
        if c in RESERVED_COLUMNS or c.startswith("Unnamed") or c == label_column:
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            numeric_scores.append(c)
    if not numeric_scores:
        raise ValueError(
            "No numeric score columns found. Pass --score-columns or check the workbook."
        )
    known_first = [c for c in KNOWN_SCORE_COLUMNS if c in numeric_scores]
    rest = sorted(c for c in numeric_scores if c not in known_first)
    return known_first + rest


def main():
    parser = argparse.ArgumentParser(
        description="Spearman: proj baseline columns vs labels (one xlsx, e.g. proj_sim.xlsx)."
    )
    parser.add_argument("--file", required=True, help="Path to proj results Excel (*_proj.xlsx)")
    parser.add_argument("--sheet-name", default=None, help="Sheet name (default: first sheet)")
    parser.add_argument("--label-column", default="label", help="Human label column")
    parser.add_argument(
        "--score-columns",
        default=None,
        help="Comma-separated score columns (default: known names present in file, else numeric cols)",
    )
    parser.add_argument("--output", default=None, help="Optional CSV path for correlation table")
    args = parser.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        print("File not found: %s" % path, file=sys.stderr)
        sys.exit(1)

    wb = pd.ExcelFile(path)
    sheet = args.sheet_name or wb.sheet_names[0]
    frame = pd.read_excel(path, sheet_name=sheet)

    if args.label_column not in frame.columns:
        print(
            "Missing required column %r. Columns: %s"
            % (args.label_column, ", ".join(map(str, frame.columns))),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        score_cols = infer_proj_score_columns(frame, args.label_column, args.score_columns)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)

    enc = encode_labels(frame[args.label_column])
    frame = frame.copy()
    frame["enc_label"] = enc

    unmapped = frame[args.label_column].notna() & frame["enc_label"].isna()
    if unmapped.any():
        bad = frame.loc[unmapped, args.label_column].dropna().unique()[:10]
        print(
            "Warning: %d rows have labels that could not be encoded (sample): %s"
            % (unmapped.sum(), list(bad)),
            file=sys.stderr,
        )

    print("File:", path)
    print("Sheet:", sheet)
    print("Rows:", len(frame))
    print("Label column: %s" % args.label_column)
    print("Score columns:", ", ".join(score_cols))
    print()

    rows = []
    for col in score_cols:
        s = pd.to_numeric(frame[col], errors="coerce")
        spr, spr_p = safe_corr(s, frame["enc_label"], spearmanr)
        rows.append(
            {
                "column": col,
                "spearman": round(spr, 4) if np.isfinite(spr) else spr,
                "spearman_p": spr_p,
                "n_valid": int(
                    np.sum(
                        np.isfinite(s.to_numpy(dtype=float))
                        & np.isfinite(frame["enc_label"].to_numpy(dtype=float))
                    )
                ),
            }
        )

    out = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(out.to_string(index=False))

    if args.output:
        out.to_csv(os.path.abspath(args.output), index=False)
        print("\nWrote:", args.output)


if __name__ == "__main__":
    main()
