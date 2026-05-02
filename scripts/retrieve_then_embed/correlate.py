#!/usr/bin/env python3
"""Spearman correlation between similarity-score columns and human labels in one Excel file."""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# String labels -> ordinal codes 0..3
LABEL_MAP = {
    "Not Found": 0,
    "Marginally Similar": 1,
    "Somewhat Similar": 2,
    "Highly Similar": 3,
}

# Preferred column names for similarity scores (checked in this order)
KNOWN_SCORE_COLUMNS = [
    "sim_score",
    "mpnet",
    "bilingual",
    "mxbai",
    "jinja",
    "mistral",
    "sfr-mistral",
    "ling",
    "ling-mistral",
    "qwen3",
    "e5",
    "gemma",
]

# Not treated as model scores when auto-picking numeric columns
RESERVED_COLUMNS = frozenset(
    {
        "label",
        "enc_label",
        "pair",
        "pairs",
        "pair1",
        "pair2",
        "pair1_error",
        "pair2_error",
        "doc_n",
        "aspect",
        "reason",
        "domain",
        "aspect_type",
        "score_error",
    }
)


def encode_labels(series):
    """Numeric labels pass through; strings use LABEL_MAP (after strip); numeric strings coerced."""

    def encode_one(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer)):
            return int(value)  # already ordinal / numeric code
        if isinstance(value, (float, np.floating)):
            return float(value) if not np.isnan(value) else np.nan
        if isinstance(value, str):
            t = value.strip()
            if not t:
                return np.nan
            try:
                return float(t)  # e.g. "2" in Excel text cells
            except ValueError:
                return LABEL_MAP.get(t, np.nan)  # map human-readable label
        return np.nan

    # Fast path: column is already float/int
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.map(encode_one)


def infer_score_columns(frame, explicit=None):
    if explicit:
        cols = [c.strip() for c in explicit.split(",") if c.strip()]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError("Requested score columns not in file: %s" % ", ".join(missing))
        return cols

    # Prefer columns from KNOWN_SCORE_COLUMNS when present
    found = [c for c in KNOWN_SCORE_COLUMNS if c in frame.columns]
    if found:
        return found

    # Fallback: any numeric column that is not metadata
    numeric = []
    for c in frame.columns:
        if c in RESERVED_COLUMNS or c.startswith("Unnamed"):
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            numeric.append(c)
    if not numeric:
        raise ValueError(
            "No score columns found. Add embedding similarity columns, or pass "
            "--score-columns. Known names: %s" % ", ".join(KNOWN_SCORE_COLUMNS)
        )
    return sorted(numeric)


def safe_corr(x, y, fn):
    # Pairwise finite rows only so NaN labels or empty scores do not break scipy
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if mask.sum() < 2:
        return np.nan, np.nan
    a = x.to_numpy(dtype=float)[mask]
    b = y.to_numpy(dtype=float)[mask]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan, np.nan  # correlation undefined if either side is constant
    stat, p = fn(a, b)
    return float(stat), float(p)


def main():
    parser = argparse.ArgumentParser(
        description="Spearman correlation of similarity columns vs encoded labels (one xlsx)."
    )
    parser.add_argument("--file", required=True, help="Path to results Excel file")
    parser.add_argument("--sheet-name", default=None, help="Sheet name (default: first sheet)")
    parser.add_argument(
        "--score-columns",
        default=None,
        help="Comma-separated similarity column names (default: known names present in file, else numeric cols)",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column with human labels (default: label)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write CSV of correlations",
    )
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

    score_cols = infer_score_columns(frame, args.score_columns)
    missing_scores = [c for c in score_cols if c not in frame.columns]
    if missing_scores:
        print("Missing score columns: %s" % missing_scores, file=sys.stderr)
        sys.exit(1)

    enc = encode_labels(frame[args.label_column])
    frame = frame.copy()  # avoid mutating the loaded frame
    frame["enc_label"] = enc  # numeric target for correlation

    # Warn if some human labels could not be mapped or coerced
    unmapped = frame[args.label_column].notna() & frame["enc_label"].isna()
    if unmapped.any():
        bad = frame.loc[unmapped, args.label_column].dropna().unique()[:10]
        print(
            "Warning: %d rows have labels that could not be encoded (showing up to 10 unique): %s"
            % (unmapped.sum(), list(bad)),
            file=sys.stderr,
        )

    print("File:", path)
    print("Sheet:", sheet)
    print("Rows:", len(frame))
    print("Label column: %s (encoded as enc_label, ordinal 0–3 for string labels)" % args.label_column)
    print("Score columns:", ", ".join(score_cols))
    print()

    rows = []
    for col in score_cols:
        s = pd.to_numeric(frame[col], errors="coerce")  # coerce in case Excel typed cells as text
        spr, spr_p = safe_corr(s, frame["enc_label"], spearmanr)
        rows.append(
            {
                "column": col,
                "spearman": round(spr, 4) if np.isfinite(spr) else spr,
                "spearman_p": spr_p,
                "n_valid": int(
                    np.sum(np.isfinite(s.to_numpy(dtype=float)) & np.isfinite(frame["enc_label"].to_numpy(dtype=float)))
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
