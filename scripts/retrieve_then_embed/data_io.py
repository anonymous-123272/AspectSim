from __future__ import print_function

import os

import pandas as pd

# Pipeline writes under results/<this>/<single|multi|summarize>/
PIPELINE_RESULTS_SUBDIR = "retrieve-then-embed"

# Canonical aspect benchmark CSV for baselines and retrieve defaults (repo root / dataset /).
ASPECT_DATASET_DIR = "dataset"
ASPECT_SIM_DATASET_FILE = "aspect-sim-dataset.csv"


def read_aspect_table(dataset_path):
    """Load consolidated label table from .csv (default) or legacy .xlsx."""
    path = os.path.abspath(dataset_path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Dataset not found: %s" % path)
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError("Unsupported dataset file type (use .csv or .xlsx): %s" % path)


def default_aspect_dataset_path(project_root):
    return os.path.join(
        os.path.abspath(project_root), ASPECT_DATASET_DIR, ASPECT_SIM_DATASET_FILE
    )


def load_aspect_frame(dataset_path, domain):
    """Load label table, optionally filtered by domain. All rows are kept regardless of method."""
    frame = read_aspect_table(dataset_path)
    if "domain" not in frame.columns:
        raise ValueError("Dataset must contain a 'domain' column")
    if domain is not None and str(domain).strip() and str(domain).lower() != "all":
        frame = frame[frame["domain"].astype(str) == str(domain)].reset_index(drop=True)
        if len(frame) == 0:
            raise ValueError("No rows for domain=%r in %s" % (domain, dataset_path))

    for col in ("doc_n", "aspect", "article1", "article2"):
        if col not in frame.columns:
            raise ValueError("Dataset missing required column: %s" % col)

    return frame


def output_path(results_root, llm_name, method):
    """results/retrieve-then-embed/<method>/<llm_name>_<method>_<suffix>.xlsx"""
    suffix = "sum" if method == "summarize" else "CE"
    out_dir = os.path.join(results_root, PIPELINE_RESULTS_SUBDIR, method)
    os.makedirs(out_dir, exist_ok=True)
    file_name = "%s_%s_%s.xlsx" % (llm_name, method, suffix)
    return os.path.join(out_dir, file_name)
