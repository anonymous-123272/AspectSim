# AspectSim Release

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Dataset Details](#dataset-contract)
- [Pipelines](#pipelines)
  - [1. Retrieve-then-Embed](#1-retrieve-then-embed)
  - [2. Add Embedding Similarities](#2-add-embedding-similarities)
  - [3. LBS Baseline](#3-lbs-baseline)
  - [4. Projection Baseline](#4-projection-baseline)
  - [5. Evaluation](#5-evaluation)

---

## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.12+ |
| **Hardware** | NVIDIA GPU strongly recommended for vLLM inference (H100 for models above 8B) |
| **Disk / Network** | SentenceTransformer and HuggingFace models download on first use; ensure sufficient space and HF access |

> CPU-only setups may not satisfy [vLLM](https://github.com/vllm-project/vllm) dependencies. Some gated models require `huggingface-cli login` before first use.

---

## Installation

```bash
git clone <repo-url>
cd AspectSim

conda create -n myenv Python=3.12
conda activate myenv

pip install -r requirements.txt
```

`requirements.txt` includes all the dependencies requried to run the code

> If installation fails due to a CUDA/driver mismatch, follow the [official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) to match your platform.

---

## Repository Structure

```
AspectSim/
├── dataset/
│   └── aspect-sim-dataset.csv       # ~26k aspect-document pairs
├── prompts/
│   ├── aspectsim_prompts.yaml       # Prompts for retrieve-then-embed pipeline
│   └── lbs_prompts.yaml             # Prompts for LBS baseline
├── scripts/
│   ├── retrieve_then_embed/         # Main pipeline package
│   └── baseline/
│       ├── lbs/                     # LLM-based similarity baseline
│       └── proj/                    # Projection baseline
└── results/                         # Generated outputs (created on first run)
    ├── retrieve-then-embed/
    │   └── <method>/                # single | multi | summarize
    ├── baseline/
    │   ├── lbs/
    │   └── proj/
```

### `scripts/retrieve_then_embed/`

| Module | Role |
|---|---|
| `data_io.py` | Load dataset (CSV/XLSX), resolve default paths |
| `prompts.py` | Load YAML prompts for the main pipeline |
| `parsing.py` | Parse model outputs in structured format |
| `generation.py` | Batch chat generation via vLLM |
| `modeling.py` | Build vLLM engine (tensor-parallel, max length, GPU memory) |
| `runner.py` | Orchestrate extraction/summarization rows, checkpoints, Excel columns `pair1` / `pair2` |
| `retreive.py` | **CLI** — run vLLM over the dataset; writes `results/retrieve-then-embed/<method>/...xlsx` |
| `embed.py` | **CLI** — read a result workbook, append cosine similarity columns using `sentence-transformers` |
| `correlate.py` | **CLI** — compute Spearman correlation between score columns and human `label` |

### `scripts/baseline/lbs/`

| File | Role |
|---|---|
| `llm_score.py` | **CLI** — direct 0–1 LLM similarity scoring; writes `results/baseline/lbs/<llm_name>_lbs.xlsx` |
| `correlate_lbs.py` | **CLI** — Spearman: `sim_score` vs labels |
| `prompts.py` | LBS prompt assembly |
| `parsing.py` | LBS output parsing |

### `scripts/baseline/proj/`

| File | Role |
|---|---|
| `proj_sim.py` | **CLI** — L2-normalized embeddings scored by `dot(A, aspect) − dot(B, aspect)`; appends to `proj_sim.xlsx` |
| `correlate_proj.py` | **CLI** — Spearman for numeric score columns vs labels |

---

## Dataset

The dataset are stored in (`dataset/aspect-sim-dataset.csv`). More desciption of the Dataset uses and construction process are availabe at [Hugginface](https://huggingface.co/datasets/aspectsim/AspectSim-Evaluation-Benchmark). The dataset has the following fields:

| Column | Description |
|---|---|
| `doc_n` | Document pair identifier |
| `aspect` | Aspect to condition on |
| `article1`  | First document |
| `article2`  | Second document |
| `domain`  | Domain label |
| `label`  | Human similarity label (ordinal string or numeric) |
| `aspect_type` | Tag describing aspect granularity (e.g. `single`/`multi`); |
| `pair` | Ground Truth evidence pair for corresponding aspect |

---

## How to Run


### 1. Retrieve-then-Embed

Extract or summarize aspects with an LLM, then (optionally) score with embedding models.

> **Important:** `retreive.py` writes **one** side per run — `--column_name article1` populates `pair1`, and `--column_name article2` populates `pair2` (in the same workbook). You must run it **twice** (once for `article1`, once for `article2`) so that both `pair1` and `pair2` exist before calling `embed.py`. The second run reads the existing workbook and appends the other pair column.

**Option A — shell wrapper** (edit `METHOD`, `MODEL_ID`, `LLM_NAME`, `COLUMN_NAME`, `BATCH_SIZE`, etc.):

```bash
# First pass: article1 -> pair1
# (set COLUMN_NAME="article1" in the script, or leave PAIR_NAME empty to auto-derive)
bash scripts/retrieve_then_embed/run_retrieve.sh

# Second pass: article2 -> pair2
# (edit COLUMN_NAME="article2" and re-run)
bash scripts/retrieve_then_embed/run_retrieve.sh
```

**Option B — direct CLI** (run both invocations, in either order):

```bash
# Pass 1: article1 -> pair1
python3 scripts/retrieve_then_embed/retreive.py \
  --method single \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --llm_name qwen2.5-14b \
  --column_name article1 \
  --pair_name pair1 \
  --batch-size 8 \
  --prompts-yaml prompts/aspectsim_prompts.yaml

# Pass 2: article2 -> pair2
python3 scripts/retrieve_then_embed/retreive.py \
  --method single \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --llm_name qwen2.5-14b \
  --column_name article2 \
  --pair_name pair2 \
  --batch-size 8 \
  --prompts-yaml prompts/aspectsim_prompts.yaml
```

After both passes, the workbook contains `pair1`, `pair2`, and a merged `pairs` column — which is what `embed.py` consumes in [step 2](#2-add-embedding-similarities).

**Required flags:**

| Flag | Description |
|---|---|
| `--method` | `single` / `multi` (extraction) or `summarize` (aspect summary). Selects the prompt style only — does not filter dataset rows. Use `--domain` / `--start-index` / `--end-index` to subset rows. |
| `--model_id` | Hugging Face / vLLM model id to load. |
| `--llm_name` | Short tag used as the Excel sheet name and in the output filename. |
| `--column_name` | Document side to process: `article1` or `article2`. |
| `--pair_name` | Output column to write: `pair1` or `pair2` (must align with `--column_name`). |

**Optional flags:**

| Flag | Description |
|---|---|
| `--prompts-yaml` | YAML with `single_extraction`, `multi_extraction`, `summarize` keys (default: `prompts/aspectsim_prompts.yaml`). |
| `--batch-size` | Rows per vLLM `generate()` call (default `1`). |
| `--dataset` | Path to dataset CSV/XLSX |
| `--domain` | Filter by domain |
| `--start-index` / `--end-index` | Inclusive row slice (useful for smoke tests) |
| `--tensor-parallel-size` | vLLM tensor parallelism |
| `--max-model-len` | Maximum sequence length |
| `--gpu-memory-utilization` | vLLM GPU memory fraction (0–1) |

**Output path pattern:**

```
results/retrieve-then-embed/<method>/<llm_name>_<method>_<suffix>.xlsx
```

- `<suffix>` is `CE` for `single`/`multi` and `sum` for `summarize`.
- Both passes write to the **same** file; the second pass merges its column into the existing sheet.

---

### 2. Add Embedding Similarities

Appends cosine similarity columns to an existing retrieve workbook.

**Option A — shell wrapper** (edit `METHOD`, `LLM_NAME`, `PRESET`, etc.):

```bash
bash scripts/retrieve_then_embed/run_embed.sh
```

**Option B — direct CLI:**

```bash
python3 scripts/retrieve_then_embed/embed.py \
  --file results/retrieve-then-embed/single/qwen2.5-14b_single_CE.xlsx \
  --preset first
```

Use `--preset first|second|all` for predefined model sets, or `--models mpnet,e5,...` to select specific models by alias (see `embed.py`'s `MODEL_REGISTRY`). It could be possible that all the embedding models can not be loaded at the same time due to resource limitation. In such cases you can run it two times with `first` and `second` arguments.

---

### 3. LBS Baseline

Prompts an LLM to produce a direct 0–1 similarity score without aspect extraction.

**Option A — shell wrapper:**

```bash
bash scripts/baseline/lbs/run_lbs.sh
```

**Option B — direct CLI:**

```bash
python3 scripts/baseline/lbs/llm_score.py \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --llm_name qwen2.5-14b \
  --batch-size 8 \
  --checkpoint-every 50 \
  --prompts-yaml prompts/lbs_prompts.yaml
```

Default dataset: `dataset/aspect-sim-dataset.csv` (override with `--dataset`).  
Output: `results/baseline/lbs/<llm_name>_lbs.xlsx`

---

### 4. Projection Baseline

Scores pairs using geometry: `dot(A, aspect) − dot(B, aspect)` on L2-normalized embeddings.

**Option A — shell wrapper:**

```bash
bash scripts/baseline/proj/run_proj.sh
```

**Option B — direct CLI:**

```bash
python3 scripts/baseline/proj/proj_sim.py \
  --preset first \
  --batch-size 32
```

Optional: `--dataset`, `--domain`, `--start-index`, `--end-index`.  
Default output: `results/baseline/proj/proj_sim.xlsx` (sheet `proj_sim`).

---

### 5. Evaluation

Compute **Spearman correlation** between model scores and human labels. Ensure the Excel sheet contains a `label` column (or pass `--label-column`).

**Retrieve-then-embed workbook:**

```bash
python3 scripts/retrieve_then_embed/correlate.py \
  --file results/retrieve-then-embed/single/qwen2.5-14b_single_CE.xlsx \
  --sheet-name qwen2.5-14b
```

**LBS baseline:**

```bash
python3 scripts/baseline/lbs/correlate_lbs.py \
  --file results/baseline/lbs/qwen2.5-14b_lbs.xlsx
```

**Projection baseline:**

```bash
python3 scripts/baseline/proj/correlate_proj.py \
  --file results/baseline/proj/proj_sim.xlsx \
  --sheet-name proj_sim
```

Use `--score-columns` and `--output` where supported (see each script's `--help`).

---

## Citation
```bibtex
@misc{aspectsim2026,
  title  = {The Critical Role of Aspects in Measuring Document Similarity},
  year   = {2026},
}
```
