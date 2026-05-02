#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Edit these values directly
PYTHON_BIN="python3"

METHOD="single"                          # single | multi | summarize
MODEL_ID="Qwen/Qwen2.5-14B-Instruct"     # HF model id
LLM_NAME="qwen2.5-14b"                   # run tag used in output file name
COLUMN_NAME="article2"                   # article1 | article2
BATCH_SIZE="8"                           # integer >= 1
START_INDEX="0"                          # optional inclusive start index, e.g. 0
END_INDEX="16"                           # optional inclusive end index, e.g. 99

# Optional settings
PAIR_NAME=""                             # pair1 | pair2 | leave empty for auto from COLUMN_NAME
DOMAIN=""                                # optional; empty means all domains
PROMPTS_YAML="$PROJECT_DIR/prompts/aspectsim_prompts.yaml"

cd "$PROJECT_DIR"

resolve_pair_name() {
  local column_name="$1"
  local pair_name="${2:-}"
  if [[ -n "$pair_name" ]]; then
    echo "$pair_name"
    return
  fi
  if [[ "$column_name" == "article1" ]]; then
    echo "pair1"
  elif [[ "$column_name" == "article2" ]]; then
    echo "pair2"
  else
    echo "Invalid COLUMN_NAME: $column_name (use article1 or article2)" >&2
    exit 1
  fi
}

run_one() {
  local column_name="$1"
  local pair_name="$2"

  local cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/retreive.py"
    --method "$METHOD"
    --model_id "$MODEL_ID"
    --llm_name "$LLM_NAME"
    --column_name "$column_name"
    --pair_name "$pair_name"
    --batch-size "$BATCH_SIZE"
    --prompts-yaml "$PROMPTS_YAML"
  )

  if [[ -n "$DOMAIN" ]]; then
    cmd+=(--domain "$DOMAIN")
  fi
  if [[ -n "$START_INDEX" ]]; then
    cmd+=(--start-index "$START_INDEX")
  fi
  if [[ -n "$END_INDEX" ]]; then
    cmd+=(--end-index "$END_INDEX")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

PAIR_NAME="$(resolve_pair_name "$COLUMN_NAME" "$PAIR_NAME")"
run_one "$COLUMN_NAME" "$PAIR_NAME"
