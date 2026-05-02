#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Edit these values directly
PYTHON_BIN="python3"

# Target response file selector (auto-resolved from method + llm name)
METHOD="single"           # single | multi | summarize
LLM_NAME="qwen2.5-14b"
SUFFIX=""                 # optional override; empty => CE for single/multi, sum for summarize

# Optional settings
SHEET_NAME=""            # empty => first sheet
PRESET="second"           # first | second | all
MODELS=""                # optional override, e.g. "mpnet,e5" (empty => use PRESET)
BATCH_SIZE="8"          # embedding encode batch size

cd "$PROJECT_DIR"

if [[ -z "$SUFFIX" ]]; then
  if [[ "$METHOD" == "summarize" ]]; then
    SUFFIX="sum"
  else
    SUFFIX="CE"
  fi
fi

RESULT_DIR="$PROJECT_DIR/results/retrieve-then-embed/$METHOD"
if [[ ! -d "$RESULT_DIR" ]]; then
  echo "Result directory not found: $RESULT_DIR" >&2
  exit 1
fi

PATTERN="$RESULT_DIR/${LLM_NAME}_${METHOD}_${SUFFIX}.xlsx"
if [[ -f "$PATTERN" ]]; then
  FILE_PATH="$PATTERN"
else
  # Alternate pattern: <llm_name>_<domain>_<method>_<suffix>.xlsx
  shopt -s nullglob
  matches=("$RESULT_DIR/${LLM_NAME}_"*"_${METHOD}_${SUFFIX}.xlsx")
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "No matching result file found for LLM_NAME=$LLM_NAME METHOD=$METHOD SUFFIX=$SUFFIX" >&2
    exit 1
  elif [[ ${#matches[@]} -gt 1 ]]; then
    echo "Multiple matching files found; refine inputs or set SUFFIX explicitly:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
  else
    FILE_PATH="${matches[0]}"
  fi
fi

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/embed.py"
  --file "$FILE_PATH"
  --preset "$PRESET"
  --batch-size "$BATCH_SIZE"
)

if [[ -n "$SHEET_NAME" ]]; then
  CMD+=(--sheet-name "$SHEET_NAME")
fi

if [[ -n "$MODELS" ]]; then
  CMD+=(--models "$MODELS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
