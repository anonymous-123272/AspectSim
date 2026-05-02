#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_BIN="python3"

# Same presets as retrieve_then_embed/embed.py: first | second | all
PRESET="second"

# Optional: override preset with comma-separated aliases (uncomment and set)
# MODELS="mpnet,e5"

BATCH_SIZE="64"
ROW_CHUNK="512"

DOMAIN=""
DATASET=""
OUTPUT=""

START_INDEX="0"
END_INDEX="16"

cd "$PROJECT_DIR"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/proj_sim.py"
  --preset "$PRESET"
  --batch-size "$BATCH_SIZE"
  --row-chunk "$ROW_CHUNK"
)

if [[ -n "${MODELS:-}" ]]; then
  cmd+=(--models "$MODELS")
fi
if [[ -n "$DATASET" ]]; then
  cmd+=(--dataset "$DATASET")
fi
if [[ -n "$DOMAIN" ]]; then
  cmd+=(--domain "$DOMAIN")
fi
if [[ -n "$OUTPUT" ]]; then
  cmd+=(--output "$OUTPUT")
fi
if [[ -n "$START_INDEX" ]]; then
  cmd+=(--start-index "$START_INDEX")
fi
if [[ -n "$END_INDEX" ]]; then
  cmd+=(--end-index "$END_INDEX")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
