#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Edit these values directly
PYTHON_BIN="python3"

MODEL_ID="Qwen/Qwen2.5-14B-Instruct"     # HF / vLLM model id
LLM_NAME="qwen2.5-14b"                   # sheet name and output file tag
BATCH_SIZE="8"                           # rows per vLLM generate() call
CHECKPOINT_EVERY="50"                    # write Excel every N completed rows

# Optional row slice (leave empty to process full filtered frame)
START_INDEX="0"                           # inclusive
END_INDEX="16"                             # inclusive

# Optional settings
DOMAIN=""                                # optional domain filter
PROMPTS_YAML="$PROJECT_DIR/prompts/lbs_prompts.yaml"
DATASET=""                               # empty => llm_score.py default dataset path

# Optional vLLM overrides (leave empty to use llm_score / modeling defaults)
TENSOR_PARALLEL_SIZE=""
MAX_MODEL_LEN=""
GPU_MEMORY_UTILIZATION=""

cd "$PROJECT_DIR"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/llm_score.py"
  --model_id "$MODEL_ID"
  --llm_name "$LLM_NAME"
  --batch-size "$BATCH_SIZE"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --prompts-yaml "$PROMPTS_YAML"
)

if [[ -n "$DATASET" ]]; then
  cmd+=(--dataset "$DATASET")
fi
if [[ -n "$DOMAIN" ]]; then
  cmd+=(--domain "$DOMAIN")
fi
if [[ -n "$START_INDEX" ]]; then
  cmd+=(--start-index "$START_INDEX")
fi
if [[ -n "$END_INDEX" ]]; then
  cmd+=(--end-index "$END_INDEX")
fi
if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
  cmd+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
  cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
