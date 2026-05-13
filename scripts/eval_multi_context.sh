#!/usr/bin/env bash
set -euo pipefail

# Multi-context evaluation with the final QAFT-QA checkpoint.
# Use a final_model directory with merged LoRA weights, not an intermediate
# checkpoint-* training-state directory.
# Required:
#   CHECKPOINT=/path/to/final_model
#   INPUT_FILE=/path/to/eval.jsonl
#   OUTPUT_FILE=/path/to/predictions.jsonl
# Optional when CHECKPOINT/drift_config.json records model sources:
#   REASONING_MODEL=/path/or/hf-id
#   KNOWLEDGE_MODEL=/path/or/hf-id

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

: "${CHECKPOINT:?Set CHECKPOINT to a DRIFT final_model checkpoint directory.}"
: "${INPUT_FILE:?Set INPUT_FILE to the multi-context JSONL file.}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE to the output JSONL path.}"

DRIFT_EVAL_CMD=${DRIFT_EVAL_CMD:-"python -m drift.inference.eval_multi"}

args=(
  --checkpoint "${CHECKPOINT}"
  --input-file "${INPUT_FILE}"
  --output-file "${OUTPUT_FILE}"
  --batch-size "${BATCH_SIZE:-4}"
  --compress-ratio "${COMPRESS_RATIO:-32}"
  --compress-mode "${COMPRESS_MODE:-small_threshold}"
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
  --chunk-size "${CHUNK_SIZE:-8192}"
  --overlap "${OVERLAP:-200}"
  --num-gpus "${NUM_GPUS:-1}"
  --num-attention-heads "${NUM_ATTENTION_HEADS:-8}"
)

if [[ -n "${DEVICE:-}" ]]; then args+=(--device "${DEVICE}"); fi
if [[ -n "${REASONING_MODEL:-}" ]]; then args+=(--reasoning-model "${REASONING_MODEL}"); fi
if [[ -n "${KNOWLEDGE_MODEL:-}" ]]; then args+=(--knowledge-model "${KNOWLEDGE_MODEL}"); fi
if [[ "${USE_LAYER_NORM:-0}" == "1" ]]; then args+=(--use-layer-norm); fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"
${DRIFT_EVAL_CMD} "${args[@]}"
