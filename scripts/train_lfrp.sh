#!/usr/bin/env bash
set -euo pipefail

# LFRP corresponds to the legacy pretrain_1 stage.
# Required:
#   REASONING_MODEL=/path/or/hf-id
#   KNOWLEDGE_MODEL=/path/or/hf-id
#   TRAIN_FILE=/path/to/train.parquet|json|jsonl
# Optional:
#   VAL_FILE=/path/to/val.parquet|json|jsonl
#   CHECKPOINT_DIR=/path/to/output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

: "${REASONING_MODEL:?Set REASONING_MODEL to the reasoning/main model path or HF id.}"
: "${KNOWLEDGE_MODEL:?Set KNOWLEDGE_MODEL to the knowledge/auxiliary model path or HF id.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the LFRP training file.}"

DRIFT_TRAIN_CMD=${DRIFT_TRAIN_CMD:-"python -m drift.training.train"}

args=(
  --stage lfrp
  --reasoning-model "${REASONING_MODEL}"
  --knowledge-model "${KNOWLEDGE_MODEL}"
  --main-device "${MAIN_DEVICE:-cuda:0}"
  --device-reasoning "${DEVICE_REASONING:-auto}"
  --device-knowledge "${DEVICE_KNOWLEDGE:-balanced_low_0}"
  --compress-ratio "${COMPRESS_RATIO:-8}"
  --compress-mode "${COMPRESS_MODE:-small_threshold}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --max-epochs "${MAX_EPOCHS:-1}"
  --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES:-1}"
  --warmup-ratio "${WARMUP_RATIO:-0.1}"
  --scheduler-type "${SCHEDULER_TYPE:-cosine}"
  --gradient-clip-val "${GRADIENT_CLIP_VAL:-0.5}"
  --label-smoothing-factor "${LABEL_SMOOTHING_FACTOR:-0.1}"
  --save-steps "${SAVE_STEPS:-600}"
  --log-steps "${LOG_STEPS:-10}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --lora-r "${LORA_R:-16}"
  --train-file "${TRAIN_FILE}"
  --token-range "${TOKEN_RANGE:-64-128}"
  --train-batch-size "${TRAIN_BATCH_SIZE:-128}"
  --val-batch-size "${VAL_BATCH_SIZE:-128}"
  --validation-steps "${VALIDATION_STEPS:-10}"
  --max-length "${MAX_LENGTH:-8192}"
  --chunk-size "${CHUNK_SIZE:-4096}"
  --overlap "${OVERLAP:-200}"
  --num-workers "${NUM_WORKERS:-4}"
  --dataset-num-proc "${DATASET_NUM_PROC:-64}"
  --num-attention-heads "${NUM_ATTENTION_HEADS:-8}"
)

if [[ -n "${VAL_FILE:-}" ]]; then args+=(--val-file "${VAL_FILE}"); fi
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then args+=(--checkpoint-dir "${CHECKPOINT_DIR}"); fi
if [[ "${FREEZE_REASONING:-1}" == "1" ]]; then args+=(--frozen-reasoning); fi
if [[ "${FREEZE_KNOWLEDGE:-0}" == "1" ]]; then args+=(--frozen-knowledge); fi
if [[ "${FREEZE_PROJECTOR:-0}" == "1" ]]; then args+=(--frozen-projector); fi
if [[ "${SMALL_COMPRESS_MODEL:-0}" == "1" ]]; then args+=(--small-compress-model); fi
if [[ "${USE_LAYER_NORM:-0}" == "1" ]]; then args+=(--use-layer-norm); fi
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  args+=(--use-wandb --wandb-project "${WANDB_PROJECT:-DRIFT_Training}")
fi

${DRIFT_TRAIN_CMD} "${args[@]}"
