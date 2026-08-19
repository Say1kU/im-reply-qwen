#!/usr/bin/env bash
set -euo pipefail

# Prevent plotting from requiring a desktop/Tk environment after training.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"

# Run in Linux/WSL with Python 3.11. The 0.6B default is selected for an 8GB RTX 4060.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3-06b-im-lora}"
DATASET="${DATASET:-data/seed_train.jsonl}"

swift sft \
  --model "${MODEL_ID}" \
  --tuner_type lora \
  --dataset "${DATASET}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 2 \
  --logging_steps 2 \
  --max_length 1024 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 2 \
  --output_dir "${OUTPUT_DIR}"
