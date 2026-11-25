#!/bin/bash
# Train all 3 Qwen3-VL LoRA modules sequentially.
#
# LoRA1: text-only input  → binary classification   (hate_semantics)
# LoRA2: image input      → reasoning text output   (met_meme_reasoning)
# LoRA3: image input      → visual description      (visual_expression)
#
# Run from repo root:  bash src/qwen3vl/train_loras.sh
#
# The saved adapter checkpoints are placed under LORA_OUT_ROOT and are
# directly usable as --lora_modules in src/zh_lora_composition.py.

set -e

BASE_MODEL=models/Qwen3-VL-8B-Instruct
REPO_ROOT=.
DATA_PATH=${REPO_ROOT}/data
LORA_OUT_ROOT=${REPO_ROOT}/lora_modules_vl

# LoRA hyper-parameters (common across all three modules)
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
MAX_LENGTH_TEXT=1024   # text-only LoRA1
MAX_LENGTH_IMG=2048    # image LoRA2 / LoRA3 (images expand to many visual tokens)
LOGGING_STEPS=20

# -----------------------------------------------------------------------
# LoRA1  –  text-only binary classification  (hate_semantics)
# -----------------------------------------------------------------------
echo "===== Training LoRA1 (text classification) ====="
python src/qwen3vl/train_lora.py \
    --module       lora1 \
    --base_model   "${BASE_MODEL}" \
    --repo_root    "${REPO_ROOT}" \
    --data_path    "${DATA_PATH}" \
    --output_dir   "${LORA_OUT_ROOT}/lora1" \
    --lora_r       ${LORA_R} \
    --lora_alpha   ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_epochs   2 \
    --batch_size   4 \
    --grad_accum   4 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --max_length   ${MAX_LENGTH_TEXT} \
    --logging_steps ${LOGGING_STEPS}

echo "LoRA1 adapter saved to: ${LORA_OUT_ROOT}/lora1/adapter"

# -----------------------------------------------------------------------
# LoRA2  –  image → reasoning-process text  (met_meme_reasoning)
# -----------------------------------------------------------------------
echo "===== Training LoRA2 (image -> reasoning) ====="
python src/qwen3vl/train_lora.py \
    --module       lora2 \
    --base_model   "${BASE_MODEL}" \
    --repo_root    "${REPO_ROOT}" \
    --data_path    "${DATA_PATH}" \
    --output_dir   "${LORA_OUT_ROOT}/lora2" \
    --lora_r       ${LORA_R} \
    --lora_alpha   ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_epochs   3 \
    --batch_size   1 \
    --grad_accum   16 \
    --learning_rate 2e-4 \
    --warmup_steps 50 \
    --max_length   ${MAX_LENGTH_IMG} \
    --logging_steps ${LOGGING_STEPS}

echo "LoRA2 adapter saved to: ${LORA_OUT_ROOT}/lora2/adapter"

# -----------------------------------------------------------------------
# LoRA3  –  image → visual description  (visual_expression)
# -----------------------------------------------------------------------
echo "===== Training LoRA3 (image -> description) ====="
python src/qwen3vl/train_lora.py \
    --module       lora3 \
    --base_model   "${BASE_MODEL}" \
    --repo_root    "${REPO_ROOT}" \
    --data_path    "${DATA_PATH}" \
    --output_dir   "${LORA_OUT_ROOT}/lora3" \
    --lora_r       ${LORA_R} \
    --lora_alpha   ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_epochs   2 \
    --batch_size   1 \
    --grad_accum   16 \
    --learning_rate 2e-4 \
    --warmup_steps 50 \
    --max_length   ${MAX_LENGTH_IMG} \
    --logging_steps ${LOGGING_STEPS}

echo "LoRA3 adapter saved to: ${LORA_OUT_ROOT}/lora3/adapter"
echo ""
echo "All 3 LoRA modules trained. Adapters under ${LORA_OUT_ROOT}/"
echo "Next: pass the three adapter directories to src/zh_lora_composition.py via --lora_modules"
