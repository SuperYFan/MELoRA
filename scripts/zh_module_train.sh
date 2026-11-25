#!/bin/bash
# Train Chinese modular LoRA modules.
# The image signal is textified first, following the original Mod_HATE design:
# image/culture description + meme text -> Qwen2.5-7B text input.

set -e

DATA_ROOT=${DATA_ROOT:-data}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
OUT_ROOT=${OUT_ROOT:-./lora_modules}
POS_WORD=${POS_WORD:-$'\u5426'}
NEG_WORD=${NEG_WORD:-$'\u662f'}

# M1: Harmful Semantics Expert, Chinese toxic-language classification on ToxiCN.
python src/main.py \
    --DATASET 'zh-hate-semantics' \
    --PATH "$DATA_ROOT" \
    --base_model "$BASE_MODEL" \
    --batch_size 16 \
    --micro_batch_size 4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --num_epochs 2 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --warmup_steps 100 \
    --eval_step 100 \
    --gen_step 100000 \
    --gen_start 100000 \
    --save_step 200 \
    --disable_torch_compile \
    --POS_WORD "$POS_WORD" \
    --NEG_WORD "$NEG_WORD" \
    --output_dir "$OUT_ROOT/zh_hate_semantics" \
    --SAVE_NUM 11

# M2 for ToxiCN-MM (Harmful Meme Reasoning Expert): MET-Meme + M3-Weibo hateful-meme reasoning, deduplicated against the ToxiCN-MM test set.
python src/main.py \
    --DATASET 'zh-met-m3-toxicn-reasoning' \
    --PATH "$DATA_ROOT" \
    --base_model "$BASE_MODEL" \
    --batch_size 8 \
    --micro_batch_size 2 \
    --lora_r 16 \
    --lora_alpha 16 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --cutoff_len 768 \
    --warmup_steps 50 \
    --eval_step 50 \
    --gen_step 100000 \
    --gen_start 100000 \
    --save_step 100 \
    --disable_torch_compile \
    --POS_WORD "$POS_WORD" \
    --NEG_WORD "$NEG_WORD" \
    --output_dir "$OUT_ROOT/zh_met_m3_toxicn_reasoning" \
    --SAVE_NUM 12

# M2 for SarcNet (Harmful Meme Reasoning Expert): MET-Meme + M3-Weibo hateful-meme reasoning, deduplicated against the SarcNet test set.
python src/main.py \
    --DATASET 'zh-met-m3-sarcnet-reasoning' \
    --PATH "$DATA_ROOT" \
    --base_model "$BASE_MODEL" \
    --batch_size 8 \
    --micro_batch_size 2 \
    --lora_r 16 \
    --lora_alpha 16 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --cutoff_len 768 \
    --warmup_steps 50 \
    --eval_step 50 \
    --gen_step 100000 \
    --gen_start 100000 \
    --save_step 100 \
    --disable_torch_compile \
    --POS_WORD "$POS_WORD" \
    --NEG_WORD "$NEG_WORD" \
    --output_dir "$OUT_ROOT/zh_met_m3_sarcnet_reasoning" \
    --SAVE_NUM 13

# M3: Visual Expression Expert, meme-meaning interpretation on emo-visual-data.
# This does not train a vision encoder; it follows the original caption-based setup.
python src/main.py \
    --DATASET 'zh-visual-expression' \
    --PATH "$DATA_ROOT" \
    --base_model "$BASE_MODEL" \
    --batch_size 8 \
    --micro_batch_size 2 \
    --lora_r 16 \
    --lora_alpha 16 \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --cutoff_len 768 \
    --warmup_steps 50 \
    --eval_step 100 \
    --gen_step 100000 \
    --gen_start 100000 \
    --save_step 200 \
    --disable_torch_compile \
    --POS_WORD "$POS_WORD" \
    --NEG_WORD "$NEG_WORD" \
    --output_dir "$OUT_ROOT/zh_visual_expression" \
    --SAVE_NUM 13
