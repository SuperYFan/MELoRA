#!/bin/bash
# Training individual LoRA modules on Chinese datasets
# ToxiCN_MM: Chinese harmful meme detection (multimodal with image description)
# sarcnet: Chinese sarcasm detection (text-based)

# Note: Set --PATH to the parent directory of your data folder
# Note: Set --output_dir to where you want to save the LoRA modules
# Note: POS_WORD='否' (No/Non-harmful), NEG_WORD='是' (Yes/Harmful)
# Note: base_model uses Qwen2.5-7B (local path on server)

# ToxiCN_MM - harmful meme detection with image description + text
python src/main.py \
    --DATASET 'toxicn' \
    --PATH 'data' \
    --base_model 'models/Qwen2.5-7B' \
    --batch_size 16 \
    --micro_batch_size 4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --num_epochs 2 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --warmup_steps 100 \
    --eval_step 50 \
    --gen_step 200 \
    --gen_start 350 \
    --save_step 100 \
    --save_case_study \
    --POS_WORD '否' \
    --NEG_WORD '是' \
    --output_dir './lora_modules/toxicn' \
    --SAVE_NUM 0

# sarcnet - sarcasm detection (text-based)
# Note: sarcnet only has 1240 train samples, ~77 iters/epoch, 154 total (2 epochs)
#       So gen_start/gen_step must be much smaller than toxicn
python src/main.py \
    --DATASET 'sarcnet' \
    --PATH 'data' \
    --base_model 'models/Qwen2.5-7B' \
    --batch_size 16 \
    --micro_batch_size 4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --warmup_steps 20 \
    --eval_step 25 \
    --gen_step 50 \
    --gen_start 50 \
    --save_step 50 \
    --save_case_study \
    --lora_dropout 0.1 \
    --POS_WORD '否' \
    --NEG_WORD '是' \
    --output_dir './lora_modules/sarcnet' \
    --SAVE_NUM 0
