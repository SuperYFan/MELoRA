#!/bin/bash
# Measure the computational cost of MELoRA (router calibration, inference, visual verbalization).
# The configuration is identical to the main run in lora_modules/pipeline.log
# (ToxiCN-MM, 16 examples per class = 32 in total, seed 1111, dense MLP router).
# Run from the Mod_HATE-main root:  bash scripts/run_cost_measurement.sh
# No existing result, router, or prediction file is overwritten.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-.}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data}
LORA_ROOT=${LORA_ROOT:-${REPO_ROOT}/lora_modules}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
VLM_MODEL=${VLM_MODEL:-models/Qwen3-VL-8B-Instruct}
M2_TOXICN=${M2_TOXICN:-${LORA_ROOT}/zh_met_m3_toxicn_reasoning_20260917_231740/checkpoint-1848}
OUT_DIR=${REPO_ROOT}/LoRA/cost_measurement
SEED=${SEED:-1111}
RUN_VERBALIZATION=${RUN_VERBALIZATION:-1}
RUN_ROUTER=${RUN_ROUTER:-1}
# Python interpreter for the Qwen3-VL step (needs transformers >= 4.57).
PYTHON_VLM=${PYTHON_VLM:-python}

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

if [[ "${RUN_ROUTER}" == "1" ]]; then
# GPU memory trace as an independent cross-check (one line per second).
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv -l 1 \
    > "${OUT_DIR}/gpu_trace_router_seed${SEED}.csv" &
SMI_PID=$!

echo "===== [1/2] Router calibration and inference (ToxiCN-MM, K=32, seed ${SEED}) ====="
python src/measure_cost.py \
    --DATASET zh-toxicn-mm \
    --PATH "${DATA_ROOT}" \
    --base_model "${BASE_MODEL}" \
    --lora_dir "${LORA_ROOT}" \
    --lora_modules "zh_hate_semantics/checkpoint-1200,${M2_TOXICN},zh_visual_expression/checkpoint-1332" \
    --num_shots 16 \
    --max_inference_step 40 \
    --batch_size 4 \
    --cutoff_len 512 \
    --eval_split test \
    --POS_WORD "否" \
    --NEG_WORD "是" \
    --SEED "${SEED}" \
    --SAVE_NUM 99001 \
    --stdout_only_log \
    --composition_method moe \
    --moe_router_lr 1e-3 \
    --moe_router_epochs 20 \
    --moe_router_max_steps 0 \
    --moe_router_micro_batch_size 1 \
    --moe_top_k 3 \
    --moe_temperature 1.0 \
    --moe_router_type mlp \
    --moe_router_hidden_size 128 \
    --moe_router_granularity token \
    --moe_load_balance_coeff 0.0 2>&1 | tee "${OUT_DIR}/router_seed${SEED}.log"

kill ${SMI_PID} 2>/dev/null || true
fi

if [[ "${RUN_VERBALIZATION}" == "1" ]]; then
    echo "===== [2/2] Visual verbalization with Qwen3-VL-8B-Instruct (100 images) ====="
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv -l 1 \
        > "${OUT_DIR}/gpu_trace_verbalization.csv" &
    SMI_PID=$!
    "${PYTHON_VLM}" src/measure_verbalization_cost.py \
        --model_dir "${VLM_MODEL}" \
        --repo_root "${REPO_ROOT}" \
        --data_root data \
        --dataset zh-toxicn-mm \
        --num_images 100 2>&1 | tee "${OUT_DIR}/verbalization.log"
    kill ${SMI_PID} 2>/dev/null || true
fi

echo
echo "===== Summary ====="
grep -h "\[COST\]" "${OUT_DIR}"/*.log || true
echo "JSON results are in ${OUT_DIR}/"
