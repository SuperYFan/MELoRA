#!/bin/bash
# Learn original-style few-shot LoRA compositions for Chinese harmful meme benchmarks.
# Update the checkpoint paths after running scripts/zh_module_train.sh.

set -e

DATA_ROOT=${DATA_ROOT:-data}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
LORA_ROOT=${LORA_ROOT:-./lora_modules}

HATE_LORA=${HATE_LORA:-zh_hate_semantics/checkpoint-REPLACE}
MET_LORA=${MET_LORA:-zh_met_m3_toxicn_reasoning/checkpoint-REPLACE}
VISUAL_LORA=${VISUAL_LORA:-zh_visual_expression/checkpoint-REPLACE}
MODULES="$HATE_LORA,$MET_LORA,$VISUAL_LORA"

NUM_SHOTS=${NUM_SHOTS:-4}
MAX_INFERENCE_STEP=${MAX_INFERENCE_STEP:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
CUTOFF_LEN=${CUTOFF_LEN:-512}
SEED=${SEED:-1111}
POS_WORD=${POS_WORD:-$'\u5426'}
NEG_WORD=${NEG_WORD:-$'\u662f'}
# Keep the original 101/103 log names for the default 4-shot setting.
# For other shot counts, shift SAVE_NUM automatically to avoid overwriting.
SAVE_OFFSET=${SAVE_OFFSET:-$(( NUM_SHOTS == 4 ? 0 : NUM_SHOTS * 10 ))}
SAVE_CASE_STUDY=${SAVE_CASE_STUDY:-0}
SAVE_INDIVIDUAL_LORA_OUTPUTS=${SAVE_INDIVIDUAL_LORA_OUTPUTS:-0}
CASE_STUDY_DIR=${CASE_STUDY_DIR:-LoRA/case_study}
COMPOSITION_METHOD=${COMPOSITION_METHOD:-static}
MOE_ROUTER_LR=${MOE_ROUTER_LR:-1e-3}
MOE_ROUTER_EPOCHS=${MOE_ROUTER_EPOCHS:-20}
MOE_ROUTER_MAX_STEPS=${MOE_ROUTER_MAX_STEPS:-0}
MOE_TOP_K=${MOE_TOP_K:-0}
MOE_TEMPERATURE=${MOE_TEMPERATURE:-1.0}
MOE_ROUTER_TYPE=${MOE_ROUTER_TYPE:-linear}
MOE_ROUTER_HIDDEN_SIZE=${MOE_ROUTER_HIDDEN_SIZE:-128}
MOE_ROUTER_GRANULARITY=${MOE_ROUTER_GRANULARITY:-token}
SAVE_MOE_ROUTER=${SAVE_MOE_ROUTER:-0}
MOE_ROUTER_DIR=${MOE_ROUTER_DIR:-LoRA/moe_routers}
NEW_LORA_COMBINED_LOG=${NEW_LORA_COMBINED_LOG:-}
NEW_LORA_LOG_APPEND=${NEW_LORA_LOG_APPEND:-0}

log_msg() {
    if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
        printf '%s\n' "$*" | tee -a "$NEW_LORA_COMBINED_LOG"
    else
        printf '%s\n' "$*"
    fi
}

CASE_STUDY_ARGS=()
if [[ "$SAVE_CASE_STUDY" == "1" ]]; then
    CASE_STUDY_ARGS+=(--save_case_study --case_study_dir "$CASE_STUDY_DIR")
fi
if [[ "$SAVE_INDIVIDUAL_LORA_OUTPUTS" == "1" ]]; then
    CASE_STUDY_ARGS+=(--save_individual_lora_outputs --case_study_dir "$CASE_STUDY_DIR")
fi

MOE_ARGS=(
    --composition_method "$COMPOSITION_METHOD"
    --moe_router_lr "$MOE_ROUTER_LR"
    --moe_router_epochs "$MOE_ROUTER_EPOCHS"
    --moe_router_max_steps "$MOE_ROUTER_MAX_STEPS"
    --moe_top_k "$MOE_TOP_K"
    --moe_temperature "$MOE_TEMPERATURE"
    --moe_router_type "$MOE_ROUTER_TYPE"
    --moe_router_hidden_size "$MOE_ROUTER_HIDDEN_SIZE"
    --moe_router_granularity "$MOE_ROUTER_GRANULARITY"
)
if [[ "$SAVE_MOE_ROUTER" == "1" ]]; then
    MOE_ARGS+=(--save_moe_router --moe_router_dir "$MOE_ROUTER_DIR")
fi

if [[ "$MODULES" == *"checkpoint-REPLACE"* ]]; then
    echo "Please set HATE_LORA, MET_LORA, and VISUAL_LORA to real adapter checkpoint paths first."
    echo "Example: HATE_LORA=zh_hate_semantics/checkpoint-1200 MET_LORA=zh_met_m3_toxicn_reasoning/checkpoint-300 VISUAL_LORA=zh_visual_expression/checkpoint-1332 bash scripts/zh_new_lora.sh"
    exit 1
fi

if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
    mkdir -p "$(dirname "$NEW_LORA_COMBINED_LOG")"
    if [[ "$NEW_LORA_LOG_APPEND" != "1" ]]; then
        : > "$NEW_LORA_COMBINED_LOG"
    fi
    log_msg "Combined log: $NEW_LORA_COMBINED_LOG"
fi

LOG_ARGS=()
if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
    LOG_ARGS+=(--stdout_only_log)
fi

run_one() {
    local dataset="$1"
    local eval_split="$2"
    local save_num="$3"

    log_msg ""
    log_msg "============================================================"
    log_msg "Dataset:  ${dataset} (${eval_split})"
    log_msg "SEED:     ${SEED}"
    log_msg "SAVE_NUM: ${save_num}"
    log_msg "============================================================"

    if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
        local tmp_run_log
        tmp_run_log=$(mktemp)
        python src/zh_lora_composition.py \
            --DATASET "$dataset" \
            --PATH "$DATA_ROOT" \
            --base_model "$BASE_MODEL" \
            --lora_dir "$LORA_ROOT" \
            --lora_modules "$MODULES" \
            --num_shots "$NUM_SHOTS" \
            --max_inference_step "$MAX_INFERENCE_STEP" \
            --batch_size "$BATCH_SIZE" \
            --cutoff_len "$CUTOFF_LEN" \
            --eval_split "$eval_split" \
            --POS_WORD "$POS_WORD" \
            --NEG_WORD "$NEG_WORD" \
            --SEED "$SEED" \
            --SAVE_NUM "$save_num" \
            "${LOG_ARGS[@]}" \
            "${CASE_STUDY_ARGS[@]}" \
            "${MOE_ARGS[@]}" \
            2>&1 | tee -a "$NEW_LORA_COMBINED_LOG" "$tmp_run_log"

        local metric_line
        metric_line=$(grep -E "AUC[[:space:]]+[0-9.]+,[[:space:]]*Acc[[:space:]]+[0-9.]+" "$tmp_run_log" | tail -n 1 || true)
        if [[ -n "$metric_line" ]]; then
            log_msg "SUMMARY | ${dataset} | seed=${SEED} | save_num=${save_num} | ${metric_line}"
        else
            log_msg "SUMMARY | ${dataset} | seed=${SEED} | save_num=${save_num} | metric not found"
        fi
        rm -f "$tmp_run_log"
    else
        python src/zh_lora_composition.py \
            --DATASET "$dataset" \
            --PATH "$DATA_ROOT" \
            --base_model "$BASE_MODEL" \
            --lora_dir "$LORA_ROOT" \
            --lora_modules "$MODULES" \
            --num_shots "$NUM_SHOTS" \
            --max_inference_step "$MAX_INFERENCE_STEP" \
            --batch_size "$BATCH_SIZE" \
            --cutoff_len "$CUTOFF_LEN" \
            --eval_split "$eval_split" \
            --POS_WORD "$POS_WORD" \
            --NEG_WORD "$NEG_WORD" \
            --SEED "$SEED" \
            --SAVE_NUM "$save_num" \
            "${CASE_STUDY_ARGS[@]}" \
            "${MOE_ARGS[@]}"
    fi
}

run_one "zh-toxicn-mm" "test" "$((101 + SAVE_OFFSET))"
run_one "zh-sarcnet" "test" "$((103 + SAVE_OFFSET))"

log_msg ""
if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
    log_msg "Run finished. Combined process+result log saved to: $NEW_LORA_COMBINED_LOG"
else
    log_msg "Run finished. Logs are written under ./LoRA as <dataset>_<SAVE_NUM>.txt."
fi
