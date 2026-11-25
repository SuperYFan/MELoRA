#!/bin/bash
# Reproduce one seed of the best MELoRA configuration:
# MLP router, all three experts (top-3), 16 shots per class (32 total).
# Saves the full process log, per-example predictions, trained router state,
# and router activation analysis generated from that exact saved state.

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
LORA_ROOT=${LORA_ROOT:-./lora_modules}

HATE_LORA=${HATE_LORA:-zh_hate_semantics/checkpoint-1200}
MET_LORA=${MET_LORA:-zh_met_m3_toxicn_reasoning/checkpoint-1812}
VISUAL_LORA=${VISUAL_LORA:-zh_visual_expression/checkpoint-1332}
MODULES=${HATE_LORA},${MET_LORA},${VISUAL_LORA}

SEED=${SEED:-1111}
DATASETS=${DATASETS:-"zh-toxicn-mm zh-sarcnet"}
NUM_SHOTS=${NUM_SHOTS:-16}
MOE_TOP_K=${MOE_TOP_K:-3}
# zh_new_lora_5runs.sh did not pass this option, so the original top-3 run
# used lora_config.py's default value 0.0 unless explicitly overridden.
MOE_LOAD_BALANCE_COEFF=${MOE_LOAD_BALANCE_COEFF:-0.0}
MOE_ROUTER_LR=${MOE_ROUTER_LR:-1e-3}
MOE_ROUTER_EPOCHS=${MOE_ROUTER_EPOCHS:-20}
MOE_ROUTER_MAX_STEPS=${MOE_ROUTER_MAX_STEPS:-0}
MOE_ROUTER_HIDDEN_SIZE=${MOE_ROUTER_HIDDEN_SIZE:-128}
MOE_TEMPERATURE=${MOE_TEMPERATURE:-1.0}
MOE_ROUTER_GRANULARITY=${MOE_ROUTER_GRANULARITY:-token}

MAX_INFERENCE_STEP=${MAX_INFERENCE_STEP:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
CUTOFF_LEN=${CUTOFF_LEN:-512}
POS_WORD=${POS_WORD:-$'\u5426'}
NEG_WORD=${NEG_WORD:-$'\u662f'}
MAX_ROUTER_ANALYSIS_SAMPLES=${MAX_ROUTER_ANALYSIS_SAMPLES:-0}
SAVE_BASE=${SAVE_BASE:-32000}

RUN_NAME=${RUN_NAME:-moe_mlp_top3_shots16_seed${SEED}}
OUTPUT_ROOT=${OUTPUT_ROOT:-LoRA/reproduction/${RUN_NAME}}
PREDICTION_DIR=${PREDICTION_DIR:-${OUTPUT_ROOT}/predictions}
ROUTER_DIR=${ROUTER_DIR:-${OUTPUT_ROOT}/routers}
ROUTER_ANALYSIS_DIR=${ROUTER_ANALYSIS_DIR:-${OUTPUT_ROOT}/router_analysis}
PROCESS_LOG=${PROCESS_LOG:-${OUTPUT_ROOT}/process.log}
ROUTER_LOG=${ROUTER_LOG:-${OUTPUT_ROOT}/router_analysis.log}
RESULTS_TSV=${RESULTS_TSV:-${OUTPUT_ROOT}/results.tsv}

if [[ "${NUM_SHOTS}" != "16" ]] || [[ "${MOE_TOP_K}" != "3" ]]; then
    echo "WARNING: overriding the reproduction setting: NUM_SHOTS=${NUM_SHOTS}, MOE_TOP_K=${MOE_TOP_K}" >&2
fi
if [[ ! -e "${BASE_MODEL}" ]]; then
    echo "ERROR: base model not found: ${BASE_MODEL}" >&2
    exit 1
fi
for module in ${HATE_LORA} ${MET_LORA} ${VISUAL_LORA}; do
    if [[ "${module}" = /* ]] || [[ -e "${module}" ]]; then
        module_path=${module}
    else
        module_path=${LORA_ROOT}/${module}
    fi
    if [[ ! -e "${module_path}" ]]; then
        echo "ERROR: LoRA module not found: ${module_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}" "${PREDICTION_DIR}" "${ROUTER_DIR}" "${ROUTER_ANALYSIS_DIR}"
: > "${PROCESS_LOG}"
: > "${ROUTER_LOG}"
printf 'config\tdataset\tseed\tsave_num\tnum_shots_per_class\ttotal_samples\ttop_k\tlb\tauc\tacc\tpredictions\trouter_checkpoint\n' > "${RESULTS_TSV}"

log_msg() { printf '%s\n' "$*" | tee -a "${PROCESS_LOG}"; }

save_num_for() {
    case "$1" in
        zh-toxicn-mm) echo $((SAVE_BASE + 101)) ;;
        zh-sarcnet) echo $((SAVE_BASE + 103)) ;;
        *) echo "ERROR: unsupported dataset: $1" >&2; return 1 ;;
    esac
}

log_msg "Configuration: ${RUN_NAME}"
log_msg "Seed: ${SEED}"
log_msg "Shots: ${NUM_SHOTS} per class ($((NUM_SHOTS * 2)) total)"
log_msg "Router: mlp, top-k=${MOE_TOP_K}, hidden=${MOE_ROUTER_HIDDEN_SIZE}, lb=${MOE_LOAD_BALANCE_COEFF}"
log_msg "Experts: ${MODULES}"

for dataset in ${DATASETS}; do
    save_num=$(save_num_for "${dataset}")
    run_case_dir=${PREDICTION_DIR}/${dataset}_${save_num}_moe
    prediction_jsonl=${run_case_dir}/eval_predictions.jsonl
    router_checkpoint=${ROUTER_DIR}/${dataset}_${save_num}_router.pt
    tmp_log=$(mktemp)

    log_msg ""
    log_msg "============================================================"
    log_msg "Training and evaluation: ${dataset}"
    log_msg "SAVE_NUM=${save_num}"
    log_msg "Predictions=${prediction_jsonl}"
    log_msg "Router checkpoint=${router_checkpoint}"
    log_msg "============================================================"

    python -u src/zh_lora_composition.py \
        --DATASET "${dataset}" \
        --PATH "${DATA_ROOT}" \
        --base_model "${BASE_MODEL}" \
        --lora_dir "${LORA_ROOT}" \
        --lora_modules "${MODULES}" \
        --num_shots "${NUM_SHOTS}" \
        --max_inference_step "${MAX_INFERENCE_STEP}" \
        --batch_size "${BATCH_SIZE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --eval_split test \
        --POS_WORD "${POS_WORD}" \
        --NEG_WORD "${NEG_WORD}" \
        --SEED "${SEED}" \
        --SAVE_NUM "${save_num}" \
        --stdout_only_log \
        --save_case_study \
        --case_study_dir "${PREDICTION_DIR}" \
        --save_moe_router \
        --moe_router_dir "${ROUTER_DIR}" \
        --composition_method moe \
        --moe_router_lr "${MOE_ROUTER_LR}" \
        --moe_router_epochs "${MOE_ROUTER_EPOCHS}" \
        --moe_router_max_steps "${MOE_ROUTER_MAX_STEPS}" \
        --moe_top_k "${MOE_TOP_K}" \
        --moe_temperature "${MOE_TEMPERATURE}" \
        --moe_router_type mlp \
        --moe_router_hidden_size "${MOE_ROUTER_HIDDEN_SIZE}" \
        --moe_router_granularity "${MOE_ROUTER_GRANULARITY}" \
        --moe_load_balance_coeff "${MOE_LOAD_BALANCE_COEFF}" \
        2>&1 | tee -a "${PROCESS_LOG}" "${tmp_log}"

    metric_line=$(grep -E "AUC[[:space:]]+[0-9.]+,[[:space:]]*Acc[[:space:]]+[0-9.]+" "${tmp_log}" | tail -n 1 || true)
    rm -f "${tmp_log}"
    if [[ -z "${metric_line}" ]]; then
        echo "ERROR: metric line not found for ${dataset}" >&2
        exit 1
    fi
    auc=$(printf '%s\n' "${metric_line}" | sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\1/')
    acc=$(printf '%s\n' "${metric_line}" | sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\2/')

    if [[ ! -s "${prediction_jsonl}" ]]; then
        echo "ERROR: prediction output missing: ${prediction_jsonl}" >&2
        exit 1
    fi
    if [[ ! -s "${router_checkpoint}" ]]; then
        echo "ERROR: router checkpoint missing: ${router_checkpoint}" >&2
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "moe_mlp_top3_shots16" "${dataset}" "${SEED}" "${save_num}" \
        "${NUM_SHOTS}" "$((NUM_SHOTS * 2))" "${MOE_TOP_K}" \
        "${MOE_LOAD_BALANCE_COEFF}" "${auc}" "${acc}" \
        "${prediction_jsonl}" "${router_checkpoint}" >> "${RESULTS_TSV}"

    log_msg "RESULT | ${dataset} | seed=${SEED} | AUC ${auc}, Acc ${acc}"
    log_msg "Starting router analysis from the saved prediction-run checkpoint."

    python -u src/zh_router_analysis.py \
        --dataset "${dataset}" \
        --base_model "${BASE_MODEL}" \
        --lora_modules "${MODULES}" \
        --lora_dir "${LORA_ROOT}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${ROUTER_ANALYSIS_DIR}" \
        --seed "${SEED}" \
        --num_shots "${NUM_SHOTS}" \
        --moe_top_k "${MOE_TOP_K}" \
        --moe_load_balance_coeff "${MOE_LOAD_BALANCE_COEFF}" \
        --moe_router_hidden_size "${MOE_ROUTER_HIDDEN_SIZE}" \
        --moe_router_epochs "${MOE_ROUTER_EPOCHS}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --batch_size "${BATCH_SIZE}" \
        --max_samples "${MAX_ROUTER_ANALYSIS_SAMPLES}" \
        --router_checkpoint "${router_checkpoint}" \
        2>&1 | tee -a "${ROUTER_LOG}"
done

log_msg ""
log_msg "Reproduction completed."
log_msg "Results:         ${RESULTS_TSV}"
log_msg "Process log:     ${PROCESS_LOG}"
log_msg "Predictions:     ${PREDICTION_DIR}"
log_msg "Router states:   ${ROUTER_DIR}"
log_msg "Router analysis: ${ROUTER_ANALYSIS_DIR}"
