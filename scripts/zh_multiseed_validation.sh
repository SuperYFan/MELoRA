#!/bin/bash
# Multi-seed validation for the best configuration found so far:
#   moe_mlp_top2 + full_hate_met_visual + shots=16 + lb=0.01
#
# Also validates static hate_plus_visual baseline with 5 seeds for fair comparison.
#
# Motivation: shots=16, lb=0.01 gave sarcnet 78.33 (single seed=1111).
# sarcnet has high variance (std ~1.84 in earlier 5-seed runs). Need 5 seeds
# to confirm this is a reliable improvement over the baseline.
#
# SAVE_BASE=14000 avoids collision with prior runs (6000, 8000, 10000, 12000).
#
# Usage:
#   HATE_LORA=zh_hate_semantics/checkpoint-1200 \
#   MET_LORA=zh_met_m3_toxicn_reasoning/checkpoint-1812 \
#   VISUAL_LORA=zh_visual_expression/checkpoint-1332 \
#   bash scripts/zh_multiseed_validation.sh
#
# To test a different configuration:
#   NUM_SHOTS=12 MOE_LOAD_BALANCE_COEFF=0.02 bash scripts/zh_multiseed_validation.sh

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
LORA_ROOT=${LORA_ROOT:-./lora_modules}

HATE_LORA=${HATE_LORA:-zh_hate_semantics/checkpoint-REPLACE}
MET_LORA=${MET_LORA:-zh_met_m3_toxicn_reasoning/checkpoint-REPLACE}
VISUAL_LORA=${VISUAL_LORA:-zh_visual_expression/checkpoint-REPLACE}

MAX_INFERENCE_STEP=${MAX_INFERENCE_STEP:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
CUTOFF_LEN=${CUTOFF_LEN:-512}
# 5 fixed seeds
SEEDS=${SEEDS:-"1111 2222 3333 4444 5555"}
POS_WORD=${POS_WORD:-$'\u5426'}
NEG_WORD=${NEG_WORD:-$'\u662f'}
LIMIT_DEBUG=${LIMIT_DEBUG:-0}

MOE_ROUTER_LR=${MOE_ROUTER_LR:-1e-3}
MOE_ROUTER_EPOCHS=${MOE_ROUTER_EPOCHS:-20}

# Configuration to validate — override from env to test other settings
NUM_SHOTS=${NUM_SHOTS:-16}
MOE_LOAD_BALANCE_COEFF=${MOE_LOAD_BALANCE_COEFF:-0.01}

SAVE_BASE=${SAVE_BASE:-14000}

RESULTS_TSV=${RESULTS_TSV:-LoRA/zh_multiseed_validation_results.tsv}
LOG_FILE=${LOG_FILE:-LoRA/zh_multiseed_validation.log}

ALL_MODULES="$HATE_LORA,$MET_LORA,$VISUAL_LORA"
if [[ "$ALL_MODULES" == *"checkpoint-REPLACE"* ]]; then
    echo "ERROR: Set HATE_LORA, MET_LORA, VISUAL_LORA to real adapter paths first." >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$RESULTS_TSV")"
: > "$LOG_FILE"
printf 'config\tdataset\tseed\tsave_num\tnum_shots\tload_balance_coeff\tauc\tacc\n' > "$RESULTS_TSV"

log_msg() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }

DEBUG_ARGS=()
if [[ "$LIMIT_DEBUG" == "1" ]]; then DEBUG_ARGS+=(--DEBUG True); fi

save_counter=0

run_one() {
    local config_name="$1"
    local modules="$2"
    local dataset="$3"
    local seed="$4"
    local n_shots="$5"
    local composition="$6"
    local router_type="$7"
    local top_k="$8"
    local lb="$9"

    save_counter=$((save_counter + 1))
    local save_num=$((SAVE_BASE + save_counter))

    log_msg ""
    log_msg "============================================================"
    log_msg "Config:  ${config_name}"
    log_msg "Dataset: ${dataset}  |  Seed: ${seed}  |  SAVE_NUM: ${save_num}"
    log_msg "shots=${n_shots}, lb=${lb}"
    log_msg "============================================================"

    local tmp_log; tmp_log=$(mktemp)

    python src/zh_lora_composition.py \
        --DATASET "$dataset" \
        --PATH "$DATA_ROOT" \
        --base_model "$BASE_MODEL" \
        --lora_dir "$LORA_ROOT" \
        --lora_modules "$modules" \
        --num_shots "$n_shots" \
        --max_inference_step "$MAX_INFERENCE_STEP" \
        --batch_size "$BATCH_SIZE" \
        --cutoff_len "$CUTOFF_LEN" \
        --eval_split "test" \
        --POS_WORD "$POS_WORD" \
        --NEG_WORD "$NEG_WORD" \
        --SEED "$seed" \
        --SAVE_NUM "$save_num" \
        --stdout_only_log \
        --composition_method "$composition" \
        --moe_router_lr "$MOE_ROUTER_LR" \
        --moe_router_epochs "$MOE_ROUTER_EPOCHS" \
        --moe_top_k "$top_k" \
        --moe_temperature 1.0 \
        --moe_router_type "$router_type" \
        --moe_router_hidden_size 128 \
        --moe_router_granularity "token" \
        --moe_load_balance_coeff "$lb" \
        "${DEBUG_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE" "$tmp_log"

    local metric_line
    metric_line=$(grep -E "AUC[[:space:]]+[0-9.]+,[[:space:]]*Acc[[:space:]]+[0-9.]+" "$tmp_log" | tail -n 1 || true)
    if [[ -z "$metric_line" ]]; then
        log_msg "RESULT NOT FOUND: ${config_name} | ${dataset} | seed=${seed}"
        rm -f "$tmp_log"; exit 1
    fi
    local auc acc
    auc=$(printf '%s\n' "$metric_line" | sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\1/')
    acc=$(printf '%s\n' "$metric_line" | sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\2/')
    log_msg "RESULT | ${config_name} | ${dataset} | seed=${seed} | AUC ${auc}, Acc ${acc}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$config_name" "$dataset" "$seed" "$save_num" \
        "$n_shots" "$lb" "$auc" "$acc" >> "$RESULTS_TSV"
    rm -f "$tmp_log"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for current_seed in $SEEDS; do
    log_msg ""
    log_msg "########################################"
    log_msg "# Seed: ${current_seed}"
    log_msg "########################################"

    # Baseline 1: static hate_plus_visual, 4-shot (original best on toxicn)
    run_one "static_hate_plus_visual_shots4" \
        "$HATE_LORA,$VISUAL_LORA" "zh-toxicn-mm" "$current_seed" \
        4 "static" "linear" 0 0.0
    run_one "static_hate_plus_visual_shots4" \
        "$HATE_LORA,$VISUAL_LORA" "zh-sarcnet" "$current_seed" \
        4 "static" "linear" 0 0.0

    # Baseline 2: moe_mlp_top2 full3, 4-shot, lb=0 (previous MoE baseline)
    run_one "moe_mlp_top2_full3_shots4_lb0" \
        "$ALL_MODULES" "zh-toxicn-mm" "$current_seed" \
        4 "moe" "mlp" 2 0.0
    run_one "moe_mlp_top2_full3_shots4_lb0" \
        "$ALL_MODULES" "zh-sarcnet" "$current_seed" \
        4 "moe" "mlp" 2 0.0

    # Candidate A: shots=12, lb=0.02 — best single-dataset toxicn (78.79 > static 78.49)
    run_one "moe_mlp_top2_full3_shots12_lb0.02" \
        "$ALL_MODULES" "zh-toxicn-mm" "$current_seed" \
        12 "moe" "mlp" 2 0.02
    run_one "moe_mlp_top2_full3_shots12_lb0.02" \
        "$ALL_MODULES" "zh-sarcnet" "$current_seed" \
        12 "moe" "mlp" 2 0.02

    # Candidate B: shots=16, lb=0.01 — best sarcnet (78.33)
    run_one "moe_mlp_top2_full3_shots16_lb0.01" \
        "$ALL_MODULES" "zh-toxicn-mm" "$current_seed" \
        16 "moe" "mlp" 2 0.01
    run_one "moe_mlp_top2_full3_shots16_lb0.01" \
        "$ALL_MODULES" "zh-sarcnet" "$current_seed" \
        16 "moe" "mlp" 2 0.01

    # Candidate C: shots=16, lb=0.02 — best mean AUC (78.01), most balanced
    run_one "moe_mlp_top2_full3_shots16_lb0.02" \
        "$ALL_MODULES" "zh-toxicn-mm" "$current_seed" \
        16 "moe" "mlp" 2 0.02
    run_one "moe_mlp_top2_full3_shots16_lb0.02" \
        "$ALL_MODULES" "zh-sarcnet" "$current_seed" \
        16 "moe" "mlp" 2 0.02
done

# ── Summary ───────────────────────────────────────────────────────────────────
python - "$RESULTS_TSV" "$LOG_FILE" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

results_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])

with results_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

# (config, dataset) -> [auc]
grouped = defaultdict(list)
for row in rows:
    grouped[(row["config"], row["dataset"])].append(float(row["auc"]))

datasets = ["zh-toxicn-mm", "zh-sarcnet"]
lines = [
    "", "=" * 72,
    "Multi-seed Validation — AUC mean ± std (5 seeds)",
    "=" * 72,
]

configs = sorted({row["config"] for row in rows})
for config in configs:
    lines.append(f"\n  {config}")
    means = []
    for ds in datasets:
        vals = grouped[(config, ds)]
        if not vals:
            continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        means.append(m)
        lines.append(f"    {ds:<16} AUC {m:.3f} ± {s:.3f}  (n={len(vals)})")
    if len(means) == 2:
        lines.append(f"    {'mean':<16} AUC {statistics.mean(means):.3f}")

lines.append("")
summary = "\n".join(lines)
print(summary)
with log_path.open("a", encoding="utf-8") as f:
    f.write(summary + "\n")
PY

log_msg ""
log_msg "Results TSV : $RESULTS_TSV"
log_msg "Full log    : $LOG_FILE"
