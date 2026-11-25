#!/bin/bash
# In-domain calibration ratio experiment.
#
# Instead of a fixed few-shot count, the MoE router is trained on a random
# subset of the full training split (natural class distribution).
#
# Ratios tested: 5%, 20%, 25%, 50%
# Base config   : MoE mlp top-2, full 3 modules, lb=0.01  (best config)
# Router epochs : 5  (fewer epochs compensate for the larger dataset size)
# Seeds         : 1111 2222 3333 4444 5555  (5-seed evaluation)
# SAVE_BASE     : 18000  (no collision with prior runs up to 16000+)
#
# For reference, 16-shot baseline results are already in
#   LoRA/zh_multiseed_validation_results.tsv
#
# Usage:
#   HATE_LORA=zh_hate_semantics/checkpoint-1200 \
#   MET_LORA=zh_met_m3_toxicn_reasoning/checkpoint-1812 \
#   VISUAL_LORA=zh_visual_expression/checkpoint-1332 \
#   bash scripts/zh_calibration_ratio.sh

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
LORA_ROOT=${LORA_ROOT:-./lora_modules}

HATE_LORA=${HATE_LORA:-zh_hate_semantics/checkpoint-REPLACE}
MET_LORA=${MET_LORA:-zh_met_m3_toxicn_reasoning/checkpoint-REPLACE}
VISUAL_LORA=${VISUAL_LORA:-zh_visual_expression/checkpoint-REPLACE}

BATCH_SIZE=${BATCH_SIZE:-4}
CUTOFF_LEN=${CUTOFF_LEN:-512}
SEEDS=${SEEDS:-"1111 2222 3333 4444 5555"}
# calibration ratios to sweep
RATIOS=${RATIOS:-"0.05 0.20 0.25 0.50"}

POS_WORD=${POS_WORD:-$'\u5426'}
NEG_WORD=${NEG_WORD:-$'\u662f'}
LIMIT_DEBUG=${LIMIT_DEBUG:-0}

MOE_ROUTER_LR=${MOE_ROUTER_LR:-1e-3}
# Fewer epochs than few-shot setting (3 router epochs for large calibration sets)
MOE_ROUTER_EPOCHS=${MOE_ROUTER_EPOCHS:-3}
MOE_LOAD_BALANCE_COEFF=${MOE_LOAD_BALANCE_COEFF:-0.01}
MOE_TOP_K=${MOE_TOP_K:-2}
MOE_TEMPERATURE=${MOE_TEMPERATURE:-1.0}
MOE_ROUTER_TYPE=${MOE_ROUTER_TYPE:-mlp}
MOE_ROUTER_HIDDEN_SIZE=${MOE_ROUTER_HIDDEN_SIZE:-128}
MOE_ROUTER_GRANULARITY=${MOE_ROUTER_GRANULARITY:-token}

SAVE_BASE=${SAVE_BASE:-18000}

RESULTS_TSV=${RESULTS_TSV:-LoRA/zh_calibration_ratio_results.tsv}
LOG_FILE=${LOG_FILE:-LoRA/zh_calibration_ratio.log}

ALL_MODULES="$HATE_LORA,$MET_LORA,$VISUAL_LORA"
if [[ "$ALL_MODULES" == *"checkpoint-REPLACE"* ]]; then
    echo "ERROR: Set HATE_LORA, MET_LORA, VISUAL_LORA to real adapter paths first." >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$RESULTS_TSV")"
: > "$LOG_FILE"
printf 'config\tdataset\tseed\tsave_num\tcalibration_ratio\tnum_samples\tauc\tacc\n' > "$RESULTS_TSV"

log_msg() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }

DEBUG_ARGS=()
if [[ "$LIMIT_DEBUG" == "1" ]]; then DEBUG_ARGS+=(--DEBUG True); fi

save_counter=0

run_one() {
    local config_name="$1"
    local ratio="$2"
    local dataset="$3"
    local seed="$4"
    local num_samples="$5"   # informational only

    save_counter=$((save_counter + 1))
    local save_num=$((SAVE_BASE + save_counter))

    log_msg ""
    log_msg "============================================================"
    log_msg "[calibration_ratio] ${config_name}"
    log_msg "Dataset: ${dataset}  Seed: ${seed}  ratio=${ratio}  n~${num_samples}  SAVE_NUM: ${save_num}"
    log_msg "============================================================"

    local tmp_log; tmp_log=$(mktemp)

    python src/zh_lora_composition.py \
        --DATASET "$dataset" \
        --PATH "$DATA_ROOT" \
        --base_model "$BASE_MODEL" \
        --lora_dir "$LORA_ROOT" \
        --lora_modules "$ALL_MODULES" \
        --calibration_ratio "$ratio" \
        --max_inference_step 0 \
        --batch_size "$BATCH_SIZE" \
        --cutoff_len "$CUTOFF_LEN" \
        --eval_split "test" \
        --POS_WORD "$POS_WORD" \
        --NEG_WORD "$NEG_WORD" \
        --SEED "$seed" \
        --SAVE_NUM "$save_num" \
        --stdout_only_log \
        --composition_method "moe" \
        --moe_router_lr "$MOE_ROUTER_LR" \
        --moe_router_epochs "$MOE_ROUTER_EPOCHS" \
        --moe_top_k "$MOE_TOP_K" \
        --moe_temperature "$MOE_TEMPERATURE" \
        --moe_router_type "$MOE_ROUTER_TYPE" \
        --moe_router_hidden_size "$MOE_ROUTER_HIDDEN_SIZE" \
        --moe_router_granularity "$MOE_ROUTER_GRANULARITY" \
        --moe_load_balance_coeff "$MOE_LOAD_BALANCE_COEFF" \
        "${DEBUG_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE" "$tmp_log"

    local metric_line
    metric_line=$(grep -E "AUC[[:space:]]+[0-9.]+,[[:space:]]*Acc[[:space:]]+[0-9.]+" \
        "$tmp_log" | tail -n 1 || true)
    if [[ -z "$metric_line" ]]; then
        log_msg "RESULT NOT FOUND: ${config_name} | ${dataset} | seed=${seed}"
        rm -f "$tmp_log"; exit 1
    fi
    local auc acc
    auc=$(printf '%s\n' "$metric_line" | \
        sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\1/')
    acc=$(printf '%s\n' "$metric_line" | \
        sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\2/')
    log_msg "RESULT | ${config_name} | ${dataset} | seed=${seed} | AUC ${auc}, Acc ${acc}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$config_name" "$dataset" "$seed" "$save_num" \
        "$ratio" "$num_samples" "$auc" "$acc" >> "$RESULTS_TSV"
    rm -f "$tmp_log"
}

# Approximate sample counts (toxicn total=9600, sarcnet total=1664)
declare -A TOXICN_N=( [0.05]=480 [0.20]=1920 [0.25]=2400 [0.50]=4800 )
declare -A SARCNET_N=( [0.05]=83  [0.20]=333  [0.25]=416  [0.50]=832  )

for current_seed in $SEEDS; do
    log_msg ""
    log_msg "########################################"
    log_msg "# Seed: ${current_seed}"
    log_msg "########################################"

    for ratio in $RATIOS; do
        pct=$(awk "BEGIN{x=$ratio*100; printf \"%g\", x}")
        config_name="moe_full3_ratio${pct}pct"

        run_one "$config_name" "$ratio" "zh-toxicn-mm" \
            "$current_seed" "${TOXICN_N[$ratio]:-?}"
        run_one "$config_name" "$ratio" "zh-sarcnet" \
            "$current_seed" "${SARCNET_N[$ratio]:-?}"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
python - "$RESULTS_TSV" "$LOG_FILE" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

results_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])

datasets = ["zh-toxicn-mm", "zh-sarcnet"]
grouped = defaultdict(list)   # (config, dataset) -> [auc...]
ratio_map = {}                # config -> ratio string

with results_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        key = (row["config"], row["dataset"])
        grouped[key].append(float(row["auc"]))
        ratio_map[row["config"]] = row["calibration_ratio"]

# Include 16-shot baseline from zh_multiseed_validation_results.tsv
prev_path = results_path.parent / "zh_multiseed_validation_results.tsv"
baseline_config = "moe_full3_shots16_lb0.01 (baseline)"
if prev_path.exists():
    with prev_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["config"] == "moe_mlp_top2_full3_shots16_lb0.01":
                key = (baseline_config, row["dataset"])
                grouped[key].append(float(row["auc"]))
    ratio_map[baseline_config] = "16-shot"

configs = [baseline_config] + sorted(
    {c for c, _ in grouped.keys() if c != baseline_config},
    key=lambda c: float(ratio_map.get(c, "0"))
)

lines = [
    "",
    "=" * 74,
    "In-domain Calibration Ratio Experiment — AUC mean ± std (5 seeds)",
    "  Base config: MoE mlp top-2, full3, lb=0.01, epochs=3",
    "=" * 74,
    f"  {'Config':<36}  {'ToxiCN-MM':>14}  {'SarcNet':>14}  {'Mean':>6}",
    f"  {'-'*36}  {'-'*14}  {'-'*14}  {'-'*6}",
]

for config in configs:
    means = []
    row_parts = [f"  {config:<36}"]
    for ds in datasets:
        vals = grouped.get((config, ds), [])
        if not vals:
            row_parts.append(f"  {'—':>14}")
            continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        means.append(m)
        row_parts.append(f"  {m:5.2f} +/-{s:4.2f}")
    if means:
        row_parts.append(f"  {sum(means)/len(means):6.2f}")
    lines.append("".join(row_parts))

lines.append("")
summary = "\n".join(lines)
print(summary)
with log_path.open("a", encoding="utf-8") as f:
    f.write(summary + "\n")
PY

log_msg ""
log_msg "Results TSV : $RESULTS_TSV"
log_msg "Full log    : $LOG_FILE"
