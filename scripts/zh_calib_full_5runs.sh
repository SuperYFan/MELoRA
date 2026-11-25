#!/bin/bash
# MoE-LoRA 100% 校准实验（calibration_ratio=1.0），5 seeds，汇总 mean±std
#
# 与校准比例实验一致的超参：
#   composition = moe, router = mlp, top_k = 0 (dense), token 粒度
#   moe_router_epochs = 3, moe_router_lr = 1e-3, lb = 0.01
#
# 用法（从项目根目录在 Qwen2.5-7B 服务器运行）：
#   HATE_LORA=zh_hate_semantics/checkpoint-1200 \
#   MET_LORA=zh_met_m3_toxicn_reasoning/checkpoint-1812 \
#   VISUAL_LORA=zh_visual_expression/checkpoint-1332 \
#   bash scripts/zh_calib_full_5runs.sh
#
# 可覆盖的环境变量：
#   SEEDS        (默认 "1111 2222 3333 4444 5555")
#   DATA_ROOT    (默认 data)
#   BASE_MODEL   (默认 models/Qwen2.5-7B)
#   LORA_ROOT    (默认 ./lora_modules)
#   LOG_FILE     合并日志路径（默认 LoRA/zh_calib_full100_5runs.log）
#   RESULT_TSV   结果 TSV 路径（默认 LoRA/zh_calib_full100_results.tsv）
#   APPEND_TSV   1=追加到已有 TSV，0=覆盖（默认 0）

set -euo pipefail

# ── 路径配置 ────────────────────────────────────────────────
DATA_ROOT=${DATA_ROOT:-data}
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
LORA_ROOT=${LORA_ROOT:-./lora_modules}

HATE_LORA=${HATE_LORA:-zh_hate_semantics/checkpoint-REPLACE}
MET_LORA=${MET_LORA:-zh_met_m3_toxicn_reasoning/checkpoint-REPLACE}
VISUAL_LORA=${VISUAL_LORA:-zh_visual_expression/checkpoint-REPLACE}

ALL_MODULES="${HATE_LORA},${MET_LORA},${VISUAL_LORA}"
if [[ "$ALL_MODULES" == *"checkpoint-REPLACE"* ]]; then
    echo "[ERROR] 请先设置 HATE_LORA / MET_LORA / VISUAL_LORA 为真实 checkpoint 路径"
    exit 1
fi

# ── 实验配置 ────────────────────────────────────────────────
SEEDS=${SEEDS:-"1111 2222 3333 4444 5555"}
RATIO="1.0"
TOXICN_N=9600
SARCNET_N=1664
BATCH_SIZE=${BATCH_SIZE:-4}
CUTOFF_LEN=${CUTOFF_LEN:-512}

# save_num 从 30000 起，避免与之前实验（≤28000）覆盖
SAVE_BASE=${SAVE_BASE:-30000}
save_counter=0

LOG_FILE=${LOG_FILE:-LoRA/zh_calib_full100_5runs.log}
RESULT_TSV=${RESULT_TSV:-LoRA/zh_calib_full100_results.tsv}
APPEND_TSV=${APPEND_TSV:-0}

# ── 初始化日志/TSV ───────────────────────────────────────────
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$RESULT_TSV")"
: > "$LOG_FILE"

if [[ "$APPEND_TSV" == "1" ]]; then
    [[ ! -s "$RESULT_TSV" ]] && \
        printf 'config\tdataset\tseed\tsave_num\tcalibration_ratio\tnum_samples\tauc\tacc\n' \
        > "$RESULT_TSV"
else
    printf 'config\tdataset\tseed\tsave_num\tcalibration_ratio\tnum_samples\tauc\tacc\n' \
        > "$RESULT_TSV"
fi

log_msg() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }

# ── 单次运行函数 ─────────────────────────────────────────────
run_once() {
    local seed="$1" dataset="$2" n_samples="$3"
    save_counter=$((save_counter + 1))
    local save_num=$((SAVE_BASE + save_counter))
    local tmp_log; tmp_log=$(mktemp)
    local config="moe_full3_dense_ratio100pct"

    log_msg ""
    log_msg "------------------------------------------------------------"
    log_msg "[calib_ratio=100%] ${dataset} | seed=${seed} | n=${n_samples} | save=${save_num}"
    log_msg "------------------------------------------------------------"

    set +e
    python src/zh_lora_composition.py \
        --DATASET     "$dataset" \
        --PATH        "$DATA_ROOT" \
        --base_model  "$BASE_MODEL" \
        --lora_dir    "$LORA_ROOT" \
        --lora_modules "$ALL_MODULES" \
        --calibration_ratio "$RATIO" \
        --max_inference_step 0 \
        --batch_size  "$BATCH_SIZE" \
        --cutoff_len  "$CUTOFF_LEN" \
        --eval_split  "test" \
        --POS_WORD    "否" \
        --NEG_WORD    "是" \
        --SEED        "$seed" \
        --SAVE_NUM    "$save_num" \
        --stdout_only_log \
        --composition_method   "moe" \
        --moe_router_lr        1e-3 \
        --moe_router_epochs    3 \
        --moe_top_k            0 \
        --moe_temperature      1.0 \
        --moe_router_type      "mlp" \
        --moe_router_hidden_size 128 \
        --moe_router_granularity "token" \
        --moe_load_balance_coeff 0.01 \
        2>&1 | tee -a "$LOG_FILE" "$tmp_log"
    set -e

    local metric_line auc acc
    metric_line=$(grep -E "AUC[[:space:]]+[0-9.]+,[[:space:]]*Acc[[:space:]]+[0-9.]+" \
        "$tmp_log" | tail -n 1 || true)
    if [[ -z "$metric_line" ]]; then
        log_msg "RESULT NOT FOUND: ${config} | ${dataset} | seed=${seed}"
        rm -f "$tmp_log"; exit 1
    fi
    auc=$(printf '%s\n' "$metric_line" | \
        sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\1/')
    acc=$(printf '%s\n' "$metric_line" | \
        sed -E 's/.*AUC[[:space:]]+([0-9.]+),[[:space:]]*Acc[[:space:]]+([0-9.]+).*/\2/')
    log_msg "RESULT | ${config} | ${dataset} | seed=${seed} | AUC ${auc}, Acc ${acc}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$config" "$dataset" "$seed" "$save_num" \
        "$RATIO" "$n_samples" "$auc" "$acc" \
        >> "$RESULT_TSV"
    rm -f "$tmp_log"
}

# ── 主循环 ───────────────────────────────────────────────────
log_msg "========================================"
log_msg "# MoE-LoRA 100% 校准实验 (5 seeds × 2 datasets)"
log_msg "# calibration_ratio=1.0  router=mlp  top_k=0  epochs=3  lb=0.01"
log_msg "========================================"

for seed in $SEEDS; do
    log_msg ""
    log_msg "######## Seed: ${seed} ########"
    run_once "$seed" "zh-toxicn-mm" "$TOXICN_N"
    run_once "$seed" "zh-sarcnet"   "$SARCNET_N"
done

log_msg ""
log_msg "========================================"
log_msg "所有 seed 完成，结果写入: ${RESULT_TSV}"
log_msg "========================================"

# ── Python 汇总（mean±std）────────────────────────────────────
python - "$RESULT_TSV" "$LOG_FILE" <<'PY'
import csv, statistics, sys
from pathlib import Path

tsv_path   = Path(sys.argv[1])
log_path   = sys.argv[2]

def emit(msg=""):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

rows = list(csv.DictReader(open(tsv_path), delimiter="\t"))
datasets = ["zh-toxicn-mm", "zh-sarcnet"]

emit()
emit("=" * 60)
emit("MoE-LoRA 100% 校准实验 5-run Summary")
emit("=" * 60)

for ds in datasets:
    ds_rows = [r for r in rows if r["dataset"] == ds]
    aucs = [float(r["auc"]) for r in ds_rows]
    accs = [float(r["acc"]) for r in ds_rows]
    for r in ds_rows:
        emit(f"  {ds} seed={r['seed']}: AUC {float(r['auc']):.4f}, Acc {float(r['acc']):.2f}%")
    if aucs:
        mean_auc = statistics.mean(aucs)
        std_auc  = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        mean_acc = statistics.mean(accs)
        std_acc  = statistics.stdev(accs) if len(accs) > 1 else 0.0
        emit(f"SUMMARY | {ds} | runs={len(aucs)} | AUC {mean_auc:.2f} ± {std_auc:.2f}, Acc {mean_acc:.2f} ± {std_acc:.2f}%")
    else:
        emit(f"SUMMARY | {ds} | no data")

emit()
PY
