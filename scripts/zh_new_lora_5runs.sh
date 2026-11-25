#!/bin/bash
# Run Chinese LoRA composition benchmarks with five fixed seeds and aggregate mean/std.
#
# Default behavior is MoE-LoRA because this script is intended for the new route-B
# experiments. Set COMPOSITION_METHOD=static if you want the old static baseline.
#
# Example:
#   COMPOSITION_METHOD=moe \
#   HATE_LORA=zh_hate_semantics/checkpoint-1200 \
#   MET_LORA=zh_met_m3_toxicn_reasoning/checkpoint-300 \
#   VISUAL_LORA=zh_visual_expression/checkpoint-1332 \
#   bash scripts/zh_new_lora_5runs.sh

set -euo pipefail

SEEDS=${SEEDS:-"1111 2222 3333 4444 5555"}
COMPOSITION_METHOD=${COMPOSITION_METHOD:-moe}
NUM_SHOTS=${NUM_SHOTS:-4}
NEW_LORA_COMBINED_LOG=${NEW_LORA_COMBINED_LOG:-}
SAVE_MEAN_SUMMARY_FILES=${SAVE_MEAN_SUMMARY_FILES:-0}
# MoE router settings (forwarded to zh_new_lora.sh)
MOE_ROUTER_TYPE=${MOE_ROUTER_TYPE:-linear}
MOE_ROUTER_LR=${MOE_ROUTER_LR:-1e-3}
MOE_ROUTER_EPOCHS=${MOE_ROUTER_EPOCHS:-20}
MOE_ROUTER_MAX_STEPS=${MOE_ROUTER_MAX_STEPS:-0}
MOE_TOP_K=${MOE_TOP_K:-0}
MOE_TEMPERATURE=${MOE_TEMPERATURE:-1.0}
MOE_ROUTER_HIDDEN_SIZE=${MOE_ROUTER_HIDDEN_SIZE:-128}
MOE_ROUTER_GRANULARITY=${MOE_ROUTER_GRANULARITY:-token}

log_msg() {
    if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
        printf '%s\n' "$*" | tee -a "$NEW_LORA_COMBINED_LOG"
    else
        printf '%s\n' "$*"
    fi
}

# Keep these logs in separate directories.
# Each run writes:
#   toxicn:   101 + SAVE_OFFSET
#   sarcnet:  103 + SAVE_OFFSET
SAVE_BASE_OFFSET=${SAVE_BASE_OFFSET:-500}
SAVE_OFFSET_STRIDE=${SAVE_OFFSET_STRIDE:-10}

if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
    mkdir -p "$(dirname "$NEW_LORA_COMBINED_LOG")"
    : > "$NEW_LORA_COMBINED_LOG"
    log_msg "Combined log: $NEW_LORA_COMBINED_LOG"
fi

run_index=0
for current_seed in $SEEDS; do
    current_offset=$((SAVE_BASE_OFFSET + run_index * SAVE_OFFSET_STRIDE))
    log_msg ""
    log_msg "============================================================"
    log_msg "Repeated LoRA run $((run_index + 1))"
    log_msg "SEED=$current_seed"
    log_msg "SAVE_OFFSET=$current_offset"
    log_msg "COMPOSITION_METHOD=$COMPOSITION_METHOD"
    log_msg "NUM_SHOTS=$NUM_SHOTS per class"
    log_msg "MOE_ROUTER_TYPE=$MOE_ROUTER_TYPE"
    log_msg "============================================================"

    SEED="$current_seed" \
    SAVE_OFFSET="$current_offset" \
    COMPOSITION_METHOD="$COMPOSITION_METHOD" \
    NUM_SHOTS="$NUM_SHOTS" \
    MOE_ROUTER_TYPE="$MOE_ROUTER_TYPE" \
    MOE_ROUTER_LR="$MOE_ROUTER_LR" \
    MOE_ROUTER_EPOCHS="$MOE_ROUTER_EPOCHS" \
    MOE_ROUTER_MAX_STEPS="$MOE_ROUTER_MAX_STEPS" \
    MOE_TOP_K="$MOE_TOP_K" \
    MOE_TEMPERATURE="$MOE_TEMPERATURE" \
    MOE_ROUTER_HIDDEN_SIZE="$MOE_ROUTER_HIDDEN_SIZE" \
    MOE_ROUTER_GRANULARITY="$MOE_ROUTER_GRANULARITY" \
    NEW_LORA_COMBINED_LOG="$NEW_LORA_COMBINED_LOG" \
    NEW_LORA_LOG_APPEND="1" \
    bash scripts/zh_new_lora.sh

    run_index=$((run_index + 1))
done

python - "$COMPOSITION_METHOD" "$NUM_SHOTS" "$SAVE_BASE_OFFSET" "$SAVE_OFFSET_STRIDE" "$SEEDS" "$NEW_LORA_COMBINED_LOG" "$SAVE_MEAN_SUMMARY_FILES" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

composition_method = sys.argv[1]
num_shots = int(sys.argv[2])
save_base_offset = int(sys.argv[3])
save_offset_stride = int(sys.argv[4])
seeds = sys.argv[5].split()
combined_log = sys.argv[6]
save_mean_summary_files = sys.argv[7] == "1"

datasets = [
    ("zh-toxicn-mm", 101),
    ("zh-sarcnet", 103),
]
metric_pattern = re.compile(r"AUC\s+([0-9.]+),\s*Acc\s+([0-9.]+)")
summary_pattern = re.compile(
    r"^SUMMARY \|\s*(?P<dataset>[^|]+?)\s*\|\s*seed=(?P<seed>\d+)\s*\|\s*save_num=(?P<save_num>\d+)\s*\|\s*"
    r"AUC\s+(?P<auc>[0-9.]+),\s*Acc\s+(?P<acc>[0-9.]+)",
    re.MULTILINE,
)


def emit(message=""):
    print(message)
    if combined_log:
        with open(combined_log, "a", encoding="utf-8") as file_obj:
            file_obj.write(message + "\n")


all_runs = []
summary = {
    "composition_method": composition_method,
    "num_shots_per_class": num_shots,
    "total_fewshot_examples": num_shots * 2,
    "seeds": [int(seed) for seed in seeds],
    "save_base_offset": save_base_offset,
    "save_offset_stride": save_offset_stride,
    "datasets": {},
}

combined_entries = {}
if combined_log:
    combined_text = Path(combined_log).read_text(encoding="utf-8", errors="ignore")
    for match in summary_pattern.finditer(combined_text):
        entry = {
            "dataset": match.group("dataset").strip(),
            "seed": int(match.group("seed")),
            "save_num": int(match.group("save_num")),
            "auc": float(match.group("auc")),
            "acc": float(match.group("acc")),
        }
        combined_entries[(entry["dataset"], entry["seed"], entry["save_num"])] = entry

for dataset, base_save_num in datasets:
    auc_values = []
    acc_values = []
    save_nums = []
    for run_index, seed in enumerate(seeds):
        save_num = base_save_num + save_base_offset + run_index * save_offset_stride
        save_nums.append(save_num)
        if combined_log:
            entry = combined_entries.get((dataset, int(seed), save_num))
            if entry is None:
                raise ValueError(
                    f"Cannot find SUMMARY line for dataset={dataset}, seed={seed}, save_num={save_num} in {combined_log}"
                )
            auc = entry["auc"]
            acc = entry["acc"]
            log_path = combined_log
        else:
            log_path = Path("LoRA") / f"{dataset}_{save_num}.txt"
            if not log_path.exists():
                raise FileNotFoundError(f"Missing log file: {log_path}")
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            matches = metric_pattern.findall(text)
            if not matches:
                raise ValueError(f"Cannot parse AUC/Acc from: {log_path}")
            auc, acc = (float(value) for value in matches[-1])
        auc_values.append(auc)
        acc_values.append(acc)
        all_runs.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "save_num": save_num,
                "log_path": str(log_path),
                "auc": auc,
                "acc": acc,
            }
        )

    summary["datasets"][dataset] = {
        "save_nums": save_nums,
        "auc_mean": round(statistics.mean(auc_values), 4),
        "auc_std": round(statistics.stdev(auc_values), 4) if len(auc_values) > 1 else 0.0,
        "acc_mean": round(statistics.mean(acc_values), 4),
        "acc_std": round(statistics.stdev(acc_values), 4) if len(acc_values) > 1 else 0.0,
        "runs": [
            row for row in all_runs
            if row["dataset"] == dataset
        ],
    }

json_path = None
csv_path = None
if (not combined_log) or save_mean_summary_files:
    out_prefix = Path("LoRA") / f"zh_new_lora_{composition_method}_shots{num_shots}_offset{save_base_offset}"
    json_path = out_prefix.with_suffix(".mean_summary.json")
    csv_path = out_prefix.with_suffix(".mean_summary.csv")

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "dataset",
                "auc_mean",
                "auc_std",
                "acc_mean",
                "acc_std",
                "seeds",
                "save_nums",
            ],
        )
        writer.writeheader()
        for dataset, values in summary["datasets"].items():
            writer.writerow(
                {
                    "dataset": dataset,
                    "auc_mean": values["auc_mean"],
                    "auc_std": values["auc_std"],
                    "acc_mean": values["acc_mean"],
                    "acc_std": values["acc_std"],
                    "seeds": " ".join(seeds),
                    "save_nums": " ".join(str(num) for num in values["save_nums"]),
                }
            )

emit("\nMean summary:")
for dataset, values in summary["datasets"].items():
    emit(
        f"{dataset}: "
        f"AUC {values['auc_mean']:.4f} +/- {values['auc_std']:.4f}, "
        f"Acc {values['acc_mean']:.4f} +/- {values['acc_std']:.4f}"
    )
if json_path is not None and csv_path is not None:
    emit(f"\nSaved: {json_path}")
    emit(f"Saved: {csv_path}")
PY

log_msg ""
if [[ -n "$NEW_LORA_COMBINED_LOG" ]]; then
    log_msg "Repeated LoRA runs finished. Combined process+result log saved to: $NEW_LORA_COMBINED_LOG"
else
    log_msg "Repeated LoRA runs finished."
fi
