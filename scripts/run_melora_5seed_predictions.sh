#!/bin/bash
# Run MELoRA (dense, MLP router, 3 experts) for all seeds/budgets and keep
# per-sample predictions + router gate weights.
#
# The RUN name is built INSIDE the seed loop, so every (seed, dataset, K)
# gets its own directory and nothing is overwritten.
# Writes to a fresh root so the original pipeline outputs are never touched.
#
# Usage (from repo root):
#   bash scripts/run_melora_5seed_predictions.sh
#   SEEDS="1111 2222" bash scripts/run_melora_5seed_predictions.sh   # subset
#   DRY_RUN=1 bash scripts/run_melora_5seed_predictions.sh           # print plan only

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data}
LORA_ROOT=${LORA_ROOT:-./lora_modules}
OUT=${OUT:-LoRA/melora_5seed_preds}
SEEDS=${SEEDS:-"1111 2222 3333 4444 5555"}
BUDGETS=${BUDGETS:-"4 8 16 32"}
DATASETS=${DATASETS:-"zh-toxicn-mm zh-sarcnet"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${OUT}"
STATUS=${OUT}/status.tsv
[[ -f "${STATUS}" ]] || printf 'seed\tdataset\tK\trun_dir\tstatus\n' > "${STATUS}"

for SEED in ${SEEDS}; do
  for ds in ${DATASETS}; do
    case "${ds}" in
      zh-toxicn-mm)
        MET_LORA=zh_met_m3_toxicn_reasoning_20260917_231740/checkpoint-1848
        TAG=toxicn_clean ;;
      zh-sarcnet)
        MET_LORA=zh_met_m3_sarcnet_reasoning_20260917_231740/checkpoint-2010
        TAG=sarcnet_full ;;
      *) echo "unknown dataset ${ds}" >&2; exit 1 ;;
    esac

    for K in ${BUDGETS}; do
      N=$((K / 2))   # --num_shots is per class
      if [[ "${K}" == "32" ]]; then
        RUN="moe_mlp_top3_shots16_seed${SEED}_${TAG}"
      else
        RUN="moe_mlp_top3_totalshots${K}_seed${SEED}_${TAG}"
      fi
      RUN_ROOT="${OUT}/${RUN}"

      # sanity: seed must be in the directory name
      [[ "${RUN}" == *"_seed${SEED}_"* ]] || { echo "BUG: seed missing in ${RUN}" >&2; exit 1; }

      if compgen -G "${RUN_ROOT}/predictions/*/eval_predictions.jsonl" > /dev/null; then
        echo "SKIP  seed=${SEED} ds=${ds} K=${K}  (exists)"
        continue
      fi

      echo "RUN   seed=${SEED} ds=${ds} K=${K}  -> ${RUN_ROOT}"
      if [[ "${DRY_RUN}" == "1" ]]; then continue; fi

      if SEED="${SEED}" NUM_SHOTS="${N}" MOE_TOP_K=3 MOE_LOAD_BALANCE_COEFF=0.0 \
         DATA_ROOT="${DATA_ROOT}" BASE_MODEL="${BASE_MODEL}" LORA_ROOT="${LORA_ROOT}" \
         MET_LORA="${MET_LORA}" DATASETS="${ds}" \
         RUN_NAME="${RUN}" OUTPUT_ROOT="${RUN_ROOT}" \
         PREDICTION_DIR="${RUN_ROOT}/predictions" \
         ROUTER_DIR="${RUN_ROOT}/routers" \
         ROUTER_ANALYSIS_DIR="${RUN_ROOT}/router_analysis" \
         bash scripts/zh_reproduce_moe_top3_16shot.sh; then
        printf '%s\t%s\t%s\t%s\tok\n' "${SEED}" "${ds}" "${K}" "${RUN_ROOT}" >> "${STATUS}"
      else
        printf '%s\t%s\t%s\t%s\tFAILED\n' "${SEED}" "${ds}" "${K}" "${RUN_ROOT}" >> "${STATUS}"
        echo "FAILED seed=${SEED} ds=${ds} K=${K}; continuing" >&2
      fi
    done
  done
done

echo
echo "Prediction files found:"
find "${OUT}" -name eval_predictions.jsonl | wc -l
echo "(expected: $(( $(wc -w <<< "${SEEDS}") * $(wc -w <<< "${DATASETS}") * $(wc -w <<< "${BUDGETS}") )))"
echo "Per-sample predictions and the AUC / ACC printed by the evaluation are in ${OUT}"
