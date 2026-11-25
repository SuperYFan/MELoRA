"""Measure the computational cost of MELoRA without modifying the original code.

This script accepts exactly the same command-line arguments as src/zh_lora_composition.py.
It wraps the router-calibration and inference functions at runtime and records:
  - trainable router parameters,
  - router calibration time and peak GPU memory,
  - inference time (total and per sample) and peak GPU memory.
Results are printed with the prefix [COST] and saved to LoRA/cost_measurement/.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import lora_config
import lora_learning
import moe_lora

COST = {}


def gb(num_bytes):
    return round(num_bytes / 1024 ** 3, 2)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_lora_path(lora_dir, module):
    module_path = Path(module)
    if module_path.is_absolute() or module_path.exists():
        return str(module_path)
    return str(Path(lora_dir) / module)


def count_test_samples(args):
    folder = {"zh-toxicn-mm": "toxicn_mm", "zh-sarcnet": "sarcnet"}.get(args.DATASET)
    if folder is None:
        return None
    path = Path(args.PATH) / "zh_modular" / "benchmarks" / folder / f"{args.eval_split}.json"
    try:
        with open(path, encoding="utf-8-sig") as f:
            return len(json.load(f))
    except Exception:
        return None


# ---------- wrap router calibration ----------
_original_train = moe_lora.train_moe_routers


def timed_train_moe_routers(*a, **kw):
    model = kw.get("model", a[0] if a else None)
    if model is not None:
        COST["router_trainable_parameters"] = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad))
        COST["total_parameters"] = int(sum(p.numel() for p in model.parameters()))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = _original_train(*a, **kw)
    torch.cuda.synchronize()
    COST["router_calibration_seconds"] = round(time.perf_counter() - start, 1)
    COST["router_calibration_peak_gpu_memory_gb"] = gb(torch.cuda.max_memory_allocated())
    print(f"[COST] router calibration: {COST['router_calibration_seconds']} s, "
          f"peak GPU memory: {COST['router_calibration_peak_gpu_memory_gb']} GB, "
          f"trainable parameters: {COST.get('router_trainable_parameters')}", flush=True)
    return result


moe_lora.train_moe_routers = timed_train_moe_routers


# ---------- wrap inference ----------
_original_generation = lora_learning.zh_modular_generation


def timed_zh_modular_generation(*a, **kw):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = _original_generation(*a, **kw)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    COST["inference_seconds"] = round(seconds, 1)
    COST["inference_peak_gpu_memory_gb"] = gb(torch.cuda.max_memory_allocated())
    n = COST.get("num_test_samples")
    if n:
        COST["inference_ms_per_sample"] = round(1000 * seconds / n, 1)
    print(f"[COST] inference: {COST['inference_seconds']} s for {n} samples "
          f"({COST.get('inference_ms_per_sample')} ms/sample), "
          f"peak GPU memory: {COST['inference_peak_gpu_memory_gb']} GB", flush=True)
    return result


lora_learning.zh_modular_generation = timed_zh_modular_generation


if __name__ == "__main__":
    args = lora_config.parse_opt()
    set_seed(args.SEED)
    COST.update({
        "dataset": args.DATASET,
        "seed": args.SEED,
        "num_shots_per_class": args.num_shots,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gb": gb(torch.cuda.get_device_properties(0).total_memory),
        "num_test_samples": count_test_samples(args),
    })
    modules = [m.strip() for m in args.lora_modules.split(",") if m.strip()]
    lora_module_list = [resolve_lora_path(args.lora_dir, m) for m in modules]

    wall_start = time.perf_counter()
    lora_learning.lorahub_learning(
        lora_module_list,
        args,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size,
    )
    COST["total_wall_seconds"] = round(time.perf_counter() - wall_start, 1)

    out_dir = Path("LoRA/cost_measurement")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.DATASET}_shots{args.num_shots}_seed{args.SEED}.json"
    out_path.write_text(json.dumps(COST, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[COST] summary:", json.dumps(COST, ensure_ascii=False), flush=True)
    print(f"[COST] saved to {out_path}", flush=True)
