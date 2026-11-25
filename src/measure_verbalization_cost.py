"""Measure the visual verbalization cost of Qwen3-VL-8B-Instruct.

Generates image descriptions with the label-agnostic prompt used in the paper
(Appendix A, Table 8) for the first N test memes, using greedy decoding, and reports
the average latency per image and the peak GPU memory. The descriptions are only
used for timing; they are written to a separate file and never overwrite the cached
descriptions used in the experiments.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

PROMPT = "请只根据图片本身生成一段中文图片说明。必须简要描述画面主体、可见文字、人物或物体关系、表情情绪等内容。只输出图片描述正文。"
BENCHMARKS = {"zh-toxicn-mm": "toxicn_mm", "zh-sarcnet": "sarcnet"}


def gb(num_bytes):
    return round(num_bytes / 1024 ** 3, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Path to Qwen3-VL-8B-Instruct")
    parser.add_argument("--repo_root", default=".", help="Mod_HATE-main root (image paths are relative to it)")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset", default="zh-toxicn-mm", choices=list(BENCHMARKS))
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    repo = Path(args.repo_root)
    test_file = repo / args.data_root / "zh_modular" / "benchmarks" / BENCHMARKS[args.dataset] / "test.json"
    rows = json.loads(test_file.read_text(encoding="utf-8-sig"))
    rows = rows[: args.num_images + args.num_warmup]

    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        raise SystemExit(
            f"transformers {transformers.__version__} does not support Qwen3-VL. "
            "Run this step in the environment used for Qwen3-VL (transformers >= 4.57), e.g.\n"
            "  RUN_ROUTER=0 PYTHON_VLM=/path/to/qwen3vl-env/bin/python bash scripts/run_cost_measurement.sh")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = model_cls.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    latencies, outputs = [], []
    torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(rows):
        image = Image.open(repo / row["image"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        enc = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt")
        enc.pop("token_type_ids", None)
        enc = {k: v.to(model.device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = generated[0, enc["input_ids"].shape[1]:]
        text = processor.decode(new_tokens, skip_special_tokens=True)
        if i >= args.num_warmup:
            latencies.append(elapsed)
            outputs.append({"id": row["id"], "seconds": round(elapsed, 3),
                            "new_tokens": int(new_tokens.numel()), "description": text})

    summary = {
        "dataset": args.dataset,
        "gpu": torch.cuda.get_device_name(0),
        "num_images": len(latencies),
        "mean_seconds_per_image": round(statistics.mean(latencies), 3),
        "median_seconds_per_image": round(statistics.median(latencies), 3),
        "mean_new_tokens": round(statistics.mean(o["new_tokens"] for o in outputs), 1),
        "peak_gpu_memory_gb": gb(torch.cuda.max_memory_allocated()),
    }
    out_dir = repo / "LoRA" / "cost_measurement"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"verbalization_{args.dataset}.json").write_text(
        json.dumps({"summary": summary, "samples": outputs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[COST] verbalization:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
