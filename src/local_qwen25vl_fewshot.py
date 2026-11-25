"""Run the Chinese few-shot benchmarks with a directly loaded Qwen2.5-VL."""

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import api_fewshot_baseline as benchmark


DTYPES = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class DirectQwen25VLPredictor:
    """Keep one local model instance alive for the complete experiment sweep."""

    def __init__(
        self,
        model_path: Path,
        dtype: str,
        device_map: str,
        attn_implementation: str,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local model is incomplete or missing: {model_path}")

        model_kwargs: dict[str, Any] = {
            "torch_dtype": DTYPES[dtype],
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation

        print(f"Loading local model once: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            **model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            local_files_only=True,
        )
        self.input_device = self.model.device
        print(f"Local model ready; input device: {self.input_device}")

    @staticmethod
    def _decode_image_url(url: str) -> Image.Image:
        if not url.startswith("data:") or "," not in url:
            raise ValueError("Direct inference expects an embedded image data URL")
        _, encoded = url.split(",", 1)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            return image.convert("RGB").copy()

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        converted = []
        images = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                converted.append({"role": message["role"], "content": content})
                continue

            converted_content = []
            for item in content:
                if item.get("type") == "image_url":
                    url = str((item.get("image_url") or {}).get("url") or "")
                    images.append(self._decode_image_url(url))
                    converted_content.append({"type": "image"})
                elif item.get("type") == "text":
                    converted_content.append(
                        {"type": "text", "text": str(item.get("text") or "")}
                    )
                else:
                    raise ValueError(f"Unsupported message content item: {item}")
            converted.append(
                {"role": message["role"], "content": converted_content}
            )
        return converted, images

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        converted, images = self._prepare_messages(messages)
        try:
            prompt = self.processor.apply_chat_template(
                converted,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt],
                images=images or None,
                padding=True,
                return_tensors="pt",
            ).to(self.input_device)
        finally:
            for image in images:
                image.close()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def predict(
        self,
        messages: list[dict[str, Any]],
        args: argparse.Namespace,
    ) -> tuple[str, int, float, str]:
        raw_response = self.generate(messages, max_new_tokens=args.max_tokens)
        pred_label, positive_probability, parsed_label_text = benchmark.parse_prediction(
            raw_response,
            label_0_word=args.label_0_word,
            label_1_word=args.label_1_word,
        )
        return raw_response, pred_label, positive_probability, parsed_label_text


def build_run_args(
    cli_args: argparse.Namespace,
    dataset: str,
    num_shots: int,
    seed: int,
) -> argparse.Namespace:
    input_source = (
        cli_args.sarcnet_input_source
        if dataset == "zh-sarcnet"
        else cli_args.input_source
    )
    return SimpleNamespace(
        dataset=dataset,
        data_root=cli_args.data_root,
        repo_root=cli_args.repo_root,
        output_dir=cli_args.output_dir,
        eval_split=benchmark.BENCHMARKS[dataset]["default_eval_split"],
        num_shots=num_shots,
        shot_mode=cli_args.shot_mode,
        input_source=input_source,
        seed=seed,
        limit=cli_args.limit,
        model=cli_args.model_name,
        api_url=f"local-transformers://{cli_args.model_path}",
        api_key="local-model",
        temperature=0.0,
        max_tokens=cli_args.max_new_tokens,
        output_retries=0,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        retry_failed=True,
        record_failed=cli_args.record_failed,
        sleep=0.0,
        progress_every=cli_args.progress_every,
        label_0_word="否",
        label_1_word="是",
        use_image=False,
        max_image_side=cli_args.max_image_side,
        jpeg_quality=cli_args.jpeg_quality,
        json_mode=False,
        resume=cli_args.resume,
    )


def aggregate_runs(
    output_dir: Path,
    dataset: str,
    model_name: str,
    shot_mode: str,
    num_shots: int,
    input_source: str,
    seeds: list[int],
) -> None:
    eval_split = benchmark.BENCHMARKS[dataset]["default_eval_split"]
    safe_model = benchmark.safe_name(model_name)
    model_dir = output_dir / dataset / safe_model
    rows = []
    missing = []
    for seed in seeds:
        run_name = (
            f"{dataset}_{safe_model}_{shot_mode}{num_shots}_"
            f"seed{seed}_{eval_split}_{input_source}"
        )
        summary_path = model_dir / f"{run_name}.summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        row = benchmark.read_json(summary_path)
        row["seed"] = seed
        rows.append(row)

    if missing:
        raise FileNotFoundError("Missing summary files:\n" + "\n".join(missing))

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    def std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        average = mean(values)
        return math.sqrt(
            sum((value - average) ** 2 for value in values) / (len(values) - 1)
        )

    aggregate: dict[str, Any] = {
        "dataset": dataset,
        "eval_split": eval_split,
        "model": model_name,
        "backend": "direct-transformers",
        "num_shots": num_shots,
        "shot_mode": shot_mode,
        "input_source": input_source,
        "seeds": seeds,
        "num_runs": len(rows),
        "runs": rows,
    }
    for metric in ("auc", "accuracy", "num_invalid"):
        values = [float(row[metric]) for row in rows]
        aggregate[f"{metric}_mean"] = round(mean(values), 4)
        aggregate[f"{metric}_std"] = round(std(values), 4)

    aggregate_name = (
        f"{dataset}_{safe_model}_{shot_mode}{num_shots}_{len(rows)}runs_"
        f"{eval_split}_{input_source}"
    )
    summary_out = model_dir / f"{aggregate_name}.mean_summary.json"
    csv_out = model_dir / f"{aggregate_name}.mean_summary.csv"
    benchmark.write_json(summary_out, aggregate)

    fieldnames = [
        "seed",
        "auc",
        "accuracy",
        "num_invalid",
        "num_eval",
        "output_jsonl",
        "output_csv",
        "shots_path",
    ]
    with csv_out.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        writer.writerow(
            {
                "seed": "mean",
                "auc": aggregate["auc_mean"],
                "accuracy": aggregate["accuracy_mean"],
                "num_invalid": aggregate["num_invalid_mean"],
            }
        )
        writer.writerow(
            {
                "seed": "std",
                "auc": aggregate["auc_std"],
                "accuracy": aggregate["accuracy_std"],
                "num_invalid": aggregate["num_invalid_std"],
            }
        )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct local Qwen2.5-VL few-shot benchmark sweep"
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("local_qwen25vl_fewshot_results")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["zh-toxicn-mm", "zh-sarcnet"],
        default=["zh-toxicn-mm", "zh-sarcnet"],
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1111, 2222, 3333, 4444, 5555]
    )
    parser.add_argument("--shot_mode", choices=["per_class", "total"], default="per_class")
    parser.add_argument("--input_source", default="image")
    parser.add_argument("--sarcnet_input_source", default="image_text")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--attn_implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa"
    )
    parser.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--max_image_side", type=int, default=512)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument(
        "--record_failed", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.repo_root = args.repo_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    predictor = DirectQwen25VLPredictor(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    for num_shots in args.shots:
        for dataset in args.datasets:
            input_source = (
                args.sarcnet_input_source
                if dataset == "zh-sarcnet"
                else args.input_source
            )
            for seed in args.seeds:
                torch.manual_seed(seed)
                run_args = build_run_args(args, dataset, num_shots, seed)
                benchmark.evaluate(run_args, prediction_fn=predictor.predict)
            aggregate_runs(
                output_dir=args.output_dir,
                dataset=dataset,
                model_name=args.model_name,
                shot_mode=args.shot_mode,
                num_shots=num_shots,
                input_source=input_source,
                seeds=args.seeds,
            )


if __name__ == "__main__":
    main()
