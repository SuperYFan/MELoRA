from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

from api_fewshot_baseline import (
    BENCHMARKS,
    InvalidPredictionError,
    append_jsonl,
    binary_auc,
    build_fewshot_rows,
    label_word,
    load_completed_predictions,
    parse_prediction,
    read_json,
    resolve_image_path,
    resolve_split_path,
    row_input,
    safe_name,
    save_csv,
    write_json,
)


DEFAULT_MODEL = "OpenGVLab/InternVL3_5-8B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

torch = None
T = None
Image = None
InterpolationMode = None
AutoModel = None
AutoTokenizer = None
MsAutoModel = None
MsAutoTokenizer = None


def ensure_runtime_deps() -> None:
    global torch, T, Image, InterpolationMode, AutoModel, AutoTokenizer, MsAutoModel, MsAutoTokenizer
    if torch is not None:
        return
    try:
        import torch as torch_module
        import torchvision.transforms as transforms_module
        from PIL import Image as image_module
        from torchvision.transforms.functional import InterpolationMode as interpolation_mode_module
        from transformers import AutoModel as auto_model_module
        from transformers import AutoTokenizer as auto_tokenizer_module
        from transformers.modeling_utils import PreTrainedModel as pretrained_model_class
    except ImportError as exc:
        raise ImportError(
            "Missing runtime dependency for local InternVL inference. "
            "Install at least: torch, torchvision, pillow, transformers>=4.52.1, accelerate."
        ) from exc
    try:
        from modelscope import AutoModel as ms_auto_model_module
        from modelscope import AutoTokenizer as ms_auto_tokenizer_module
    except ImportError:
        ms_auto_model_module = None
        ms_auto_tokenizer_module = None
    torch = torch_module
    T = transforms_module
    Image = image_module
    InterpolationMode = interpolation_mode_module
    AutoModel = auto_model_module
    AutoTokenizer = auto_tokenizer_module
    MsAutoModel = ms_auto_model_module
    MsAutoTokenizer = ms_auto_tokenizer_module

    if not hasattr(pretrained_model_class, "all_tied_weights_keys"):
        def _infer_tied_weight_source_name(model):
            try:
                input_embeddings = model.get_input_embeddings()
            except Exception:
                input_embeddings = None
            input_weight = getattr(input_embeddings, "weight", None)
            if input_weight is None:
                return None
            try:
                named_parameters = model.named_parameters(remove_duplicate=False)
            except TypeError:
                named_parameters = model.named_parameters()
            for name, param in named_parameters:
                if param is input_weight:
                    return name
            return None

        def _normalize_tied_weights_keys(model, value):
            if value is None:
                return {}
            if isinstance(value, dict):
                return value
            if hasattr(value, "items"):
                return {str(key): str(source) for key, source in value.items()}
            if isinstance(value, (list, tuple, set)):
                items = list(value)
                if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in items):
                    return {str(key): str(source) for key, source in items}
                if all(isinstance(item, str) for item in items):
                    source_name = _infer_tied_weight_source_name(model)
                    return {str(key): str(source_name or key) for key in items}
            return {}

        def _get_all_tied_weights_keys(self):
            stored = self.__dict__.get("_all_tied_weights_keys")
            if isinstance(stored, dict):
                return stored
            normalized = _normalize_tied_weights_keys(self, stored)
            if stored is None:
                normalized = _normalize_tied_weights_keys(self, getattr(self, "_tied_weights_keys", None))
            self.__dict__["_all_tied_weights_keys"] = normalized
            return normalized

        def _set_all_tied_weights_keys(self, value):
            self.__dict__["_all_tied_weights_keys"] = _normalize_tied_weights_keys(self, value)

        pretrained_model_class.all_tied_weights_keys = property(
            _get_all_tied_weights_keys,
            _set_all_tied_weights_keys,
        )


def is_dynamic_module_missing(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, FileNotFoundError) and "transformers_modules" in message


def hydrate_dynamic_module_cache(model_dir: Path, exc: Exception) -> bool:
    missing_filename = getattr(exc, "filename", None)
    if not missing_filename:
        return False

    missing_path = Path(missing_filename)
    if "transformers_modules" not in str(missing_path):
        return False
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    source_root = model_dir.resolve()
    if not source_root.exists() or not source_root.is_dir():
        return False

    cache_dir = missing_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    copied = False
    for current_root, _, filenames in os.walk(source_root, followlinks=True):
        current_root_path = Path(current_root)
        relative_root = current_root_path.relative_to(source_root)
        target_root = cache_dir if str(relative_root) == "." else cache_dir / relative_root
        target_root.mkdir(parents=True, exist_ok=True)

        init_file = target_root / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            copied = True

        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            source_file = current_root_path / filename
            target_file = target_root / filename
            shutil.copy2(source_file, target_file)
            copied = True

    importlib.invalidate_caches()
    return copied


def build_transform(input_size: int):
    ensure_runtime_deps()
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image,
    min_num: int = 1,
    max_num: int = 1,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_tensor(image_path: Path, input_size: int, max_tiles: int) -> torch.Tensor:
    ensure_runtime_deps()
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_tiles,
        use_thumbnail=True,
    )
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def resolve_dtype(dtype_name: str):
    ensure_runtime_deps()
    normalized = dtype_name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    ensure_runtime_deps()
    dtype = resolve_dtype(args.torch_dtype)
    local_model_dir = Path(args.model_path).expanduser()

    def load_with_transformers(current_kwargs):
        model = AutoModel.from_pretrained(args.model_path, **current_kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        return model, tokenizer

    model_kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs.pop("dtype", None)  # quantization handles precision
    if args.use_flash_attn:
        model_kwargs["use_flash_attn"] = True

    try:
        model, tokenizer = load_with_transformers(model_kwargs)
        return model, tokenizer, dtype
    except Exception as exc:
        if args.use_flash_attn and not is_dynamic_module_missing(exc):
            print("Loading with flash-attn failed. Retrying with --no-use_flash_attn behavior.")
            model_kwargs.pop("use_flash_attn", None)
            try:
                model, tokenizer = load_with_transformers(model_kwargs)
                return model, tokenizer, dtype
            except Exception as retry_exc:
                exc = retry_exc

        if is_dynamic_module_missing(exc) and hydrate_dynamic_module_cache(local_model_dir, exc):
            print("transformers dynamic-module cache is missing local InternVL code files. Hydrating cache and retrying.")
            try:
                model, tokenizer = load_with_transformers(model_kwargs)
                return model, tokenizer, dtype
            except Exception as retry_exc:
                exc = retry_exc

        if not is_dynamic_module_missing(exc) or MsAutoModel is None or MsAutoTokenizer is None:
            raise

        print("transformers dynamic-module cache missed a relative import. Falling back to ModelScope loader.")
        ms_kwargs = dict(model_kwargs)
        ms_kwargs["torch_dtype"] = ms_kwargs.pop("dtype")
        model = MsAutoModel.from_pretrained(args.model_path, **ms_kwargs).eval()
        tokenizer = MsAutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    return model, tokenizer, dtype


def build_image_question(
    dataset: str,
    shots: list[dict[str, Any]],
    label_0_word: str,
    label_1_word: str,
) -> str:
    benchmark = BENCHMARKS[dataset]
    lines = [
        f"你正在完成任务：{benchmark['task_name']}。",
        f"回答“{label_1_word}”表示{benchmark['positive_definition']}。",
        f"回答“{label_0_word}”表示{benchmark['negative_definition']}。",
        "下面给出少样本示例。每个示例图片之后会给出正确答案。",
        "请直接观察图片本身，包括画面内容和图中文字。",
        "",
    ]
    image_index = 1
    for shot_index, shot in enumerate(shots, start=1):
        lines.extend(
            [
                f"Image-{image_index}: <image>",
                f"示例{shot_index}答案：{label_word(shot, label_0_word, label_1_word)}",
                "",
            ]
        )
        image_index += 1
    lines.extend(
        [
            f"Image-{image_index}: <image>",
            "请判断最后一张待测图片属于哪个标签。",
            f"请只输出一个严格标签：{label_0_word} 或 {label_1_word}。",
            "不要输出解释、标点、Markdown、JSON 或任何多余文字；否则本次回答会被视为无效。",
        ]
    )
    return "\n".join(lines)


def build_text_question(
    dataset: str,
    shots: list[dict[str, Any]],
    row: dict[str, Any],
    label_0_word: str,
    label_1_word: str,
    input_source: str,
) -> str:
    benchmark = BENCHMARKS[dataset]
    examples = []
    for index, shot in enumerate(shots, start=1):
        examples.append(
            "\n".join(
                [
                    f"示例{index}：",
                    f"输入：{row_input(shot, input_source='original')}",
                    f"答案：{label_word(shot, label_0_word, label_1_word)}",
                ]
            )
        )
    return "\n\n".join(
        [
            f"你正在完成任务：{benchmark['task_name']}。",
            f"回答“{label_1_word}”表示{benchmark['positive_definition']}；回答“{label_0_word}”表示{benchmark['negative_definition']}。",
            "请学习下面的少样本示例，然后判断最后一个待判断样本。",
            "\n\n".join(examples),
            "\n".join(
                [
                    "待判断样本：",
                    f"输入：{row_input(row, input_source=input_source)}",
                    f"请只输出一个严格标签：{label_0_word} 或 {label_1_word}。",
                    "不要输出解释、标点、Markdown、JSON 或任何多余文字；否则本次回答会被视为无效。",
                ]
            ),
        ]
    )


def build_pixel_values_and_question(
    dataset: str,
    shots: list[dict[str, Any]],
    row: dict[str, Any],
    args: argparse.Namespace,
    dtype,
) -> tuple[torch.Tensor | None, list[int] | None, str]:
    # image_text: text prompt (with original_input) + only the target image (no shot images)
    if args.input_source == "image_text":
        image_value = str(row.get("image") or "").strip()
        if not image_value:
            return (
                None,
                None,
                build_text_question(
                    dataset=dataset,
                    shots=shots,
                    row=row,
                    label_0_word=args.label_0_word,
                    label_1_word=args.label_1_word,
                    input_source="original",
                ),
            )
        image_path = resolve_image_path(args.repo_root, image_value)
        if not image_path.exists():
            print(f"Warning: image not found for image_text mode, falling back to text-only: {image_path}")
            return (
                None,
                None,
                build_text_question(
                    dataset=dataset,
                    shots=shots,
                    row=row,
                    label_0_word=args.label_0_word,
                    label_1_word=args.label_1_word,
                    input_source="original",
                ),
            )
        pixel_values = load_image_tensor(
            image_path,
            input_size=args.image_size,
            max_tiles=args.max_tiles_per_image,
        ).to(dtype)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        num_patches_list = [pixel_values.shape[0]]
        # Append the image placeholder to the text prompt so InternVL attaches the image
        question_base = build_text_question(
            dataset=dataset,
            shots=shots,
            row=row,
            label_0_word=args.label_0_word,
            label_1_word=args.label_1_word,
            input_source=args.input_source,
        )
        question = question_base + "\n\n待判断样本的原图如下，请结合图片内容和文字进行判断：\n<image>"
        return pixel_values, num_patches_list, question

    if args.input_source != "image":
        return (
            None,
            None,
            build_text_question(
                dataset=dataset,
                shots=shots,
                row=row,
                label_0_word=args.label_0_word,
                label_1_word=args.label_1_word,
                input_source=args.input_source,
            ),
        )

    rows = [*shots, row]
    pixel_values_list = []
    num_patches_list = []
    for item in rows:
        image_path = resolve_image_path(args.repo_root, str(item.get("image") or ""))
        pixel_values = load_image_tensor(
            image_path,
            input_size=args.image_size,
            max_tiles=args.max_tiles_per_image,
        )
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    question = build_image_question(
        dataset=dataset,
        shots=shots,
        label_0_word=args.label_0_word,
        label_1_word=args.label_1_word,
    )
    return pixel_values, num_patches_list, question


def generate_until_valid(
    model,
    tokenizer,
    pixel_values,
    question: str,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
    num_patches_list: list[int] | None = None,
) -> tuple[str, int, float, str]:
    last_error = ""
    for attempt in range(args.output_retries + 1):
        try:
            with torch.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False,
                )
            pred_label, positive_probability, parsed_label_text = parse_prediction(
                response,
                label_0_word=args.label_0_word,
                label_1_word=args.label_1_word,
            )
            return response, pred_label, positive_probability, parsed_label_text
        except Exception as exc:
            last_error = str(exc)
            if attempt >= args.output_retries:
                break
            print(f"Invalid local model output, retrying: {last_error}")
    raise InvalidPredictionError(last_error)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_split == "auto":
        args.eval_split = BENCHMARKS[args.dataset]["default_eval_split"]

    train_path = resolve_split_path(args.data_root, args.dataset, "train")
    eval_path = resolve_split_path(args.data_root, args.dataset, args.eval_split)
    train_rows = [row for row in read_json(train_path) if row.get("label") in (0, 1)]
    eval_rows = [row for row in read_json(eval_path) if row.get("label") in (0, 1)]
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    shots = build_fewshot_rows(train_rows, args.num_shots, args.seed, args.shot_mode)
    safe_model = safe_name(args.model_path)
    run_name = (
        f"{args.dataset}_{safe_model}_{args.shot_mode}{args.num_shots}_"
        f"seed{args.seed}_{args.eval_split}_{args.input_source}"
    )
    output_dir = args.output_dir / args.dataset / safe_model
    output_jsonl = output_dir / f"{run_name}.jsonl"
    output_csv = output_dir / f"{run_name}.csv"
    summary_path = output_dir / f"{run_name}.summary.json"
    shots_path = output_dir / f"{run_name}.shots.json"
    write_json(shots_path, shots)

    completed = load_completed_predictions(output_jsonl, retry_failed=True) if args.resume else {}
    records = list(completed.values())

    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")
    print(f"Train path: {train_path}")
    print(f"Eval path: {eval_path}")
    print(f"Few-shot examples: {len(shots)} ({args.shot_mode}={args.num_shots})")
    print(f"Eval examples: {len(eval_rows)}")
    print(f"Output: {output_jsonl}")

    model, tokenizer, dtype = load_model_and_tokenizer(args)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
    }

    for index, row in enumerate(eval_rows, start=1):
        case_id = str(row.get("id") or row.get("image") or index)
        if case_id in completed:
            if index % args.progress_every == 0:
                print(f"Skipped completed {index}/{len(eval_rows)}")
            continue

        record = {
            "case_id": case_id,
            "dataset": args.dataset,
            "eval_split": args.eval_split,
            "model": args.model_path,
            "gold_label": int(row["label"]),
            "pred_label": -1,
            "correct": False,
            "positive_probability": 0.5,
            "raw_response": "",
            "input": row_input(row, input_source=args.input_source),
            "image": row.get("image", ""),
            "text": row.get("text", ""),
            "error_message": "",
        }
        try:
            pixel_values, num_patches_list, question = build_pixel_values_and_question(
                dataset=args.dataset,
                shots=shots,
                row=row,
                args=args,
                dtype=dtype,
            )
            raw_response, pred_label, positive_probability, parsed_label_text = generate_until_valid(
                model=model,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                args=args,
                num_patches_list=num_patches_list,
            )
            record.update(
                {
                    "pred_label": pred_label,
                    "pred_label_text": parsed_label_text,
                    "correct": pred_label == int(row["label"]),
                    "positive_probability": positive_probability,
                    "raw_response": raw_response,
                }
            )
        except Exception as exc:
            record["error_message"] = str(exc)
            if not args.record_failed:
                print(f"Skipped failed {index}/{len(eval_rows)}: {record['error_message']}")
                continue

        records.append(record)
        append_jsonl(output_jsonl, record)

        if args.sleep > 0:
            time.sleep(args.sleep)
        if index % args.progress_every == 0 or index == len(eval_rows):
            print(f"Finished {index}/{len(eval_rows)}; latest pred={record['pred_label']} gold={record['gold_label']}")

    labels = [int(row["gold_label"]) for row in records]
    scores = [float(row["positive_probability"]) for row in records]
    valid_predictions = [row for row in records if row["pred_label"] in (0, 1)]
    correct = sum(1 for row in records if row["correct"])
    accuracy = correct / max(len(records), 1)
    auc = binary_auc(labels, scores) if records else 0.0
    invalid = len(records) - len(valid_predictions)

    summary = {
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "model": args.model_path,
        "num_shots": args.num_shots,
        "shot_mode": args.shot_mode,
        "input_source": args.input_source,
        "actual_num_fewshot_examples": len(shots),
        "num_total_eval": len(eval_rows),
        "num_eval": len(records),
        "num_skipped_failed": max(len(eval_rows) - len(records), 0),
        "num_invalid": invalid,
        "accuracy": round(accuracy * 100.0, 4),
        "auc": round(auc * 100.0, 4),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "shots_path": str(shots_path),
    }
    write_json(summary_path, summary)
    save_csv(output_csv, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local HuggingFace InternVL few-shot baselines.")
    parser.add_argument("--dataset", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("hf_internvl_fewshot_results"))
    parser.add_argument("--eval_split", choices=["auto", "test", "dev", "test_unlabeled"], default="auto")
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--shot_mode", choices=["per_class", "total"], default="per_class")
    parser.add_argument(
        "--input_source",
        choices=["image", "original", "generated", "image_text"],
        default="image",
        help=(
            "image: send all shot images + target image to the VLM. "
            "original/generated: text-only using original_input or input field. "
            "image_text: text prompt (original_input) + target image only (recommended for sarcnet)."
        ),
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--torch_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--use_flash_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_tiles_per_image", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_retries", type=int, default=2)
    parser.add_argument("--record_failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--label_0_word", default="否")
    parser.add_argument("--label_1_word", default="是")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
