import argparse
import base64
import csv
import io
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

BENCHMARKS = {
    "zh-toxicn-mm": {
        "relative_dir": "benchmarks/toxicn_mm",
        "default_eval_split": "test",
        "task_name": "中文有害 meme 检测",
        "positive_definition": "有害、仇恨、攻击、歧视、侮辱、冒犯或包含针对群体/个人的恶意表达",
        "negative_definition": "无害、普通玩笑、自嘲、日常幽默或不构成攻击/歧视",
    },
    "zh-misogyny-meme": {
        "relative_dir": "benchmarks/misogyny_meme",
        "default_eval_split": "dev",
        "task_name": "中文厌女 meme 检测",
        "positive_definition": "包含厌女、贬低女性、性别刻板印象、物化女性或针对女性的攻击表达",
        "negative_definition": "不包含厌女含义，或只是普通中性/无害表达",
    },
    "zh-sarcnet": {
        "relative_dir": "benchmarks/sarcnet",
        "default_eval_split": "test",
        "task_name": "中文讽刺文本检测",
        "positive_definition": "包含讽刺、反语、阴阳怪气、表面称赞实际批评或带有强烈反差的表达",
        "negative_definition": "不包含讽刺含义，是普通陈述、直接表达观点或中性/非反讽文本",
    },
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resolve_split_path(data_root: Path, dataset: str, split: str) -> Path:
    benchmark = BENCHMARKS[dataset]
    filename = "test_unlabeled.json" if split == "test_unlabeled" else f"{split}.json"
    return data_root / "zh_modular" / benchmark["relative_dir"] / filename


def build_fewshot_rows(rows: list[dict[str, Any]], num_shots: int, seed: int, shot_mode: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    if shot_mode == "per_class":
        target_counts = {0: num_shots, 1: num_shots}
    else:
        negative_count = num_shots // 2
        positive_count = num_shots - negative_count
        target_counts = {0: negative_count, 1: positive_count}

    counts = {0: 0, 1: 0}
    selected = []
    for row in shuffled:
        label = row.get("label")
        if label not in (0, 1):
            continue
        if counts[label] >= target_counts[label]:
            continue
        counts[label] += 1
        selected.append(row)
        if counts[0] >= target_counts[0] and counts[1] >= target_counts[1]:
            break

    missing = {label: target_counts[label] - counts[label] for label in (0, 1)}
    if any(value > 0 for value in missing.values()):
        raise ValueError(f"Not enough labeled examples for {shot_mode}={num_shots}: missing {missing}")
    return selected


def row_input(row: dict[str, Any], input_source: str = "original") -> str:
    if input_source == "image":
        return ""
    if input_source in ("generated",):
        input_text = str(row.get("input") or "").strip()
    else:
        # "original" and "image_text" both use original_input
        input_text = str(row.get("original_input") or "").strip()
        if not input_text:
            metadata = row.get("metadata") or {}
            input_text = str(metadata.get("original_input") or "").strip()
        if not input_text:
            input_text = str(row.get("input") or "").strip()
    if input_text:
        return input_text
    text = str(row.get("text") or "").strip()
    return f"图片文字：{text}" if text else ""


def label_word(row: dict[str, Any], label_0_word: str, label_1_word: str) -> str:
    return label_1_word if int(row["label"]) == 1 else label_0_word


def build_prompt(
    *,
    dataset: str,
    shots: list[dict[str, Any]],
    row: dict[str, Any],
    label_0_word: str,
    label_1_word: str,
    use_image: bool,
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

    image_note = (
        "当前待判断样本的原图会随消息一起提供，请结合图片、图片文字和上下文判断。"
        if use_image
        else "当前输入已经是图片描述、图片文字或 OCR 文本，请只根据这些文本化信息判断。"
    )

    return "\n\n".join(
        [
            f"你正在完成任务：{benchmark['task_name']}。",
            f"标签定义：回答“{label_1_word}”表示{benchmark['positive_definition']}；回答“{label_0_word}”表示{benchmark['negative_definition']}。",
            image_note,
            "请学习下面的少样本示例，然后判断最后一个待判断样本。",
            "\n\n".join(examples),
            "\n".join(
                [
                    "待判断样本：",
                    f"输入：{row_input(row, input_source=input_source)}",
                    f"请只输出一个严格标签：{label_0_word} 或 {label_1_word}。",
                    f"其中 {label_0_word} 表示负类，{label_1_word} 表示正类。",
                    "不要输出解释、Markdown、JSON 或任何多余文字；否则本次回答会被视为无效。",
                ]
            ),
        ]
    )


def raw_image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def normalized_image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    if max_side <= 0:
        return raw_image_to_data_url(path)
    try:
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as exc:
        print(f"Image normalization failed for {path}: {exc}; using original bytes.")
        return raw_image_to_data_url(path)


def resolve_image_path(repo_root: Path, image_value: str) -> Path:
    path = Path(image_value)
    if path.is_absolute():
        return path
    return repo_root / path


def image_content_item(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_value = str(row.get("image") or "").strip()
    if not image_value:
        raise FileNotFoundError("Missing image field.")
    image_path = resolve_image_path(args.repo_root, image_value)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_data_url = normalized_image_to_data_url(
        image_path,
        max_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
    )
    return {"type": "image_url", "image_url": {"url": image_data_url}}


def build_image_fewshot_content(
    *,
    dataset: str,
    shots: list[dict[str, Any]],
    row: dict[str, Any],
    label_0_word: str,
    label_1_word: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    benchmark = BENCHMARKS[dataset]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    f"你正在完成任务：{benchmark['task_name']}。",
                    f"回答“{label_1_word}”表示{benchmark['positive_definition']}。",
                    f"回答“{label_0_word}”表示{benchmark['negative_definition']}。",
                    "下面给出少样本示例。每个示例的图片之后会给出正确答案。",
                    "请直接观察图片本身，包括画面内容和图中文字；不要依赖外部文本化输入。",
                ]
            ),
        }
    ]
    for index, shot in enumerate(shots, start=1):
        content.append({"type": "text", "text": f"示例{index}图片："})
        content.append(image_content_item(shot, args))
        content.append({"type": "text", "text": f"示例{index}答案：{label_word(shot, label_0_word, label_1_word)}"})

    content.append(
        {
            "type": "text",
            "text": "\n".join(
                [
                    "待判断图片：",
                    "请判断下面这张图片属于哪个标签。",
                    f"请只输出一个严格标签：{label_0_word} 或 {label_1_word}。",
                    f"其中 {label_0_word} 表示负类，{label_1_word} 表示正类。",
                    "不要输出解释、Markdown、JSON 或任何多余文字；否则本次回答会被视为无效。",
                ]
            ),
        }
    )
    content.append(image_content_item(row, args))
    return content


def build_messages(
    *,
    dataset: str,
    shots: list[dict[str, Any]],
    prompt: str,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    system_message = {
        "role": "system",
        "content": "你是严谨的中文 meme 内容安全分类器。必须遵守输出格式，只输出一个字：是或否。",
    }
    if args.input_source == "image":
        return [
            system_message,
            {
                "role": "user",
                "content": build_image_fewshot_content(
                    dataset=dataset,
                    shots=shots,
                    row=row,
                    label_0_word=args.label_0_word,
                    label_1_word=args.label_1_word,
                    args=args,
                ),
            },
        ]

    # image_text: send text prompt (with original_input) AND attach the image
    force_image = (args.input_source == "image_text")
    if not args.use_image and not force_image:
        return [system_message, {"role": "user", "content": prompt}]

    image_value = str(row.get("image") or "").strip()
    if not image_value:
        return [system_message, {"role": "user", "content": prompt}]

    image_path = resolve_image_path(args.repo_root, image_value)
    if not image_path.exists():
        print(f"Warning: image not found, falling back to text-only prompt: {image_path}")
        return [system_message, {"role": "user", "content": prompt}]

    image_data_url = normalized_image_to_data_url(
        image_path,
        max_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
    )
    return [
        system_message,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


def build_payload(messages: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def describe_api_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if len(body) > 1000:
            body = body[:1000] + "...<truncated>"
        return f"HTTP {exc.code} {exc.reason}: {body}" if body else f"HTTP {exc.code} {exc.reason}"
    return repr(exc)


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in API response: {response}")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        content = "".join(parts)
    return str(content).strip()


def call_api(messages: list[dict[str, Any]], args: argparse.Namespace) -> str:
    payload = build_payload(messages, args)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for attempt in range(args.retries + 1):
        request = urllib.request.Request(args.api_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_body = response.read().decode("utf-8")
            return extract_content(json.loads(response_body))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = describe_api_error(exc)
            if attempt >= args.retries:
                break
            base_delay = max(float(args.retry_base_delay), 0.1)
            wait_seconds = min(base_delay * (2**attempt), float(args.retry_max_delay))
            wait_seconds += random.uniform(0.0, min(1.0, wait_seconds * 0.1))
            print(f"API call failed, retrying in {wait_seconds}s: {last_error}")
            time.sleep(wait_seconds)
    raise RuntimeError(last_error)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class InvalidPredictionError(ValueError):
    pass


def parse_prediction(text: str, label_0_word: str, label_1_word: str) -> tuple[int, float, str]:
    cleaned = strip_code_fence(text)
    normalized = cleaned.strip().strip('"').strip("'").strip()
    if normalized == label_0_word:
        return 0, 0.0, label_0_word
    if normalized == label_1_word:
        return 1, 1.0, label_1_word
    raise InvalidPredictionError(
        f"Expected exact output '{label_0_word}' or '{label_1_word}', got: {cleaned[:200]}"
    )


def call_api_until_valid_prediction(
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str, int, float, str]:
    last_error = ""
    for attempt in range(args.output_retries + 1):
        try:
            raw_response = call_api(messages, args)
            pred_label, positive_probability, parsed_label_text = parse_prediction(
                raw_response,
                label_0_word=args.label_0_word,
                label_1_word=args.label_1_word,
            )
            return raw_response, pred_label, positive_probability, parsed_label_text
        except Exception as exc:
            last_error = str(exc)
            if attempt >= args.output_retries:
                break
            wait_seconds = min(float(args.retry_base_delay) * (2**attempt), float(args.retry_max_delay))
            wait_seconds += random.uniform(0.0, min(1.0, wait_seconds * 0.1))
            print(f"Invalid API output or API error, retrying in {wait_seconds:.2f}s: {last_error}")
            time.sleep(wait_seconds)
    raise RuntimeError(last_error)


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return 0.0

    sorted_pairs = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1] == sorted_pairs[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            ranks[sorted_pairs[rank_index][0]] = average_rank
        index = end

    positive_rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc


def load_completed_predictions(path: Path, retry_failed: bool = True) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if case_id:
                if retry_failed and (row.get("pred_label") not in (0, 1) or row.get("error_message")):
                    completed.pop(case_id, None)
                    continue
                completed[case_id] = row
    return completed


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "gold_label",
        "pred_label",
        "correct",
        "positive_probability",
        "raw_response",
        "input",
        "image",
        "text",
        "error_message",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


PredictionFunction = Callable[
    [list[dict[str, Any]], argparse.Namespace],
    tuple[str, int, float, str],
]


def evaluate(
    args: argparse.Namespace,
    prediction_fn: PredictionFunction | None = None,
) -> dict[str, Any]:
    if prediction_fn is None:
        prediction_fn = call_api_until_valid_prediction
    benchmark = BENCHMARKS[args.dataset]
    eval_split = args.eval_split or benchmark["default_eval_split"]
    train_path = resolve_split_path(args.data_root, args.dataset, "train")
    eval_path = resolve_split_path(args.data_root, args.dataset, eval_split)

    train_rows = [row for row in read_json(train_path) if row.get("label") in (0, 1)]
    eval_rows = [row for row in read_json(eval_path) if row.get("label") in (0, 1)]
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    shots = build_fewshot_rows(train_rows, args.num_shots, args.seed, args.shot_mode)
    run_name = (
        f"{args.dataset}_{safe_name(args.model)}_{args.shot_mode}{args.num_shots}_"
        f"seed{args.seed}_{eval_split}_{args.input_source}"
    )
    if args.use_image:
        run_name += "_image"

    output_dir = args.output_dir / args.dataset / safe_name(args.model)
    output_jsonl = output_dir / f"{run_name}.jsonl"
    output_csv = output_dir / f"{run_name}.csv"
    summary_path = output_dir / f"{run_name}.summary.json"
    shots_path = output_dir / f"{run_name}.shots.json"
    write_json(shots_path, shots)

    completed = load_completed_predictions(output_jsonl, retry_failed=args.retry_failed) if args.resume else {}
    records = []
    for row in completed.values():
        records.append(row)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Train path: {train_path}")
    print(f"Eval path: {eval_path}")
    print(f"Few-shot examples: {len(shots)} ({args.shot_mode}={args.num_shots})")
    print(f"Eval examples: {len(eval_rows)}")
    print(f"Output: {output_jsonl}")

    for index, row in enumerate(eval_rows, start=1):
        case_id = str(row.get("id") or row.get("image") or index)
        if case_id in completed:
            if index % args.progress_every == 0:
                print(f"Skipped completed {index}/{len(eval_rows)}")
            continue

        prompt = build_prompt(
            dataset=args.dataset,
            shots=shots,
            row=row,
            label_0_word=args.label_0_word,
            label_1_word=args.label_1_word,
            use_image=args.use_image,
            input_source=args.input_source,
        )
        record = {
            "case_id": case_id,
            "dataset": args.dataset,
            "eval_split": eval_split,
            "model": args.model,
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
            messages = build_messages(dataset=args.dataset, shots=shots, prompt=prompt, row=row, args=args)
            raw_response, pred_label, positive_probability, parsed_label_text = prediction_fn(
                messages,
                args,
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
                if args.sleep > 0:
                    time.sleep(args.sleep)
                continue
            print(
                f"Counted failed {index}/{len(eval_rows)} as incorrect: "
                f"{record['error_message']}"
            )

        records.append(record)
        append_jsonl(output_jsonl, record)

        if args.sleep > 0:
            time.sleep(args.sleep)
        if index % args.progress_every == 0 or index == len(eval_rows):
            current = records[-len(eval_rows) :] if len(records) >= len(eval_rows) else records
            print(f"Finished {index}/{len(eval_rows)}; latest pred={record['pred_label']} gold={record['gold_label']}")

    labels = [int(row["gold_label"]) for row in records]
    scores = [float(row["positive_probability"]) for row in records]
    valid_predictions = [row for row in records if row["pred_label"] in (0, 1)]
    correct = sum(1 for row in records if row["correct"])
    accuracy = correct / max(len(records), 1)
    auc = binary_auc(labels, scores)
    invalid = len(records) - len(valid_predictions)

    summary = {
        "dataset": args.dataset,
        "eval_split": eval_split,
        "model": args.model,
        "api_url": args.api_url,
        "num_shots": args.num_shots,
        "shot_mode": args.shot_mode,
        "input_source": args.input_source,
        "actual_num_fewshot_examples": len(shots),
        "use_image": args.use_image,
        "num_total_eval": len(eval_rows),
        "num_eval": len(records),
        "num_skipped_failed": max(len(eval_rows) - len(records), 0),
        "num_invalid": invalid,
        "num_invalid_counted_as_incorrect": invalid,
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
    parser = argparse.ArgumentParser(description="Run VLM API few-shot baselines on Chinese meme benchmarks.")
    parser.add_argument("--dataset", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("api_fewshot_results"))
    parser.add_argument("--eval_split", choices=["auto", "test", "dev", "test_unlabeled"], default="auto")
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--shot_mode", choices=["per_class", "total"], default="per_class")
    parser.add_argument(
        "--input_source",
        choices=["image", "original", "generated", "image_text"],
        default="image",
        help=(
            "image sends few-shot example images and the target image to a VLM. "
            "original/generated use original_input or input as a text-only reference. "
            "image_text sends both the original text AND the image (recommended for sarcnet)."
        ),
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--api_key", default="", help="API key for the endpoint given by --api_url.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output_retries", type=int, default=3)
    parser.add_argument("--retry_base_delay", type=float, default=2.0)
    parser.add_argument("--retry_max_delay", type=float, default=60.0)
    parser.add_argument("--retry_failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record_failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--label_0_word", default="否")
    parser.add_argument("--label_1_word", default="是")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--json_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.eval_split == "auto":
        args.eval_split = BENCHMARKS[args.dataset]["default_eval_split"]
    args.api_key = (args.api_key or "").strip()
    if not args.api_key:
        raise EnvironmentError("Please pass --api_key.")
    args.data_root = args.data_root.resolve()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
