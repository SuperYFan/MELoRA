import os
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from generate_lora3_visual_inputs import (
        DEFAULT_MODEL,
    append_jsonl,
    call_api,
    load_cache,
    strip_json_fence,
    write_json_atomic,
)


LABEL_RULE = "1 if intention detection is 4(offensive), else 0"
PROMPT_VERSION = "met_reasoning_label_specific_instruction_v1"
POSITIVE_INSTRUCTION = "请根据表情包描述解释它为什么属于仇恨内容。"
NEGATIVE_INSTRUCTION = "请根据表情包描述解释它为什么不属于仇恨内容。"

PROMPT_TEMPLATE = """请根据给定图片和任务指令，生成一个用于训练中文 meme 仇恨推理模块的 JSON 对象。

任务指令：{instruction}
OCR参考文字：{ocr_text}

要求：
1. "description"：客观描述图片主体、图片文字、人物或物体关系、表情、动作和必要语境。不要写“图片描述：”前缀。
2. "reasoning"：根据给定标签解释为什么该表情包属于或不属于仇恨内容。需要结合图片文字、视觉元素、隐喻/谐音/攻击对象等线索；不要只复述标签。
3. 只做二分类：仇恨 / 非仇恨。不要输出多分类标签。
4. 不要输出 Markdown，不要输出列表，不要额外解释。

只输出 JSON，格式如下：
{{"description":"...","reasoning":"..."}}
"""


def read_csv_with_encoding(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def normalize_text(text: Any) -> str:
    text = str(text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(part.strip() for part in text.split("\n") if part.strip())


def image_number(image_name: str) -> int | None:
    match = re.search(r"\((\d+)\)", image_name)
    if not match:
        return None
    return int(match.group(1))


def resolve_image_path(image_root: Path, image_name: str) -> Path:
    direct = image_root / image_name
    if direct.exists():
        return direct
    number = image_number(image_name)
    if number is not None:
        converted = image_root / f"Image- ({number}).jpg"
        if converted.exists():
            return converted
    return direct


def is_hate(row: dict[str, str]) -> int:
    intention = (row.get("intention detection") or "").lower()
    return int("offensive" in intention or intention.startswith("4("))


def instruction_for_label(label: int) -> str:
    return POSITIVE_INSTRUCTION if int(label) == 1 else NEGATIVE_INSTRUCTION


def parse_response(text: str) -> dict[str, str]:
    try:
        parsed = json.loads(strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM response is not valid JSON: {exc}; response={text[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"VLM response JSON must be an object, got {type(parsed).__name__}")

    description = normalize_text(parsed.get("description"))
    reasoning = normalize_text(parsed.get("reasoning"))
    if not description:
        raise ValueError(f"Missing description in response: {text[:500]!r}")
    if not reasoning:
        raise ValueError(f"Missing reasoning in response: {text[:500]!r}")
    return {"description": description, "reasoning": reasoning}


def build_base_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    label_rows = read_csv_with_encoding(args.data_root / "label_C.csv")
    text_rows = read_csv_with_encoding(args.data_root / "C_text.csv")
    text_map = {
        row.get("images_name", "").strip(): normalize_text(row.get("text"))
        for row in text_rows
        if row.get("images_name")
    }

    rng = random.Random(args.seed)
    indices_by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, label_row in enumerate(label_rows):
        indices_by_label[is_hate(label_row)].append(index)
    test_indices: set[int] = set()
    for label_indices in indices_by_label.values():
        rng.shuffle(label_indices)
        test_count = int(round(len(label_indices) * args.test_ratio))
        test_indices.update(label_indices[:test_count])

    rows = []
    for idx, row in enumerate(label_rows):
        image_name = row.get("images_name", "").strip()
        number = image_number(image_name)
        row_id = f"met_meme_{number}" if number is not None else f"met_meme_{idx}"
        label = is_hate(row)
        split = "test" if idx in test_indices else "train"
        image_path = resolve_image_path(args.image_root, image_name)
        rows.append(
            {
                "id": row_id,
                "source": "MET-Meme",
                "split": split,
                "task": "met_meme_reasoning",
                "instruction": instruction_for_label(label),
                "input": "",
                "output": "",
                "image": image_path.as_posix(),
                "text": text_map.get(image_name, ""),
                "label": label,
                "label_text": "是" if label else "否",
                "metadata": {
                    "images_name": image_name,
                    "sentiment_category": row.get("sentiment category"),
                    "sentiment_degree": row.get("sentiment degree"),
                    "intention_detection": row.get("intention detection"),
                    "offensiveness_detection": row.get("offensiveness detection"),
                    "metaphor_occurrence": row.get("metaphor occurrence"),
                    "metaphor_category": row.get("metaphor category"),
                    "source_domain": row.get("source domain"),
                    "target_domain": row.get("target domain"),
                    "target_modality": row.get("target modality"),
                    "source_modality": row.get("source modality"),
                    "binary_label_rule": LABEL_RULE,
                    "requires_vision": True,
                },
            }
        )
    return rows


def merge_existing(rows: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        path = output_root / f"{split}.json"
        if path.exists():
            for row in json.load(path.open("r", encoding="utf-8")):
                existing[row.get("id", "")] = row

    merged = []
    for row in rows:
        old = existing.get(row["id"])
        old_metadata = (old or {}).get("metadata") or {}
        same_label_rule = old_metadata.get("binary_label_rule") == LABEL_RULE
        same_label = old and old.get("label") == row.get("label")
        same_prompt_version = old_metadata.get("met_reasoning_prompt_version") == PROMPT_VERSION
        if (
            old
            and old.get("input")
            and old.get("output")
            and same_label
            and same_label_rule
            and same_prompt_version
        ):
            old["instruction"] = row["instruction"]
            old["split"] = row["split"]
            old["image"] = row["image"]
            old["text"] = row["text"]
            old["label"] = row["label"]
            old["label_text"] = row["label_text"]
            old_metadata = old.setdefault("metadata", {})
            old_metadata.update(row.get("metadata") or {})
            old_metadata["requires_vision"] = False
            merged.append(old)
        else:
            merged.append(row)
    return merged


def save_splits(rows: list[dict[str, Any]], output_root: Path) -> None:
    for split in ("train", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        write_json_atomic(output_root / f"{split}.json", split_rows)


def row_needs_generation(row: dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return True
    metadata = row.get("metadata") or {}
    return not (
        row.get("input")
        and row.get("output")
        and metadata.get("met_reasoning_model")
        and metadata.get("met_reasoning_prompt_version") == PROMPT_VERSION
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MET-Meme hate/non-hate reasoning data with a VLM.")
    parser.add_argument("--data_root", type=Path, default=Path("data/MET-Meme"))
    parser.add_argument("--image_root", type=Path, default=Path("data/MET-Meme/Cimages/Cimages"))
    parser.add_argument("--output_root", type=Path, default=Path("data/zh_modular/modules/met_m3_toxicn_reasoning"))
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default="", help="API key for the endpoint given by --api_url.")
    parser.add_argument("--cache_jsonl", type=Path, default=Path("data/zh_modular/modules/met_m3_toxicn_reasoning/reasoning_cache.jsonl"))
    parser.add_argument("--failed_jsonl", type=Path, default=Path("data/zh_modular/modules/met_m3_toxicn_reasoning/reasoning_failed.jsonl"))
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    args = parser.parse_args()

    rows = build_base_rows(args)
    if not args.overwrite:
        rows = merge_existing(rows, args.output_root)

    cache = load_cache(args.cache_jsonl)
    attempted = 0
    changed_since_save = 0
    print(f"Prepared {len(rows)} MET-Meme rows. Output root: {args.output_root}")
    print(f"Train rows: {sum(row.get('split') == 'train' for row in rows)}")
    print(f"Test rows: {sum(row.get('split') == 'test' for row in rows)}")

    for index, row in enumerate(rows):
        if index < args.start or not row_needs_generation(row, args.overwrite):
            continue
        if args.limit is not None and attempted >= args.limit:
            break

        image_path = Path(row["image"])
        if not image_path.exists():
            message = f"Missing image for row {index} ({row['id']}): {image_path}"
            if args.skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)

        cache_key = f"{LABEL_RULE}:{PROMPT_VERSION}:{row['id']}"
        try:
            if cache_key in cache and not args.overwrite and "met_reasoning" in cache[cache_key]:
                parsed = cache[cache_key]["met_reasoning"]
                print(f"[cache] {index}: {row['id']}")
            elif args.dry_run:
                print(f"[dry-run] {index}: {row['id']} label={row['label_text']} image={image_path.name}")
                attempted += 1
                continue
            else:
                prompt = PROMPT_TEMPLATE.format(
                    instruction=row["instruction"],
                    label_text="仇恨" if row["label"] else "非仇恨",
                    ocr_text=row.get("text") or "无",
                )
                print(f"[api] {index}: {row['id']} label={row['label_text']} -> {image_path.name}")
                raw_response = call_api(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                    image_path=image_path,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_image_side=args.max_image_side,
                    jpeg_quality=args.jpeg_quality,
                    prompt=prompt,
                )
                parsed = parse_response(raw_response)
                cache[cache_key] = {
                    "id": cache_key,
                    "image": row["image"],
                    "model": args.model,
                    "binary_label_rule": LABEL_RULE,
                    "met_reasoning_prompt_version": PROMPT_VERSION,
                    "label": row["label"],
                    "label_text": row["label_text"],
                    "met_reasoning": parsed,
                    "raw_response": raw_response,
                }
                append_jsonl(args.cache_jsonl, cache[cache_key])
                if args.sleep > 0:
                    time.sleep(args.sleep)
        except (RuntimeError, ValueError) as exc:
            error_message = str(exc)
            print(f"[failed] {index}: {row['id']} -> {error_message}")
            metadata = row.setdefault("metadata", {})
            metadata["met_reasoning_error"] = error_message
            metadata["met_reasoning_failed_model"] = args.model
            append_jsonl(
                args.failed_jsonl,
                {
                    "id": row["id"],
                    "index": index,
                    "image": row["image"],
                    "model": args.model,
                    "error": error_message,
                },
            )
            attempted += 1
            changed_since_save += 1
            if changed_since_save >= args.save_every:
                save_splits(rows, args.output_root)
                print(f"Saved progress after {attempted} attempted rows.")
                changed_since_save = 0
            if args.fail_fast:
                save_splits(rows, args.output_root)
                raise
            continue

        row["input"] = parsed["description"]
        row["output"] = parsed["reasoning"]
        metadata = row.setdefault("metadata", {})
        metadata["met_description"] = parsed["description"]
        metadata["met_reasoning"] = parsed["reasoning"]
        metadata["met_reasoning_model"] = args.model
        metadata["met_reasoning_prompt_version"] = PROMPT_VERSION
        metadata["requires_vision"] = False
        metadata.pop("met_reasoning_error", None)
        metadata.pop("met_reasoning_failed_model", None)

        attempted += 1
        changed_since_save += 1
        if changed_since_save >= args.save_every:
            save_splits(rows, args.output_root)
            print(f"Saved progress after {attempted} attempted rows.")
            changed_since_save = 0

    if not args.dry_run:
        save_splits(rows, args.output_root)
    print(f"Finished. Attempted rows: {attempted}. Output root: {args.output_root}")


if __name__ == "__main__":
    main()
