import argparse
import base64
import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PLACEHOLDER_PREFIX = "<image>"
BENCHMARK_JSON_NAMES = ("train.json", "test.json", "dev.json", "test_unlabeled.json")
DEFAULT_BENCHMARK_EVAL_JSONS = {
    "toxicn_mm": ("test.json",),
    "misogyny_meme": ("dev.json",),
}

PROMPT = (
    "请只根据图片本身生成一段中文图片说明。"
    "必须简要描述画面主体、可见文字、人物或物体关系、表情情绪等内容。"
    "不要进行仇恨分类，不要输出标题、列表、Markdown、前缀或解释性套话，也不要使用“似乎”这类主观判断。"
    "只输出图片描述正文。"
)

MEME_INTERP_PROMPT = (
    "请根据图片和参考说明生成一个中文 JSON 对象，字段格式要对齐英文 meme-interp 数据集。"
    "参考说明来自原始 emo-visual-data 的 content，可用于理解图片想表达的含义，但不要照抄其中关于笑点、幽默效果、适用场景或社交媒体使用方式的内容。"
    "\n参考说明：{reference_content}\n"
    "必须包含两个字段："
    "\"img_captions\"：数组，包含 1 到 2 条客观图片描述，描述画面主体、可见文字、人物或物体关系、动作、表情和场景；"
    "\"meme_captions\"：数组，包含 1 条中文含义解释，必须用“这个表情包表达了……”或“这个表情包表达的是……”开头，只说明这张 meme 表达的整体含义、态度或隐含信息。"
    "不要分析笑点，不要写适合使用场景，不要写“产生幽默效果”“引发共鸣和笑声”等套话，不要输出 Markdown。"
    "只输出 JSON，例如："
    "{{\"img_captions\":[\"...\"],\"meme_captions\":[\"...\"]}}"
)

MEME_INTERP_INSTRUCTION = "请根据图片描述和图片文字解释这张中文表情包表达的整体含义。"
MEME_INTERP_PROMPT_VERSION = "img_captions_meme_captions_reference_content_expression_prefix_v1"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if row_id:
                cache[row_id] = row
    return cache


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        print(f"Image normalization failed for {path.name}, using original bytes: {exc}")
        return raw_image_to_data_url(path)


def build_payload(
    model: str,
    image_data_url: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def clean_description(text: str) -> str:
    text = text.strip()
    for prefix in ("图片说明：", "图片描述：", "描述：", "说明："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)


def extract_message_content(response: dict[str, Any]) -> str:
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


def extract_content(response: dict[str, Any]) -> str:
    return clean_description(extract_message_content(response))


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def clean_single_line(text: Any) -> str:
    text = str(text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(line.strip() for line in text.split("\n") if line.strip())


def normalize_meme_caption(text: Any) -> str:
    caption = clean_single_line(text)
    if not caption:
        return ""
    normalized = caption.lstrip("，。；：、,.;: ")
    if normalized.startswith(("这个表情包表达了", "这个表情包表达的是")):
        return normalized
    if normalized.startswith("表达了"):
        return "这个表情包" + normalized
    if normalized.startswith("表达的是"):
        return "这个表情包" + normalized
    if normalized.startswith("表达"):
        return "这个表情包" + normalized
    return "这个表情包表达了" + normalized


def parse_meme_interp_response(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM response is not valid JSON: {exc}; response={text[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"VLM response JSON must be an object, got {type(parsed).__name__}")

    img_captions_raw = parsed.get("img_captions")
    if img_captions_raw is None:
        img_captions_raw = parsed.get("image_caption")
    if isinstance(img_captions_raw, str):
        img_captions = [clean_single_line(img_captions_raw)]
    elif isinstance(img_captions_raw, list):
        img_captions = [clean_single_line(item) for item in img_captions_raw]
    else:
        img_captions = []
    img_captions = [caption for caption in img_captions if caption]

    meme_captions_raw = parsed.get("meme_captions")
    if meme_captions_raw is None:
        meme_captions_raw = parsed.get("meme_caption")
    if isinstance(meme_captions_raw, str):
        meme_captions = [normalize_meme_caption(meme_captions_raw)]
    elif isinstance(meme_captions_raw, list):
        meme_captions = [normalize_meme_caption(item) for item in meme_captions_raw]
    else:
        meme_captions = []
    meme_captions = [caption for caption in meme_captions if caption]

    result = {
        "img_captions": img_captions,
        "meme_captions": meme_captions,
        "meme_text": clean_single_line(parsed.get("meme_text")),
    }
    if not result["img_captions"]:
        raise ValueError(f"Missing img_captions in VLM response: {text[:500]!r}")
    if not result["meme_captions"]:
        raise ValueError(f"Missing meme_captions in VLM response: {text[:500]!r}")
    return result


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


def call_api(
    *,
    api_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
    max_image_side: int,
    jpeg_quality: int,
    prompt: str,
) -> str:
    image_data_url = normalized_image_to_data_url(image_path, max_image_side, jpeg_quality)
    payload = build_payload(
        model=model,
        image_data_url=image_data_url,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt=prompt,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for attempt in range(retries + 1):
        request = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
            return extract_message_content(json.loads(response_body))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = describe_api_error(exc)
            if attempt >= retries:
                break
            wait_seconds = min(2**attempt, 30)
            print(f"API call failed for {image_path.name}, retrying in {wait_seconds}s: {last_error}")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to generate description for {image_path}: {last_error}")


def infer_target(path: Path, explicit_target: str) -> str:
    if explicit_target != "auto":
        return explicit_target
    normalized = str(path).replace("\\", "/")
    if "/benchmarks/" in normalized or normalized.startswith("data/zh_modular/benchmarks/"):
        return "benchmark"
    return "lora3"


def extract_existing_text(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    if text:
        return text

    for source in (
        row.get("original_input"),
        (row.get("metadata") or {}).get("original_input"),
        row.get("input"),
    ):
        input_text = str(source or "").strip()
        marker = "图片文字："
        if marker in input_text:
            return input_text.split(marker, 1)[1].strip()
    return ""


def build_benchmark_input(row: dict[str, Any], description: str) -> str:
    description = clean_description(description)
    text = extract_existing_text(row)
    if text:
        return f"{description}\n{text}"
    return description


def build_meme_interp_input(fields: dict[str, Any]) -> str:
    image_caption = fields["img_captions"][0]
    meme_text = fields.get("meme_text", "")
    if meme_text:
        return f"{image_caption}\n{meme_text}"
    return image_caption


def get_meme_interp_reference_content(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    for value in (
        metadata.get("original_content"),
        row.get("content"),
        row.get("original_content"),
        row.get("output") if not metadata.get("meme_interp_model") else "",
    ):
        content = clean_single_line(value)
        if content:
            return content
    return "无"


def build_meme_interp_prompt(row: dict[str, Any]) -> str:
    return MEME_INTERP_PROMPT.format(
        reference_content=get_meme_interp_reference_content(row)
    )


def update_row_with_description(
    row: dict[str, Any],
    generated_text: str,
    *,
    target: str,
    model: str,
    api_url: str,
    lora3_format: str,
) -> None:
    metadata = row.setdefault("metadata", {})
    if target == "benchmark":
        description = clean_description(generated_text)
        original_input = row.get("input", "")
        if original_input and "original_input" not in row:
            row["original_input"] = original_input
        if original_input and not metadata.get("original_input"):
            metadata["original_input"] = original_input
        original_description = metadata.get("image_description")
        if original_description and not metadata.get("original_image_description"):
            metadata["original_image_description"] = original_description
        metadata["image_description"] = description
        row["input"] = build_benchmark_input(row, description)
    elif lora3_format == "meme_interp":
        fields = parse_meme_interp_response(generated_text)
        if row.get("output") and not metadata.get("original_content"):
            metadata["original_content"] = row.get("output")
        if row.get("instruction") and not metadata.get("original_instruction"):
            metadata["original_instruction"] = row.get("instruction")
        row["instruction"] = MEME_INTERP_INSTRUCTION
        row["input"] = build_meme_interp_input(fields)
        row["output"] = fields["meme_captions"][0]
        row["text"] = fields["meme_text"] or None
        row["img_captions"] = fields["img_captions"]
        row["meme_captions"] = fields["meme_captions"]
        metadata["img_captions"] = fields["img_captions"]
        metadata["meme_captions"] = fields["meme_captions"]
        metadata["image_caption"] = fields["img_captions"][0]
        metadata["meme_text"] = fields["meme_text"]
        metadata["meme_caption"] = fields["meme_captions"][0]
        metadata["meme_interp_model"] = model
        metadata["meme_interp_prompt_version"] = MEME_INTERP_PROMPT_VERSION
        metadata["note"] = (
            "The image was converted by a VLM into img_captions and meme_captions, "
            "following the original meme-interp raw-data style. "
            "The original emo-visual content is used only as meaning reference; joke/use-case wording is removed."
        )
    else:
        description = clean_description(generated_text)
        row["input"] = description
        metadata["note"] = (
            "The image was first converted into a Chinese visual description by a VLM; "
            "LoRA-3 uses the generated text as input."
        )

    metadata["visual_description_model"] = model
    metadata["requires_vision"] = False
    metadata.pop("visual_description_error", None)
    metadata.pop("visual_description_failed_model", None)


def should_generate(row: dict[str, Any], overwrite: bool, target: str, lora3_format: str) -> bool:
    if overwrite:
        return True
    metadata = row.get("metadata") or {}
    if target == "lora3" and lora3_format == "meme_interp":
        return metadata.get("meme_interp_prompt_version") != MEME_INTERP_PROMPT_VERSION
    if metadata.get("visual_description_model"):
        return False
    if target == "benchmark":
        return not metadata.get("image_description")
    input_text = str(row.get("input", "")).strip()
    return not input_text or input_text.startswith(PLACEHOLDER_PREFIX)


def resolve_image_path(repo_root: Path, image_value: str) -> Path:
    path = Path(image_value)
    if path.is_absolute():
        return path
    return repo_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual descriptions for the Visual Expression Expert with a VLM."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path("data/zh_modular/modules/visual_expression/train.json"),
        help="LoRA-3 JSON file to read. Image paths are read from each row's image field.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON file. Defaults to in-place update of --input_json.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve relative image paths.",
    )
    parser.add_argument(
        "--target",
        choices=("auto", "lora3", "benchmark"),
        default="auto",
        help="How to write generated descriptions. auto uses benchmark mode for data/zh_modular/benchmarks.",
    )
    parser.add_argument(
        "--lora3_format",
        choices=("visual_description", "meme_interp"),
        default="visual_description",
        help=(
            "How to write LoRA-3 rows. visual_description keeps the old description-only input; "
            "meme_interp writes meme-interp style img_captions + meme_captions, then uses "
            "the first image caption as input and the first meme meaning as output."
        ),
    )
    parser.add_argument(
        "--process_benchmarks",
        action="store_true",
        help="Process every benchmark split JSON under --benchmarks_dir.",
    )
    parser.add_argument(
        "--benchmarks_dir",
        type=Path,
        default=Path("data/zh_modular/benchmarks"),
        help="Benchmark root used when --process_benchmarks is set.",
    )
    parser.add_argument(
        "--benchmark_files",
        nargs="+",
        default=None,
        help=(
            "Specific benchmark split files to process, relative to each benchmark dataset directory. "
            "Defaults to toxicn_mm/test.json and misogyny_meme/dev.json. Use 'all' for every split."
        ),
    )
    parser.add_argument(
        "--cache_jsonl",
        type=Path,
        default=Path("data/zh_modular/modules/visual_expression/description_cache.jsonl"),
        help="JSONL cache for successful descriptions.",
    )
    parser.add_argument(
        "--failed_jsonl",
        type=Path,
        default=Path("data/zh_modular/modules/visual_expression/description_failed.jsonl"),
        help="JSONL log for failed image generations.",
    )
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default="", help="API key for the endpoint given by --api_url.")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between successful calls.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many rows needing generation.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the JSON list.")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if a row already has generated input.")
    parser.add_argument("--dry_run", action="store_true", help="Print rows that would be processed without API calls.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip rows with missing local images.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately when an API call fails.")
    parser.add_argument(
        "--max_image_side",
        type=int,
        default=1024,
        help="Resize the longest image side before sending. Use 0 to send original bytes.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=90)
    return parser.parse_args()


def discover_benchmark_jsons(benchmarks_dir: Path, benchmark_files: list[str] | None) -> list[Path]:
    paths = []
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmarks_dir}")
    for dataset_dir in sorted(path for path in benchmarks_dir.iterdir() if path.is_dir()):
        if benchmark_files and "all" in benchmark_files:
            filenames = BENCHMARK_JSON_NAMES
        elif benchmark_files:
            filenames = tuple(benchmark_files)
        else:
            filenames = DEFAULT_BENCHMARK_EVAL_JSONS.get(dataset_dir.name, ())
        for filename in filenames:
            path = dataset_dir / filename
            if path.exists():
                paths.append(path)
    return paths


def process_json_file(args: argparse.Namespace, input_json: Path, output_json: Path, cache: dict[str, dict[str, Any]]) -> int:
    rows = read_json(input_json)
    target = infer_target(input_json, args.target)
    processed = 0
    changed_since_save = 0

    print(f"Processing {input_json} as target={target}")
    for index, row in enumerate(rows):
        if index < args.start or not should_generate(row, args.overwrite, target, args.lora3_format):
            continue
        if args.limit is not None and processed >= args.limit:
            break

        row_id = str(row.get("id", index))
        image_value = str(row.get("image", "")).strip()
        image_path = resolve_image_path(args.repo_root, image_value)
        if not image_value or not image_path.exists():
            message = f"Missing image for row {index} ({row_id}): {image_value}"
            if args.skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)

        cache_row = cache.get(row_id)
        if (
            cache_row
            and not args.overwrite
            and not (target == "lora3" and args.lora3_format == "meme_interp")
            and "description" in cache_row
        ):
            generated_text = clean_description(str(cache_row["description"]))
            print(f"[cache] {index}: {row_id}")
        elif (
            cache_row
            and not args.overwrite
            and target == "lora3"
            and args.lora3_format == "meme_interp"
            and "meme_interp" in cache_row
            and cache_row.get("meme_interp_prompt_version") == MEME_INTERP_PROMPT_VERSION
        ):
            generated_text = json.dumps(cache_row["meme_interp"], ensure_ascii=False)
            print(f"[cache:meme-interp] {index}: {row_id}")
        elif args.dry_run:
            print(f"[dry-run] {index}: {row_id} -> {image_path}")
            processed += 1
            continue
        else:
            print(f"[api] {index}: {row_id} -> {image_path.name}")
            prompt = build_meme_interp_prompt(row) if target == "lora3" and args.lora3_format == "meme_interp" else PROMPT
            try:
                generated_text = call_api(
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
                if target == "lora3" and args.lora3_format == "meme_interp":
                    parsed_for_cache = parse_meme_interp_response(generated_text)
                else:
                    parsed_for_cache = None
            except RuntimeError as exc:
                error_message = str(exc)
                print(f"[failed] {index}: {row_id} -> {error_message}")
                metadata = row.setdefault("metadata", {})
                metadata["visual_description_error"] = error_message
                metadata["visual_description_failed_model"] = args.model
                append_jsonl(
                    args.failed_jsonl,
                    {
                        "id": row_id,
                        "index": index,
                        "image": image_value,
                        "file": str(input_json),
                        "model": args.model,
                        "error": error_message,
                    },
                )
                processed += 1
                changed_since_save += 1
                if changed_since_save >= args.save_every:
                    write_json_atomic(output_json, rows)
                    print(f"Saved progress to {output_json} after {processed} attempted rows.")
                    changed_since_save = 0
                if args.fail_fast:
                    write_json_atomic(output_json, rows)
                    raise
                continue
            except ValueError as exc:
                error_message = str(exc)
                print(f"[failed-parse] {index}: {row_id} -> {error_message}")
                metadata = row.setdefault("metadata", {})
                metadata["visual_description_error"] = error_message
                metadata["visual_description_failed_model"] = args.model
                append_jsonl(
                    args.failed_jsonl,
                    {
                        "id": row_id,
                        "index": index,
                        "image": image_value,
                        "file": str(input_json),
                        "model": args.model,
                        "error": error_message,
                        "raw_response": generated_text if "generated_text" in locals() else "",
                    },
                )
                processed += 1
                changed_since_save += 1
                if changed_since_save >= args.save_every:
                    write_json_atomic(output_json, rows)
                    print(f"Saved progress to {output_json} after {processed} attempted rows.")
                    changed_since_save = 0
                if args.fail_fast:
                    write_json_atomic(output_json, rows)
                    raise
                continue

            cache[row_id] = {
                "id": row_id,
                "image": image_value,
                "file": str(input_json),
                "model": args.model,
                "description": clean_description(generated_text),
            }
            if parsed_for_cache is not None:
                cache[row_id]["meme_interp"] = parsed_for_cache
                cache[row_id]["meme_interp_prompt_version"] = MEME_INTERP_PROMPT_VERSION
                cache[row_id]["reference_content"] = get_meme_interp_reference_content(row)
                cache[row_id]["raw_response"] = generated_text
            append_jsonl(args.cache_jsonl, cache[row_id])
            if args.sleep > 0:
                time.sleep(args.sleep)

        update_row_with_description(
            row,
            generated_text,
            target=target,
            model=args.model,
            api_url=args.api_url,
            lora3_format=args.lora3_format,
        )

        processed += 1
        changed_since_save += 1
        if changed_since_save >= args.save_every:
            write_json_atomic(output_json, rows)
            print(f"Saved progress to {output_json} after {processed} attempted rows.")
            changed_since_save = 0

    if not args.dry_run:
        write_json_atomic(output_json, rows)
    print(f"Finished {input_json}. Generated, restored, or logged {processed} rows. Output: {output_json}")
    return processed


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()

    if not args.dry_run and not args.api_key:
        raise SystemExit("Please pass --api_key.")

    cache = load_cache(args.cache_jsonl)
    if args.process_benchmarks:
        total_processed = 0
        benchmark_paths = discover_benchmark_jsons(args.benchmarks_dir, args.benchmark_files)
        print("Benchmark files to process:")
        for path in benchmark_paths:
            print(f"\t{path}")
        for input_json in benchmark_paths:
            total_processed += process_json_file(args, input_json, input_json, cache)
        print(f"Finished benchmark processing. Total attempted rows: {total_processed}")
        return

    input_json = args.input_json
    output_json = args.output_json or input_json
    process_json_file(args, input_json, output_json, cache)


if __name__ == "__main__":
    main()
