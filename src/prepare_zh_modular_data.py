import argparse
import csv
import json
from pathlib import Path
from typing import Any


YES = "是"
NO = "否"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rel(path: Path) -> str:
    return path.as_posix()


def yn(value: bool | int | None) -> str:
    return YES if bool(value) else NO


def make_record(
    *,
    row_id: str,
    source: str,
    split: str,
    task: str,
    instruction: str,
    input_text: str,
    output: str,
    image: str | None = None,
    text: str | None = None,
    label: int | None = None,
    label_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": source,
        "split": split,
        "task": task,
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "image": image,
        "text": text,
        "label": label,
        "label_text": label_text,
        "metadata": metadata or {},
    }


def convert_hate_semantics(data_root: Path, output_root: Path) -> None:
    instruction = "请判断下面中文文本是否包含有害、歧视、攻击或冒犯性内容。只回答“是”或“否”。"
    src_dir = data_root / "ToxiCN-main" / "data"
    for split in ("train", "test"):
        rows = read_json(src_dir / f"{split}.json")
        converted = []
        for i, row in enumerate(rows):
            label = int(row["toxic"])
            content = str(row.get("content", "")).strip()
            converted.append(
                make_record(
                    row_id=f"toxicn_text_{split}_{i}",
                    source="ToxiCN",
                    split=split,
                    task="hate_semantics",
                    instruction=instruction,
                    input_text=f"文本：{content}",
                    output=yn(label),
                    text=content,
                    label=label,
                    label_text=yn(label),
                    metadata={
                        "platform": row.get("platform"),
                        "topic": row.get("topic"),
                        "toxic_type": row.get("toxic_type"),
                        "expression": row.get("expression"),
                        "target": row.get("target"),
                        "length": row.get("length"),
                    },
                )
            )
        write_json(output_root / "modules" / "hate_semantics" / f"{split}.json", converted)


def convert_sarcasm_semantics(data_root: Path, output_root: Path, seed: int = 1111) -> None:
    instruction = "请判断下面中文文本是否包含讽刺。只回答“是”或“否”。"
    split_map = {
        "train": read_json(data_root / "ToSarcasm" / "train.json"),
        "test": read_json(data_root / "ToSarcasm" / "dev.json"),
    }

    for split, split_rows_data in split_map.items():
        converted = []
        for i, row in enumerate(split_rows_data):
            label = int(row.get("label", 0))
            text = str(row.get("text", "")).strip()
            topic = str(row.get("topic", "")).strip()
            input_text = f"{topic}\n{text}" if topic and text else (text or topic)
            converted.append(
                make_record(
                    row_id=f"sarcasm_semantics_{split}_{i}",
                    source="ToSarcasm",
                    split=split,
                    task="sarcasm_semantics",
                    instruction=instruction,
                    input_text=input_text,
                    output=yn(label),
                    text=text,
                    label=label,
                    label_text=yn(label),
                    metadata={
                        "topic": topic,
                    },
                )
            )
        write_json(output_root / "modules" / "sarcasm_semantics" / f"{split}.json", converted)


def convert_meme_comprehension(data_root: Path, output_root: Path) -> None:
    instruction = "请解释下面中文网络梗或 meme 表达的含义、来源、使用场景，并判断是否具有冒犯性。"
    rows = read_json(data_root / "chime_full" / "chime_full.json")
    converted = []
    for i, row in enumerate(rows):
        examples = row.get("examples") or []
        example_text = "；".join(str(x).strip() for x in examples if str(x).strip())
        origin = row.get("origin") or "暂无明确来源。"
        output = "\n".join(
            [
                f"含义：{row.get('meaning', '')}",
                f"来源：{origin}",
                f"类型：{row.get('type_cn', '')}（{row.get('type_en', '')}）",
                f"是否含有粗俗表达：{yn(row.get('profanity'))}",
                f"是否具有冒犯性：{yn(row.get('offense'))}",
                f"示例用法：{example_text}",
            ]
        )
        meme = str(row.get("meme", "")).strip()
        converted.append(
            make_record(
                row_id=f"chime_train_{i}",
                source="CHIME",
                split="train",
                task="meme_comprehension",
                instruction=instruction,
                input_text=f"梗语：{meme}",
                output=output,
                text=meme,
                label=int(bool(row.get("offense"))),
                label_text=yn(row.get("offense")),
                metadata={
                    "meaning": row.get("meaning"),
                    "origin": row.get("origin"),
                    "examples": examples,
                    "profanity": row.get("profanity"),
                    "offense": row.get("offense"),
                    "type_cn": row.get("type_cn"),
                    "type_en": row.get("type_en"),
                },
            )
        )
    write_json(output_root / "modules" / "meme_comprehension" / "train.json", converted)


def convert_visual_expression(data_root: Path, output_root: Path) -> None:
    instruction = "请根据图片描述和图片文字解释这张中文表情包表达的整体含义。"
    rows = read_json(data_root / "emo-visual-data" / "data.json")
    converted = []
    for i, row in enumerate(rows):
        filename = str(row.get("filename", "")).strip()
        image = rel(Path("data") / "emo-visual-data" / "emo" / filename)
        content = str(row.get("content", "")).strip()
        converted.append(
            make_record(
                row_id=f"emo_visual_train_{i}",
                source="emo-visual-data",
                split="train",
                task="visual_expression",
                instruction=instruction,
                input_text="<image>\n请先用 VLM 生成图片描述、图片文字和 meme 整体含义。",
                output="",
                image=image,
                metadata={
                    "filename": filename,
                    "requires_vision": True,
                    "original_content": content,
                    "note": (
                        "Run generate_lora3_visual_inputs.py --lora3_format meme_interp "
                        "to create image caption + meme text -> meme meaning training rows."
                    ),
                },
            )
        )
    write_json(output_root / "modules" / "visual_expression" / "train.json", converted)


def load_target_map(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    mapping: dict[int, str] = {}
    for row in read_csv(path):
        raw_id = (row.get("ID") or "").strip()
        if raw_id.isdigit():
            mapping[int(raw_id)] = (row.get("Category") or "").strip()
    return mapping


def convert_toxicn_mm(data_root: Path, output_root: Path) -> None:
    instruction = "请根据中文 meme 的图片描述和文字内容判断它是否有害。只回答“是”或“否”。"
    src_dir = data_root / "ToxiCN_MM"
    split_files = {
        "train": "train_data_discription_2.0.json",
        "test": "test_data_discription_2.0.json",
    }
    target_map = load_target_map(src_dir / "target_info.csv")
    for split, filename in split_files.items():
        rows = read_json(src_dir / filename)
        converted = []
        for i, row in enumerate(rows):
            label = int(row["label"])
            image_name = str(row.get("path", "")).strip()
            image = rel(Path("data") / "ToxiCN_MM" / "meme" / image_name)
            text = str(row.get("text", "")).strip()
            image_desc = str(row.get("meme_discription", "")).strip()
            converted.append(
                make_record(
                    row_id=f"toxicn_mm_{split}_{i}",
                    source="ToxiCN_MM",
                    split=split,
                    task="harmful_meme_detection",
                    instruction=instruction,
                    input_text=f"{image_desc}\n{text}" if text else image_desc,
                    output=yn(label),
                    image=image,
                    text=text,
                    label=label,
                    label_text=yn(label),
                    metadata={
                        "image_description": image_desc,
                        "meme_meaning": row.get("text_discription"),
                        "type": row.get("type"),
                        "text_modal": row.get("text_modal"),
                        "image_modal": row.get("image_modal"),
                        "target": row.get("target"),
                        "target_category": target_map.get(row.get("target", -1)),
                    },
                )
            )
        write_json(output_root / "benchmarks" / "toxicn_mm" / f"{split}.json", converted)


def convert_misogyny_meme(data_root: Path, output_root: Path) -> None:
    instruction = "请判断这张中文 meme 是否包含厌女内容。只回答“是”或“否”。"
    base = data_root / "Mysogyny-Meme-Detection-main" / "Datasets"
    split_info = {
        "train": (Path("train") / "train.csv", "train images"),
        "dev": (Path("dev") / "dev.csv", "dev images"),
        "test": (Path("test") / "test.csv", "test images"),
    }
    for split, (csv_rel, image_dir) in split_info.items():
        csv_path = base / csv_rel
        rows = read_csv(csv_path)
        converted = []
        for i, row in enumerate(rows):
            image_name = str(row.get("image_name", "")).strip()
            image = rel(Path("data") / "Mysogyny-Meme-Detection-main" / "Datasets" / split / image_dir / image_name)
            text = str(row.get("transcriptions", "")).strip()
            raw_label = (row.get("labels") or "").strip()
            label = None
            output = ""
            label_text = None
            if raw_label:
                label = 1 if raw_label.lower() == "misogyny" else 0
                output = yn(label)
                label_text = yn(label)
            converted.append(
                make_record(
                    row_id=f"misogyny_meme_{split}_{i}",
                    source="Chinese-Misogyny-Meme-Detection",
                    split=split,
                    task="misogyny_meme_detection",
                    instruction=instruction,
                    input_text=f"图片文字：{text}",
                    output=output,
                    image=image,
                    text=text,
                    label=label,
                    label_text=label_text,
                    metadata={
                        "raw_label": raw_label or None,
                        "is_labeled": label is not None,
                    },
                )
            )
        out_name = f"{split}.json" if split != "test" else "test_unlabeled.json"
        write_json(output_root / "benchmarks" / "misogyny_meme" / out_name, converted)


def write_schema(output_root: Path) -> None:
    schema = {
        "format": "Unified instruction-tuning JSON for Chinese modular harmful meme detection.",
        "fields": {
            "id": "Stable sample id.",
            "source": "Original dataset name.",
            "split": "train/dev/test.",
            "task": "Module or benchmark task name.",
            "instruction": "Natural-language task instruction.",
            "input": "Textual input consumed by the current LLM pipeline.",
            "output": "Gold answer. For unlabeled test sets this is an empty string.",
            "image": "Repository-relative image path when available.",
            "text": "Raw OCR/text content when available.",
            "label": "Binary label when available. 1 means harmful/toxic/misogyny/offensive for the task.",
            "label_text": "Chinese verbalizer, 是 for positive and 否 for negative.",
            "metadata": "Dataset-specific extra fields.",
        },
        "tasks": {
            "hate_semantics": "LoRA-1, ToxiCN text toxic classification.",
            "sarcasm_semantics": "Sarcasm LoRA-1 for SarcNet transfer, trained from ToSarcasm topic-aware text classification.",
            "met_meme_reasoning": "LoRA-2, MET-Meme hate/non-hate reasoning generated from image-conditioned VLM descriptions.",
            "visual_expression": "LoRA-3, emo-visual expression semantics from textified visual descriptions, following the original caption-based setup.",
            "harmful_meme_detection": "Main benchmark, ToxiCN_MM.",
            "misogyny_meme_detection": "Few-shot benchmark, Chinese misogyny meme detection.",
        },
    }
    write_json(output_root / "schema.json", schema)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified Chinese modular meme datasets.")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--output_root", type=Path, default=Path("data") / "zh_modular")
    args = parser.parse_args()

    convert_hate_semantics(args.data_root, args.output_root)
    convert_sarcasm_semantics(args.data_root, args.output_root, seed=1111)
    # LoRA-2 now uses MET-Meme + VLM-generated reasoning. Run
    # src/generate_met_meme_reasoning.py after this script to build that module.
    convert_visual_expression(args.data_root, args.output_root)
    convert_toxicn_mm(args.data_root, args.output_root)
    convert_misogyny_meme(args.data_root, args.output_root)
    write_schema(args.output_root)
    print(f"Unified Chinese modular datasets written to {args.output_root}")


if __name__ == "__main__":
    main()
