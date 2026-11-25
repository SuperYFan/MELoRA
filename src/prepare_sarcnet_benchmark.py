import argparse
import json
from pathlib import Path
from typing import Any


INSTRUCTION = "请根据文本内容判断该帖子是否包含讽刺。只回答“是”或“否”。"


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def convert_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        label = int(row.get("multi_label", 0))
        text = str(row.get("text") or "").strip()
        image_name = str(row.get("path") or "").strip()
        converted.append(
            {
                "id": f"sarcnet_{split}_{index}",
                "source": "SarcNet",
                "split": split,
                "task": "sarcasm_detection",
                "instruction": INSTRUCTION,
                "input": f"文本内容：{text}",
                "output": "是" if label else "否",
                "image": str(Path("data") / "sarcnet" / "meme" / image_name).replace("\\", "/"),
                "text": text,
                "label": label,
                "label_text": "是" if label else "否",
                "metadata": {
                    "path": image_name,
                    "text_label": row.get("text_label"),
                    "image_label": row.get("image_label"),
                    "multi_label": row.get("multi_label"),
                    "language": row.get("language"),
                    "note": "SarcNet is used here as an additional Chinese sarcasm-detection transfer benchmark.",
                },
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SarcNet into zh_modular benchmark format.")
    parser.add_argument("--data_root", type=Path, default=Path("data/sarcnet"))
    parser.add_argument("--output_root", type=Path, default=Path("data/zh_modular/benchmarks/sarcnet"))
    args = parser.parse_args()

    split_files = {
        "train": "train_data.json",
        "dev": "validation_data.json",
        "test": "test_data.json",
    }
    for split, filename in split_files.items():
        rows = read_json(args.data_root / filename)
        converted = convert_split(rows, split)
        write_json(args.output_root / f"{split}.json", converted)
        positives = sum(row["label"] == 1 for row in converted)
        negatives = sum(row["label"] == 0 for row in converted)
        print(f"{split}: {len(converted)} rows, positive={positives}, negative={negatives}")


if __name__ == "__main__":
    main()
