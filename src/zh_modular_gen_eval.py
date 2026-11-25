import os
from pathlib import Path

from transformers import GenerationConfig

from case_study_utils import run_binary_generation_case_study, save_case_study_records
from zh_fewshot_dataset import ZH_BENCHMARKS, load_json
from zh_modular_dataset import generate_prompt_test


def build_eval_data(args):
    if args.DATASET not in ZH_BENCHMARKS:
        raise ValueError(f"Unsupported Chinese benchmark for generation eval: {args.DATASET}")

    eval_split = getattr(args, "eval_split", None)
    if not eval_split or eval_split == "auto":
        eval_split = "test" if args.DATASET in ("zh-toxicn-mm", "zh-sarcnet") else "dev"

    filename = f"{eval_split}.json"
    if eval_split == "test_unlabeled":
        filename = "test_unlabeled.json"

    data_path = Path(args.PATH) / "zh_modular" / ZH_BENCHMARKS[args.DATASET] / filename
    rows = load_json(data_path)

    data = []
    for row in rows:
        if row.get("label") not in (0, 1):
            continue
        metadata = row.get("metadata") or {}
        data.append(
            {
                "case_id": row.get("id", ""),
                "path": row.get("image", ""),
                "img": row.get("image", ""),
                "image_path": row.get("image", ""),
                "label": int(row["label"]),
                "text": row.get("text", ""),
                "meme_description": metadata.get("image_description", ""),
                "text_description": metadata.get("meme_meaning", ""),
                "input": row.get("input", ""),
                "instruction": row.get("instruction", ""),
                "output": row.get("output", ""),
                "source": row.get("source", ""),
                "task": row.get("task", ""),
            }
        )

    if args.DEBUG:
        data = data[:48]
    return data, data_path


def zh_modular_generation(model, tokenizer, args, global_step=None, save_case_study=False, output_dir=None):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(pad_token_id=tokenizer.pad_token_id)
    model.eval()

    data, data_path = build_eval_data(args)
    print("Number of examples:", len(data))
    print("\t", data_path)

    invalid, auc, acc, records = run_binary_generation_case_study(
        model=model,
        tokenizer=tokenizer,
        data=data,
        generation_config=generation_config,
        generate_prompt_fn=generate_prompt_test,
        label_0_word=args.POS_WORD,
        label_1_word=args.NEG_WORD,
        progress_every=500,
        max_new_tokens=args.max_new_tokens,
    )

    if save_case_study:
        case_dir = output_dir or os.path.join(args.output_dir, "case_study")
        file_stem = "eval_predictions" if global_step is None else f"step_{global_step}_eval_predictions"
        save_paths = save_case_study_records(
            records=records,
            output_dir=case_dir,
            stem=file_stem,
            title=f"{args.DATASET} Case Study",
        )
        print("Saved case study files:")
        for path in save_paths.values():
            print("\t", path)

    model.train(True)
    return invalid, auc, acc
