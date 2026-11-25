import os

from transformers import GenerationConfig

from case_study_utils import run_binary_generation_case_study, save_case_study_records
from toxicn_dataset import load_json, process_toxicn_data


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:{data_point["instruction"]}
                
                ### Input:{data_point["input"]}
                
                ### Response:"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:{data_point["instruction"]}
                
                ### Response:"""


def build_toxicn_eval_data(args):
    data_path = os.path.join(args.PATH, "ToxiCN_MM", "test_data_discription_2.0.json")
    data_raw = load_json(data_path)
    prompt_rows = process_toxicn_data(data_raw)

    data = []
    for raw_row, prompt_row in zip(data_raw, prompt_rows):
        image_name = raw_row["path"]
        data.append(
            {
                "case_id": image_name,
                "path": image_name,
                "image_path": os.path.join(args.PATH, "ToxiCN_MM", "meme", image_name),
                "label": int(raw_row["label"]),
                "text": raw_row.get("text", ""),
                "meme_description": raw_row.get("meme_discription", ""),
                "text_description": raw_row.get("text_discription", ""),
                "type": raw_row.get("type"),
                "text_modal": raw_row.get("text_modal"),
                "image_modal": raw_row.get("image_modal"),
                "target": raw_row.get("target"),
                "input": prompt_row["input"],
                "instruction": prompt_row["instruction"],
                "output": prompt_row["output"],
            }
        )

    if args.DEBUG:
        data = data[:48]

    return data, data_path


def toxicn_generation(model, tokenizer, args, global_step=None, save_case_study=False, output_dir=None):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
    )
    model.eval()

    data, data_path = build_toxicn_eval_data(args)
    print("Number of examples:", len(data))
    print("\t", data_path)

    invalid, auc, acc, records = run_binary_generation_case_study(
        model=model,
        tokenizer=tokenizer,
        data=data,
        generation_config=generation_config,
        generate_prompt_fn=generate_prompt,
        label_0_word=args.POS_WORD,
        label_1_word=args.NEG_WORD,
        progress_every=500,
        max_new_tokens=args.max_new_tokens,
    )

    if save_case_study:
        case_dir = output_dir or os.path.join(args.output_dir, "case_study")
        file_stem = "test_predictions" if global_step is None else f"step_{global_step}_test_predictions"
        save_paths = save_case_study_records(
            records=records,
            output_dir=case_dir,
            stem=file_stem,
            title="ToxiCN Case Study",
        )
        print("Saved case study files:")
        for path in save_paths.values():
            print("\t", path)

    model.train(True)
    return invalid, auc, acc
