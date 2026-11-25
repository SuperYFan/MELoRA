import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sarcnet_gen_eval import sarcnet_generation
from toxicn_gen_eval import toxicn_generation


def parse_args():
    parser = argparse.ArgumentParser(description="Export per-example predictions for case study.")
    parser.add_argument("--DATASET", type=str, required=True, choices=["toxicn", "sarcnet"])
    parser.add_argument("--PATH", type=str, required=True, help="Parent directory that contains dataset folders.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path or model name.")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter directory or checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save case study files.")
    parser.add_argument("--POS_WORD", type=str, default="No", help="Verbalizer for label 0.")
    parser.add_argument("--NEG_WORD", type=str, default="Yes", help="Verbalizer for label 1.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    return parser.parse_args()


def get_torch_dtype():
    if torch.cuda.is_available():
        return torch.float16
    try:
        if torch.backends.mps.is_available():
            return torch.float16
    except Exception:
        pass
    return torch.float32


def load_model_and_tokenizer(args):
    torch_dtype = get_torch_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "qwen" not in args.base_model.lower():
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args)

    if args.DATASET == "toxicn":
        invalid, auc, acc = toxicn_generation(
            model=model,
            tokenizer=tokenizer,
            args=args,
            save_case_study=True,
            output_dir=args.output_dir,
        )
    else:
        invalid, auc, acc = sarcnet_generation(
            model=model,
            tokenizer=tokenizer,
            args=args,
            save_case_study=True,
            output_dir=args.output_dir,
        )

    print(f"Finished exporting case study for {args.DATASET}.")
    print(f"AUC: {auc:.2f}")
    print(f"ACC: {acc:.2f}")
    if invalid:
        print(f"Invalid examples: {len(invalid)}")


if __name__ == "__main__":
    main()
