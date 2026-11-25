import json
import os
from pathlib import Path

from datasets import load_dataset


ZH_DATASETS = {
    "zh-hate-semantics": ("modules/hate_semantics", "test"),
    "zh-sarcasm-semantics": ("modules/sarcasm_semantics", "test"),
    "zh-met-meme-reasoning": ("modules/met_m3_toxicn_reasoning", "test"),
    "zh-met-m3-toxicn-reasoning": ("modules/met_m3_toxicn_reasoning", "test"),
    "zh-met-m3-sarcnet-reasoning": ("modules/met_m3_sarcnet_reasoning", "test"),
    "zh-visual-expression": ("modules/visual_expression", "train"),
    "zh-toxicn-mm": ("benchmarks/toxicn_mm", "test"),
    "zh-misogyny-meme": ("benchmarks/misogyny_meme", "dev"),
    "zh-sarcnet": ("benchmarks/sarcnet", "test"),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:{data_point["instruction"]}
                
                ### Input:{data_point["input"]}
                
                ### Response:{data_point["output"]}"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:{data_point["instruction"]}
                
                ### Response:{data_point["output"]}"""


def generate_prompt_test(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:{data_point["instruction"]}
                
                ### Input:{data_point["input"]}
                
                ### Response:"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:{data_point["instruction"]}
                
                ### Response:"""


def tokenize(prompt, tokenizer, cutoff_len, base_model, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in base_model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in base_model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    return result


class ZH_Modular_Data:
    def __init__(self, opt, tokenizer, mode="train"):
        super().__init__()
        if opt.DATASET not in ZH_DATASETS:
            raise ValueError(f"Unsupported Chinese modular dataset: {opt.DATASET}")
        self.opt = opt
        self.tokenizer = tokenizer
        self.mode = mode
        self.relative_dir, self.default_eval_split = ZH_DATASETS[opt.DATASET]
        print("Initializing tokenizer from:", self.opt.base_model)
        print("\tUnified Chinese dataset:", self.opt.DATASET)

    def get_data_path(self):
        data_dir = Path(self.opt.PATH) / "zh_modular" / self.relative_dir
        if self.mode == "train":
            return data_dir / "train.json"

        preferred = data_dir / f"{self.default_eval_split}.json"
        if preferred.exists():
            return preferred

        for filename in ("test.json", "dev.json", "train.json"):
            candidate = data_dir / filename
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"No usable split found under {data_dir}")

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(
            full_prompt,
            self.tokenizer,
            self.opt.cutoff_len,
            self.opt.base_model,
        )
        if not self.opt.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(
                user_prompt,
                self.tokenizer,
                self.opt.cutoff_len,
                self.opt.base_model,
                add_eos_token=False,
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    def get_dataset(self):
        load_path = self.get_data_path()
        data = load_dataset("json", data_files=str(load_path))
        print("Loading data from %s mode, from %s" % (self.mode, load_path))

        # Unlabeled benchmark test files are for submission/inference, not supervised training.
        if "output" in data["train"].column_names:
            data["train"] = data["train"].filter(lambda row: row.get("output") not in (None, ""))

        if self.opt.DEBUG:
            print("Select a subset of data...")
            data["train"] = data["train"].select([i for i in range(min(16, len(data["train"])))])
        elif self.mode != "train" and os.path.basename(load_path) == "train.json":
            eval_size = min(getattr(self.opt, "val_set_size", 500), len(data["train"]))
            data["train"] = data["train"].select([i for i in range(eval_size)])

        return data["train"].shuffle(seed=self.opt.SEED).map(self.generate_and_tokenize_prompt)
