import json
import random
from pathlib import Path

import torch

from zh_modular_dataset import ZH_DATASETS, generate_prompt, tokenize


ZH_BENCHMARKS = {
    "zh-toxicn-mm": ZH_DATASETS["zh-toxicn-mm"][0],
    "zh-misogyny-meme": ZH_DATASETS["zh-misogyny-meme"][0],
    "zh-sarcnet": ZH_DATASETS["zh-sarcnet"][0],
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fewshot_rows(rows, num_shots, seed):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    counts = {0: 0, 1: 0}
    selected = []
    for row in shuffled:
        if counts[0] >= num_shots and counts[1] >= num_shots:
            break
        label = row.get("label")
        if label not in (0, 1):
            continue
        if counts[label] >= num_shots:
            continue
        counts[label] += 1
        selected.append(row)
    return selected


def build_ratio_rows(rows, calibration_ratio, seed):
    """Sample *calibration_ratio* fraction of training rows, preserving the
    natural class distribution (no forced balancing).  The subset is drawn
    by shuffling with *seed* and taking the first N items."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = max(1, round(len(shuffled) * calibration_ratio))
    return shuffled[:n]


class ZH_FewShot_Data:
    def __init__(self, opt, tokenizer, mode="train"):
        super().__init__()
        if opt.DATASET not in ZH_BENCHMARKS:
            raise ValueError(f"Unsupported Chinese few-shot benchmark: {opt.DATASET}")
        self.opt = opt
        self.tokenizer = tokenizer
        self.mode = mode
        self.entries = self.get_entries()
        print("Length of few-shot dataset", len(self.entries))

    def get_entries(self):
        data_path = Path(self.opt.PATH) / "zh_modular" / ZH_BENCHMARKS[self.opt.DATASET] / "train.json"
        rows = load_json(data_path)

        calibration_ratio = getattr(self.opt, "calibration_ratio", 0.0)
        if calibration_ratio > 0:
            selected = build_ratio_rows(rows, calibration_ratio, self.opt.SEED)
            cache_name = f"ratio{calibration_ratio:.2f}_{self.opt.SEED}.json"
        else:
            selected = build_fewshot_rows(rows, self.opt.num_shots, self.opt.SEED)
            cache_name = f"fewshot_{self.opt.num_shots}_{self.opt.SEED}.json"

        cache_path = (
            Path(self.opt.PATH)
            / "zh_modular"
            / ZH_BENCHMARKS[self.opt.DATASET]
            / cache_name
        )
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return selected

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

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        pad_len = self.opt.cutoff_len - len(tokenized_full_prompt["input_ids"])
        if pad_len > 0:
            tokenized_full_prompt["input_ids"] = [pad_token_id] * pad_len + tokenized_full_prompt["input_ids"]
            tokenized_full_prompt["labels"] = [-100] * pad_len + tokenized_full_prompt["labels"]
            tokenized_full_prompt["attention_mask"] = [0] * pad_len + tokenized_full_prompt["attention_mask"]

        return tokenized_full_prompt

    def __getitem__(self, index):
        batch = self.generate_and_tokenize_prompt(self.entries[index])
        batch["input_ids"] = torch.LongTensor(batch["input_ids"])
        batch["labels"] = torch.LongTensor(batch["labels"])
        batch["attention_mask"] = torch.LongTensor(batch["attention_mask"])
        return batch

    def __len__(self):
        return len(self.entries)
