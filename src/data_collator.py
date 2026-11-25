from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CausalLMDataCollator:
    """Pad causal-LM batches while keeping labels aligned with input_ids."""

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        allowed_model_keys = {"input_ids", "attention_mask", "token_type_ids"}
        model_features = [
            {
                key: value
                for key, value in feature.items()
                if key in allowed_model_keys
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            label = list(label)
            if len(label) > sequence_length:
                if self.tokenizer.padding_side == "left":
                    label = label[-sequence_length:]
                else:
                    label = label[:sequence_length]

            padding_length = sequence_length - len(label)
            if self.tokenizer.padding_side == "left":
                padded_label = [self.label_pad_token_id] * padding_length + label
            else:
                padded_label = label + [self.label_pad_token_id] * padding_length
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
