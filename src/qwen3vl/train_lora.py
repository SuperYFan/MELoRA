"""
Train a single PEFT LoRA module on Qwen3-VL-8B-Instruct for one of three tasks:

  lora1  – text-only input  → binary classification
             Training data: zh_modular/modules/hate_semantics/train.json
  lora2  – image input      → reasoning-process text output
             Training data: zh_modular/modules/met_m3_toxicn_reasoning/train.json
  lora3  – image input      → visual-description text output
             Training data: zh_modular/modules/visual_expression/train.json

The saved checkpoint is a standard PEFT adapter checkpoint.

Usage (run from repo root)
--------------------------
python src/qwen3vl/train_lora.py \
    --module lora2 \
    --base_model models/Qwen3-VL-8B-Instruct \
    --repo_root  . \
    --data_path  data \
    --output_dir ./lora_modules_vl/lora2
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow importing sibling modules from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
try:
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration as _AutoVLM  # native
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM  # type: ignore[assignment]
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM  # type: ignore[assignment]

from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Module configuration
# ---------------------------------------------------------------------------

MODULE_CONFIG = {
    "lora1": {
        "data_subpath": "zh_modular/modules/hate_semantics/train.json",
        "use_image": False,
        "description": "text-only binary classification",
    },
    "lora2": {
        "data_subpath": "zh_modular/modules/met_m3_toxicn_reasoning/train.json",
        "use_image": True,
        "description": "image -> reasoning-process text",
    },
    "lora3": {
        "data_subpath": "zh_modular/modules/visual_expression/train.json",
        "use_image": True,
        "description": "image -> visual-description text",
    },
}

# Qwen3-VL chat template assistant marker (used for label masking)
ASSISTANT_MARKER = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VLMLoRADataset(Dataset):
    """
    Dataset for supervised fine-tuning of Qwen3-VL with LoRA.
    Each sample is pre-processed into (input_ids, labels, optional pixel tensors).
    Labels have the user/system prefix masked with -100 so loss is only on
    the assistant response.
    """

    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        repo_root: str,
        use_image: bool,
        max_length: int = 1024,
        max_pixels: int = 200704,
        debug: bool = False,
    ):
        rows = json.loads(Path(data_path).read_text("utf-8"))
        # Filter rows with missing output
        self.rows = [r for r in rows if r.get("output")]
        if debug:
            self.rows = self.rows[:32]

        self.processor = processor
        self.repo_root  = Path(repo_root)
        self.use_image  = use_image
        self.max_length = max_length
        self.max_pixels = max_pixels  # hard pixel cap applied before processor call

        # Tokenise the assistant marker once to find its length
        self._marker_ids = processor.tokenizer(
            ASSISTANT_MARKER, add_special_tokens=False
        ).input_ids
        print(f"  Loaded {len(self.rows)} training samples (use_image={use_image})")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image(self, image_ref: str) -> Optional[Image.Image]:
        if not image_ref:
            return None
        p = Path(image_ref)
        if not p.is_absolute():
            p = self.repo_root / image_ref
        if not p.exists():
            return None
        img = Image.open(str(p)).convert("RGB")
        # Hard-resize to stay within visual-token budget regardless of
        # processor-level max_pixels support (worker-safe).
        w, h = img.size
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return img

    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        """Return the index of the first response token (after the assistant marker)."""
        marker = self._marker_ids
        mlen = len(marker)
        for i in range(len(input_ids) - mlen, -1, -1):
            if input_ids[i: i + mlen].tolist() == marker:
                return i + mlen
        return len(input_ids)   # fallback: mask all

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        instruction  = str(row.get("instruction") or "").strip()
        input_text   = str(row.get("input") or "").strip()
        output_text  = str(row.get("output") or "").strip()
        image_ref    = str(row.get("image") or "").strip()

        # ----------------------------------------------------------------
        # Build the full conversation and user-only conversation
        # ----------------------------------------------------------------
        if self.use_image:
            image = self._resolve_image(image_ref)
            user_content: List[Dict[str, Any]] = [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ]
            images_list = [image] if image is not None else []
        else:
            image       = None
            images_list = []
            text_body   = (instruction + "\n" + input_text).strip() if input_text else instruction
            user_content = [{"type": "text", "text": text_body}]

        user_messages  = [{"role": "user",      "content": user_content}]
        full_messages  = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": output_text}]},
        ]

        # Apply chat template
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # ----------------------------------------------------------------
        # Tokenise the full conversation
        # ----------------------------------------------------------------
        proc_kwargs: Dict[str, Any] = dict(
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        if images_list:
            proc_kwargs["images"] = images_list

        encoding = self.processor(**proc_kwargs)

        input_ids = encoding["input_ids"][0]          # (L,)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask[0]
        else:
            attention_mask = torch.ones_like(input_ids)

        # ----------------------------------------------------------------
        # Label masking: -100 for everything up to and including the
        # assistant marker; compute loss only on the response tokens.
        # ----------------------------------------------------------------
        labels = input_ids.clone()
        response_start = self._find_response_start(input_ids)
        labels[:response_start] = -100

        result: Dict[str, torch.Tensor] = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

        # Qwen3-VL M-RoPE requires mm_token_type_ids (always present when images passed)
        if "mm_token_type_ids" in encoding:
            result["mm_token_type_ids"] = encoding["mm_token_type_ids"][0]

        if images_list and "pixel_values" in encoding:
            result["pixel_values"] = encoding["pixel_values"]   # (patches, C, H, W)
            if "image_grid_thw" in encoding:
                result["image_grid_thw"] = encoding["image_grid_thw"]  # (1, 3)

        return result


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForVLM:
    """
    Pads input_ids / labels / attention_mask to the longest sequence in the
    batch.  Pixel tensors (variable # of patches per image) are concatenated
    along dim=0 as required by Qwen2-VL.
    """

    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)
        batch: Dict[str, Any] = {}

        # Pad sequence tensors (include mm_token_type_ids for Qwen3-VL M-RoPE)
        seq_keys = [("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)]
        if "mm_token_type_ids" in features[0]:
            seq_keys.append(("mm_token_type_ids", 0))
        for key, pad_value in seq_keys:
            tensors = []
            for f in features:
                t = f[key]
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    t = torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)])
                tensors.append(t)
            batch[key] = torch.stack(tensors)

        # Concatenate visual tensors (only present for image LoRAs)
        if "pixel_values" in features[0]:
            batch["pixel_values"]   = torch.cat([f["pixel_values"] for f in features], dim=0)
            if "image_grid_thw" in features[0]:
                batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        return batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        # Target language-model attention projections only.
        # Qwen2-VL vision encoder uses different naming (not matched here).
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    cfg = MODULE_CONFIG[args.module]
    data_path = Path(args.data_path) / cfg["data_subpath"]
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"Module   : {args.module} ({cfg['description']})")
    print(f"Data     : {data_path}")
    print(f"Model    : {args.base_model}")
    print(f"Output   : {args.output_dir}")

    # ------------------------------------------------------------------
    # Load processor and model
    # ------------------------------------------------------------------
    # Pass max_pixels via from_pretrained (official API) so the image processor
    # respects the pixel budget at all levels including worker processes.
    proc_kwargs: Dict[str, Any] = dict(trust_remote_code=True)
    if cfg["use_image"]:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.base_model, **proc_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = _AutoVLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Apply PEFT LoRA
    lora_cfg = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset and collator
    # ------------------------------------------------------------------
    dataset = VLMLoRADataset(
        data_path=str(data_path),
        processor=processor,
        repo_root=args.repo_root,
        use_image=cfg["use_image"],
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        debug=args.debug,
    )

    collator = DataCollatorForVLM(
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,   # keep pixel_values etc.
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Save PEFT adapter (loadable with PeftModel.from_pretrained)
    # ------------------------------------------------------------------
    adapter_dir = Path(args.output_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\nPEFT adapter saved to: {adapter_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen3-VL LoRA module")
    parser.add_argument("--module", required=True, choices=list(MODULE_CONFIG.keys()),
                        help="Which LoRA module to train: lora1 | lora2 | lora3")

    # Paths
    parser.add_argument("--base_model", type=str,
                        default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--repo_root", type=str,
                        default=".",
                        help="Repo root used to resolve relative image paths")
    parser.add_argument("--data_path", type=str,
                        default="data")
    parser.add_argument("--output_dir", type=str, required=True)

    # LoRA hyper-params
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training hyper-params
    parser.add_argument("--num_epochs",    type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=1,
                        help="Per-device batch size. Keep at 1 for image LoRAs to avoid "
                             "variable-patch batching issues; use --grad_accum instead.")
    parser.add_argument("--grad_accum",    type=int,   default=16,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum).")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=50)
    parser.add_argument("--max_length",    type=int,   default=1024)
    parser.add_argument("--max_pixels",    type=int,   default=200704,
                        help="Max pixels per image (H*W). Images above this are resized. "
                             "200704 = 448x448 ≈ 256 visual tokens with 28x28 effective patches.")
    parser.add_argument("--logging_steps", type=int,   default=20)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--debug",         action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
