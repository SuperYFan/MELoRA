import os
import torch
import numpy as np
import transformers
import random
import json
import pickle as pkl

import config
from data_collator import CausalLMDataCollator
from train import train_for_epochs

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
if __name__ == "__main__":
    args=config.parse_opt()
    set_seed(args.SEED)
    
    device_map = "auto"
    #model initialization
    if args.load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=args.use_gradient_checkpointing)
    # Qwen2.5 uses q_proj, k_proj, v_proj, o_proj; LLaMA uses q_proj, v_proj
    if "qwen" in args.base_model.lower():
        lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        lora_target = ["q_proj", "v_proj"]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    #tokenizer instantiation
    if model.config.model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # Set pad token: use eos_token for Qwen, 0 (unk) for LLaMA
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "qwen" not in args.base_model.lower():
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference
    
    #data preparation
    if args.DATASET in ['mem','harm','mimc']:
        from hm_dataset import HM_Data
        train_cls=HM_Data(args,tokenizer,mode='train')
        val_cls=HM_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['meme-captions','hatred']:
        from gen_dataset import GEN_Data
        train_cls=GEN_Data(args,tokenizer,mode='train')
        val_cls=GEN_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['hate-speech']:
        from text_dataset import TEXT_Data
        train_cls=TEXT_Data(args,tokenizer,mode='train')
        val_cls=TEXT_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['toxicn']:
        from toxicn_dataset import ToxiCN_Data
        train_cls=ToxiCN_Data(args,tokenizer,mode='train')
        val_cls=ToxiCN_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['sarcnet']:
        from sarcnet_dataset import SarcNet_Data
        train_cls=SarcNet_Data(args,tokenizer,mode='train')
        val_cls=SarcNet_Data(args,tokenizer,mode='test')
    elif args.DATASET in ['zh-hate-semantics', 'zh-sarcasm-semantics', 'zh-met-meme-reasoning',
                          'zh-met-m3-toxicn-reasoning', 'zh-met-m3-sarcnet-reasoning',
                          'zh-visual-expression',
                          'zh-toxicn-mm', 'zh-misogyny-meme', 'zh-sarcnet']:
        from zh_modular_dataset import ZH_Modular_Data
        train_cls=ZH_Modular_Data(args,tokenizer,mode='train')
        val_cls=ZH_Modular_Data(args,tokenizer,mode='test')
    else:
        raise ValueError(f"Unsupported dataset: {args.DATASET}")
    train_set=train_cls.get_dataset()
    val_set=val_cls.get_dataset()
    data_collator=CausalLMDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    train_for_epochs(model,tokenizer,args,  
                     data_collator,
                     train_set,val_set)
