import os
import json
import pickle as pkl
import numpy as np
import torch
import random

from transformers import LlamaTokenizer
from datasets import load_dataset

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(f)
    return data

def process_sarcnet_data(rows):
    instruct="请根据文本内容判断该帖子是否包含讽刺。"
    label_mapper={
        0:'否',
        1:'是'
    }
    lines=[]
    for row in rows:
        img=row['path']
        label=row['multi_label']
        ans=label_mapper[label]
        text=row['text']
        cur_row={
            'img':img,
            'input':'文本内容:'+text,
            'output':ans,
            'instruction':instruct
        }
        lines.append(cur_row)
    return lines

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:{data_point["instruction"]}
                
                ### Input:{data_point["input"]}
                
                ### Response:{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:{data_point["instruction"]}
                
                ### Response:{data_point["output"]}"""

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
    else:
        return result

class SarcNet_Data():
    def __init__(self, opt, tokenizer, mode='train'):
        super(SarcNet_Data, self).__init__()
        self.opt = opt
        self.mode = mode
        self.instruct = "请根据文本内容判断该帖子是否包含讽刺。"
        self.label_mapper = {
            0: self.opt.POS_WORD,
            1: self.opt.NEG_WORD
        }
        print('Initializing tokenizer from:', self.opt.base_model)
        print('\tVerbalizer:', self.opt.POS_WORD, self.opt.NEG_WORD)
        self.tokenizer = tokenizer

    def organize_data(self, save_path):
        if self.mode == 'train':
            json_file = 'train_data.json'
        elif self.mode == 'test':
            json_file = 'test_data.json'
        else:
            json_file = 'validation_data.json'
        data_raw = load_json(os.path.join(self.opt.PATH, 'sarcnet', json_file))
        data = process_sarcnet_data(data_raw)
        print('Processing files and dumping....')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt, self.tokenizer,
                                         self.opt.cutoff_len, self.opt.base_model)
        if not self.opt.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, self.tokenizer,
                                             self.opt.cutoff_len, self.opt.base_model,
                                             add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    def get_dataset(self):
        if self.mode == 'train':
            up_file = 'sarcnet_train_up.json'
        elif self.mode == 'test':
            up_file = 'sarcnet_test_up.json'
        else:
            up_file = 'sarcnet_val_up.json'
        load_path = os.path.join(self.opt.PATH, 'sarcnet', up_file)
        if not os.path.exists(load_path):
            self.organize_data(load_path)
        data = load_dataset("json", data_files=load_path)
        print('Loading data from %s mode, from %s' % (self.mode, load_path))
        if self.opt.DEBUG:
            print('Select a subset of data...')
            data["train"] = data["train"].select([i for i in range(16)])
        wrap_data = data['train'].shuffle().map(self.generate_and_tokenize_prompt)
        return wrap_data
