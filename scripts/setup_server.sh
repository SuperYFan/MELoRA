#!/bin/bash
# ============================================
# Qwen2.5-7B 环境配置 + 部署脚本
# 安装目录，默认放在当前目录下（可用 ROOT_DIR 覆盖）
# ============================================

set -e

ROOT_DIR=${ROOT_DIR:-$(pwd)}
MODEL_DIR=${ROOT_DIR}/models/Qwen2.5-7B

echo "===== Step 1: 创建目录 ====="
mkdir -p ${ROOT_DIR}/models
mkdir -p ${ROOT_DIR}

echo "===== Step 2: 安装依赖 ====="
pip install -q transformers==4.44.0
pip install -q peft==0.12.0
pip install -q datasets==2.20.0
pip install -q accelerate==0.33.0
pip install -q bitsandbytes==0.43.0
pip install -q scikit-learn nevergrad sentencepiece protobuf

echo "===== Step 3: 下载 Qwen2.5-7B ====="
# 使用 wget 逐文件断点续传，避免 Python 下载器占用过多内存被系统 kill。
mkdir -p ${MODEL_DIR}
BASE_URL="https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main"

FILES=(
    "config.json"
    "generation_config.json"
    "model.safetensors.index.json"
    "model-00001-of-00004.safetensors"
    "model-00002-of-00004.safetensors"
    "model-00003-of-00004.safetensors"
    "model-00004-of-00004.safetensors"
    "merges.txt"
    "vocab.json"
    "tokenizer.json"
    "tokenizer_config.json"
)

for file in "${FILES[@]}"; do
    echo "下载 ${file}"
    wget -c --tries=20 --waitretry=5 -P "${MODEL_DIR}" "${BASE_URL}/${file}"
done

echo "模型下载完成!"

echo "===== Step 4: 验证环境 ====="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('显存:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')

import transformers; print('Transformers:', transformers.__version__)
import peft; print('PEFT:', peft.__version__)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen2.5-7B', trust_remote_code=True)
print('Qwen2.5 vocab size:', tok.vocab_size)
yes_id = tok.encode('是', add_special_tokens=False)
no_id = tok.encode('否', add_special_tokens=False)
print(f'Token IDs - 是: {yes_id}, 否: {no_id}')
print('环境验证通过!')
"
