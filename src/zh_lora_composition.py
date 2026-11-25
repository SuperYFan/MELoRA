import os
import random
from pathlib import Path

import numpy as np
import torch

import lora_config
from lora_learning import lorahub_learning


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_lora_path(lora_dir, module):
    module_path = Path(module)
    if module_path.is_absolute() or module_path.exists():
        return str(module_path)
    return str(Path(lora_dir) / module)


if __name__ == "__main__":
    args = lora_config.parse_opt()
    set_seed(args.SEED)

    modules = [m.strip() for m in args.lora_modules.split(",") if m.strip()]
    lora_module_list = [resolve_lora_path(args.lora_dir, module) for module in modules]
    lorahub_learning(
        lora_module_list,
        args,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size,
    )
