import argparse 
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, 
                        default='set the path')
    parser.add_argument('--DATASET', type=str, 
                        default='mem')
    parser.add_argument('--SEED', type=int, 
                        default=1111)
    parser.add_argument('--SAVE_NUM', type=int, 
                        default=1)
    parser.add_argument('--DEBUG', type=bool, 
                        default=False)
    parser.add_argument('--NEG_WORD', type=str, 
                        default='Yes')
    parser.add_argument('--POS_WORD', type=str, 
                        default='No')

    #new added settings for LoRA
    parser.add_argument('--max_inference_step', type=int, 
                        default=40)
    parser.add_argument('--num_shots', type=int, 
                        default=4)
    parser.add_argument('--lora_dir', type=str, 
                        default='set the path')
    parser.add_argument('--lora_modules', type=str, 
                        default='hate-exp,meme-captions,hate-speech')
    parser.add_argument('--eval_split', type=str,
                        default='auto',
                        help='Evaluation split for Chinese benchmarks: auto, test, dev, or test_unlabeled.')
    parser.add_argument('--composition_method', type=str,
                        choices=['static', 'moe', 'score_fusion'],
                        default='static',
                        help='LoRA composition method: static weighted sum, adapter-level MoE-LoRA, or output-level score fusion.')
    parser.add_argument('--moe_router_lr', type=float,
                        default=1e-3,
                        help='Learning rate for adapter-level MoE-LoRA routers.')
    parser.add_argument('--moe_router_epochs', type=int,
                        default=20,
                        help='Number of few-shot epochs for MoE-LoRA router calibration.')
    parser.add_argument('--moe_router_max_steps', type=int,
                        default=0,
                        help='Maximum MoE router update steps. 0 means no step limit.')
    parser.add_argument('--moe_top_k', type=int,
                        default=0,
                        help='Use sparse top-k expert routing for MoE-LoRA. 0 means dense routing.')
    parser.add_argument('--moe_temperature', type=float,
                        default=1.0,
                        help='Softmax temperature for MoE-LoRA router gates.')
    parser.add_argument('--moe_router_type', type=str,
                        choices=['linear', 'mlp'],
                        default='linear',
                        help='Router architecture for MoE-LoRA: linear or mlp.')
    parser.add_argument('--moe_router_hidden_size', type=int,
                        default=128,
                        help='Hidden size for mlp MoE-LoRA router. Ignored by linear router.')
    parser.add_argument('--moe_router_granularity', type=str,
                        choices=['token', 'sample'],
                        default='token',
                        help='Routing granularity for MoE-LoRA: token-wise or sample-wise.')
    parser.add_argument('--moe_load_balance_coeff', type=float,
                        default=0.0,
                        help='Coefficient for MoE load-balancing auxiliary loss (Switch Transformer style). '
                             '0 disables it. Typical values: 0.01-0.1.')
    parser.add_argument('--calibration_ratio', type=float,
                        default=0.0,
                        help='Fraction of the training split to use for MoE router calibration. '
                             'When > 0, overrides --num_shots and uses a random subset of the '
                             'full training data (natural class distribution). '
                             'E.g. 0.05 = 5%%, 0.20 = 20%%, 0.50 = 50%%.')
    parser.add_argument('--save_moe_router', action='store_true',
                        help='Save trained MoE-LoRA router parameters.')
    parser.add_argument('--moe_router_dir', type=str,
                        default='LoRA/moe_routers',
                        help='Directory for saved MoE-LoRA router parameters.')

    #fine-tuning LLaMA related
    parser.add_argument('--base_model', type=str, 
                        default='yahma/llama-7b-hf')
    parser.add_argument('--load_8bit', type=bool, 
                        default=False)
    parser.add_argument('--batch_size', type=int, 
                        default=16)
    parser.add_argument('--micro_batch_size', type=int, 
                        default=4)
    parser.add_argument('--logging_steps', type=int, 
                        default=5)
    parser.add_argument('--save_total_limit', type=int, 
                        default=5)
    parser.add_argument('--warmup_steps', type=int, 
                        default=100)
    parser.add_argument('--num_epochs', type=int, 
                        default=2)
    parser.add_argument('--learning_rate', type=float, 
                        default=5e-4)
    parser.add_argument('--cutoff_len', type=int, help='Maximum sequence length to process.',
                        default=256)
    parser.add_argument('--use_gradient_checkpointing', type=bool, 
                        default=False)
    parser.add_argument('--group_by_length', type=bool, 
                        default=False)
    parser.add_argument('--fp16', type=bool, 
                        default=True)

    parser.add_argument('--lora_r', type=int, default=8,
                        help='curvature.')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='The initialization coefficient of lora-alpha.')  
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='The initialization coefficient of lora_dropout.')
    parser.add_argument('--target_modules', type=str, 
                        default=r'.*language_model.*\.(q_proj|v_proj)')
    parser.add_argument('--train_on_inputs', type=bool, 
                        default=False)

    #saving and evaluation
    parser.add_argument('--eval_step', type=int, 
                        default=50)
    parser.add_argument('--gen_step', type=int, 
                        default=200)
    parser.add_argument('--gen_start', type=int, 
                        default=350)
    parser.add_argument('--val_set_size', type=int, 
                        default=500)
    parser.add_argument('--save_step', type=int, 
                        default=100)
    parser.add_argument('--output_dir', type=str, 
                        default='set the path')
    parser.add_argument('--resume_from_checkpoint', type=str, 
                         default=None)
    parser.add_argument('--save_case_study', action='store_true',
                        help='Save per-example predictions for Chinese LoRA benchmark evaluation.')
    parser.add_argument('--save_individual_lora_outputs', action='store_true',
                        help='Also save per-example benchmark predictions for each individual LoRA module.')
    parser.add_argument('--case_study_dir', type=str,
                        default='LoRA/case_study',
                        help='Directory for saved LoRA benchmark predictions.')
    parser.add_argument('--stdout_only_log', action='store_true',
                        help='Write training/eval logs to stdout only instead of creating a per-run log file.')

    #generation related
    parser.add_argument('--temperature', type=float, 
                        default=0.5)
    parser.add_argument('--top_p', type=float, 
                        default=0.75)
    parser.add_argument('--top_k', type=int, 
                        default=40)
    parser.add_argument('--num_beams', type=int, 
                        default=4)
    parser.add_argument('--max_new_tokens', type=int, 
                        default=128)
    args=parser.parse_args()
    return args
