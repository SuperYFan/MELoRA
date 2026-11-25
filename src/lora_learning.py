import os
import torch
import numpy as np
import random
import json
import pickle as pkl
import pandas as pd
from itertools import combinations
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
import nevergrad as ng
import transformers
from transformers import default_data_collator
from sklearn.metrics import roc_auc_score
import config
import utils
from few_hm_dataset import Few_HM_Data
from hfm_gen_eval import hfm_generation
from zh_fewshot_dataset import ZH_FewShot_Data, ZH_BENCHMARKS
from zh_modular_gen_eval import build_eval_data, zh_modular_generation
from zh_modular_dataset import generate_prompt_test
from case_study_utils import collect_binary_generation_outputs, save_case_study_records
import copy
import time

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def load_json(path):
    data=json.load(open(path,'rb'))
    return data
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,PeftConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
#from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

def load_base_model_and_lora_modules(args,
                                     lora_module_list,
                                     model_name_or_path):
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_peft_model_id = lora_module_list[0]
        
    load_started = time.perf_counter()
    print(f"> Loading base model: {args.base_model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(
        f"> Base model loaded in {time.perf_counter() - load_started:.1f}s",
        flush=True,
    )
    # load tokenizer
    tokenizer_started = time.perf_counter()
    print("> Loading tokenizer ...", flush=True)
    if getattr(base_model.config, "model_type", "") == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(
        f"> Tokenizer loaded in {time.perf_counter() - tokenizer_started:.1f}s",
        flush=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load the first adapter once, then attach the remaining adapters to the
    # same PEFT model. Re-wrapping the already modified base model is both slow
    # and can create nested PEFT wrappers.
    try:
        adapter_started = time.perf_counter()
        print(f"> Loading initial LoRA adapter: {default_peft_model_id}", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id,
                                               torch_dtype=torch.float16)
        print(
            f"> Initial LoRA adapter loaded in "
            f"{time.perf_counter() - adapter_started:.1f}s",
            flush=True,
        )
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
        
    #peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to cache LoRA module weights", flush=True)
    cache = {
        default_peft_model_id: get_peft_model_state_dict(
            peft_model, adapter_name="default"
        )
    }
    first_dict = cache[default_peft_model_id]
    parameter_count = sum(tensor.numel() for tensor in first_dict.values())
    print(
        f"> Cached {default_peft_model_id}: {len(first_dict)} tensors, "
        f"{parameter_count:,} parameters",
        flush=True,
    )

    for adapter_index, peft_model_id in enumerate(lora_module_list[1:], start=1):
        module_started = time.perf_counter()
        adapter_name = f"cached_expert_{adapter_index}"
        print("> Loading and caching {} ...".format(peft_model_id), flush=True)
        peft_model.load_adapter(peft_model_id, adapter_name=adapter_name)
        cache[peft_model_id] = get_peft_model_state_dict(
            peft_model,
            adapter_name=adapter_name,
        )
        parameter_count = sum(tensor.numel() for tensor in cache[peft_model_id].values())
        print(
            f"> Cached {peft_model_id}: {len(cache[peft_model_id])} tensors, "
            f"{parameter_count:,} parameters, "
            f"{time.perf_counter() - module_started:.1f}s",
            flush=True,
        )

        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
               
    print("> All base-model and LoRA loading steps completed.", flush=True)
    return peft_model, tokenizer, cache

def get_score(weights, tokenizer, model, cache, example_dataset, batch_size, get_loss, get_regular, logger):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = list(cache[lora_module_list[0]].keys())
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
        #print (peft_model_id,lora_state_dict[keys[-1]])
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(tokenizer,example_dataset, model, batch_size, logger)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    #metric_val=loss
    return metric_val

def default_get_loss(tokenizer,example_dataset,model, batch_size,logger):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    #print(len(example_dataset))
    data_batch_size =  batch_size
    # use gpu if available
    #data_collator=transformers.DataCollatorForSeq2Seq(
    #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #)
    train_dataloader = DataLoader(example_dataset,
                                  batch_size,
                                  shuffle=True)
    #print(len(train_dataloader))
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for iter, batch in enumerate(train_dataloader):
            #print (batch)
            #batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if hasattr(batch, "to"):
                    batch = batch.to(device)
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
            loss = outputs.loss
            #print(outputs.logits.shape)
            logger.write('%d-th iteration, loss %.2f' % (iter,loss))
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset)

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict

def safe_module_name(module_path):
    return os.path.basename(os.path.dirname(module_path)) + "_" + os.path.basename(module_path)

def save_individual_lora_predictions(model, tokenizer, args, lora_module_list, cache, logger):
    if args.DATASET not in ZH_BENCHMARKS:
        return
    base_case_dir = getattr(args, "case_study_dir", "LoRA/case_study")
    run_dir = os.path.join(base_case_dir, f"{args.DATASET}_{args.SAVE_NUM}", "individual_loras")
    for module_name in lora_module_list:
        set_peft_model_state_dict(model, cache[module_name])
        output_dir = os.path.join(run_dir, safe_module_name(module_name))
        invalid, auc, acc = zh_modular_generation(
            model,
            tokenizer,
            args,
            save_case_study=True,
            output_dir=output_dir,
        )
        logger.write(
            "Individual LoRA %s: invalid %d, AUC %.2f, Acc %.2f"
            % (module_name, len(invalid), auc, acc)
        )


def build_score_fusion_expert_specs(lora_module_list):
    if not lora_module_list:
        return []

    experts = []
    for module_name in lora_module_list:
        experts.append(
            {
                "name": f"single_{safe_module_name(module_name)}",
                "modules": [module_name],
            }
        )

    for left_module, right_module in combinations(lora_module_list, 2):
        experts.append(
            {
                "name": f"static_{safe_module_name(left_module)}_plus_{safe_module_name(right_module)}",
                "modules": [left_module, right_module],
            }
        )
    return experts


def materialize_expert_model(base_model, cache, expert_modules, expert_weights):
    model_copy = copy.deepcopy(base_model)
    if len(expert_modules) == 1:
        set_peft_model_state_dict(model_copy, cache[expert_modules[0]])
    else:
        subset_cache = {module_name: cache[module_name] for module_name in expert_modules}
        final_lora = get_final_weights(expert_weights, expert_modules, subset_cache)
        set_peft_model_state_dict(model_copy, final_lora)
    return model_copy.merge_and_unload()


def collect_generation_outputs_for_rows(model, tokenizer, rows, args):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(pad_token_id=tokenizer.pad_token_id)
    model.eval()
    outputs = collect_binary_generation_outputs(
        model=model,
        tokenizer=tokenizer,
        data=rows,
        generation_config=generation_config,
        generate_prompt_fn=generate_prompt_test,
        label_0_word=args.POS_WORD,
        label_1_word=args.NEG_WORD,
        progress_every=500,
        max_new_tokens=args.max_new_tokens,
    )
    model.train(True)
    return outputs


def compute_binary_metrics_from_scores(score_tensor, label_tensor):
    labels = label_tensor.view(-1).long().cpu().numpy()
    probabilities = torch.softmax(score_tensor, dim=-1)[:, 1].cpu().numpy()
    predictions = torch.argmax(score_tensor, dim=-1).cpu().numpy()
    auc = roc_auc_score(labels, probabilities) * 100.0
    acc = float((predictions == labels).mean() * 100.0)
    return auc, acc


def annotate_selected_expert_case_records(records, selected_expert_name):
    annotated_records = []
    for record in records:
        annotated = dict(record)
        annotated["fusion_model"] = "score_fusion_select"
        annotated["selected_expert"] = selected_expert_name
        annotated_records.append(annotated)
    return annotated_records


def get_expert_selection_key(result):
    return (
        float(result["calibration_auc"]),
        float(result["calibration_acc"]),
        -int(result["num_modules"]),
        result["name"],
    )


def optimize_static_prior_weights(
    model,
    tokenizer,
    cache,
    example_dataset,
    batch_size,
    max_inference_step,
    get_loss,
    get_regular,
    logger,
):
    lora_module_list = list(cache.keys())
    if len(lora_module_list) == 1:
        return [1.0]

    model_copy = copy.deepcopy(model)
    get_score_partial = partial(
        get_score,
        tokenizer=tokenizer,
        model=model_copy,
        cache=cache,
        example_dataset=example_dataset,
        batch_size=batch_size,
        get_loss=get_loss,
        get_regular=get_regular,
        logger=logger,
    )
    instrum = ng.p.Array(
        init=[0] * len(lora_module_list),
        upper=[1.5] * len(lora_module_list),
        lower=[-1.5] * len(lora_module_list),
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    logger.write("> Begin to perform gradient-free optimization for static prior weights ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    return list(recommendation.value)

def lorahub_learning(lora_module_list, 
                     args,
                     max_inference_step,
                     model_name_or_path=None,
                     batch_size=None,
                     get_loss=default_get_loss, 
                     get_regular=default_l1_regularization,
                     seed=42):
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    log_path='LoRA'
    if os.path.exists(log_path)==False:
        os.mkdir(log_path)
    if getattr(args, "stdout_only_log", False):
        logger = utils.Logger(None)
    else:
        logger=utils.Logger(os.path.join(log_path,args.DATASET+'_'+str(args.SAVE_NUM)+'.txt'))
    log_hyperpara(logger,args)
    
    #lora_module_list.append('haonan-li/bactrian-ru-llama-7b-lora')
    #lora_module_list.append('haonan-li/bactrian-cs-llama-7b-lora')
    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None
    logger.write('Number of module list in total: %d' % (number_of_loras))
    for i,module_name in enumerate(lora_module_list):
        logger.write('\tThe %d-th module: %s' % (i,module_name))
        
    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(args,
                                                               lora_module_list, 
                                                               model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "qwen" not in args.base_model.lower():
        tokenizer.pad_token_id = 0  # original LLaMA setting
    tokenizer.padding_side = "left"  # Allow batched inference
    # process dataset
    if args.DATASET in ZH_BENCHMARKS:
        dataset = ZH_FewShot_Data(args,tokenizer)
    else:
        dataset = Few_HM_Data(args,tokenizer)

    if getattr(args, "composition_method", "static") == "score_fusion":
        if args.DATASET not in ZH_BENCHMARKS:
            raise ValueError("score_fusion is currently implemented for Chinese benchmarks only.")
        if batch_size is None:
            batch_size = args.batch_size

        logger.write("LoRA composition method: calibration-based expert selection")
        calibration_rows = list(getattr(dataset, "entries", []))
        if not calibration_rows:
            raise ValueError("score_fusion requires accessible few-shot calibration rows.")
        eval_rows, eval_path = build_eval_data(args)
        logger.write("Expert selection calibration examples: %d" % len(calibration_rows))
        logger.write("Expert selection evaluation examples: %d" % len(eval_rows))
        logger.write("Expert selection evaluation path: %s" % eval_path)

        if getattr(args, "save_individual_lora_outputs", False):
            model_copy = copy.deepcopy(model)
            save_individual_lora_predictions(
                model_copy,
                tokenizer,
                args,
                lora_module_list,
                cache,
                logger,
            )

        expert_specs = build_score_fusion_expert_specs(lora_module_list)
        if not expert_specs:
            raise ValueError("score_fusion could not build any experts from the provided LoRA modules.")

        best_single_result = None
        best_pair_result = None

        for expert_spec in expert_specs:
            expert_modules = expert_spec["modules"]
            logger.write("Score fusion expert: %s" % expert_spec["name"])
            for module_name in expert_modules:
                logger.write("\tuses module: %s" % module_name)

            if len(expert_modules) == 1:
                expert_weights = [1.0]
            else:
                subset_cache = {module_name: cache[module_name] for module_name in expert_modules}
                expert_weights = optimize_static_prior_weights(
                    model=model,
                    tokenizer=tokenizer,
                    cache=subset_cache,
                    example_dataset=dataset,
                    batch_size=batch_size,
                    max_inference_step=max_inference_step,
                    get_loss=get_loss,
                    get_regular=get_regular,
                    logger=logger,
                )
            logger.write("Expert weights: %s" % ", ".join(f"{weight:.4f}" for weight in expert_weights))

            expert_model = materialize_expert_model(model, cache, expert_modules, expert_weights)
            calibration_outputs = collect_generation_outputs_for_rows(expert_model, tokenizer, calibration_rows, args)
            eval_outputs = collect_generation_outputs_for_rows(expert_model, tokenizer, eval_rows, args)

            calibration_auc, calibration_acc = compute_binary_metrics_from_scores(
                calibration_outputs["score_tensor"],
                calibration_outputs["label_tensor"],
            )
            expert_auc, expert_acc = compute_binary_metrics_from_scores(
                eval_outputs["score_tensor"],
                eval_outputs["label_tensor"],
            )
            logger.write(
                "Expert %s calibration AUC %.2f, Acc %.2f" % (
                    expert_spec["name"],
                    calibration_auc,
                    calibration_acc,
                )
            )
            logger.write(
                "Expert %s eval AUC %.2f, Acc %.2f" % (
                    expert_spec["name"],
                    expert_auc,
                    expert_acc,
                )
            )
            result = {
                "name": expert_spec["name"],
                "modules": list(expert_modules),
                "num_modules": len(expert_modules),
                "weights": list(expert_weights),
                "calibration_auc": calibration_auc,
                "calibration_acc": calibration_acc,
                "eval_auc": expert_auc,
                "eval_acc": expert_acc,
                "eval_outputs": eval_outputs,
            }
            if result["num_modules"] == 1:
                if best_single_result is None or get_expert_selection_key(result) > get_expert_selection_key(best_single_result):
                    best_single_result = result
                else:
                    del result["eval_outputs"]
            else:
                if best_pair_result is None or get_expert_selection_key(result) > get_expert_selection_key(best_pair_result):
                    best_pair_result = result
                else:
                    del result["eval_outputs"]

            del expert_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_single_result is not None:
            logger.write(
                "Best single expert on calibration: %s | AUC %.2f, Acc %.2f"
                % (
                    best_single_result["name"],
                    best_single_result["calibration_auc"],
                    best_single_result["calibration_acc"],
                )
            )
        if best_pair_result is not None:
            logger.write(
                "Best double expert on calibration: %s | AUC %.2f, Acc %.2f"
                % (
                    best_pair_result["name"],
                    best_pair_result["calibration_auc"],
                    best_pair_result["calibration_acc"],
                )
            )

        selected_result = best_single_result
        if best_pair_result is not None and (
            selected_result is None or get_expert_selection_key(best_pair_result) > get_expert_selection_key(selected_result)
        ):
            selected_result = best_pair_result
        if selected_result is None:
            raise ValueError("score_fusion could not select any valid expert.")

        logger.write(
            "Selected expert: %s | modules=%d | calibration AUC %.2f, Acc %.2f"
            % (
                selected_result["name"],
                selected_result["num_modules"],
                selected_result["calibration_auc"],
                selected_result["calibration_acc"],
            )
        )
        for module_name, weight in zip(selected_result["modules"], selected_result["weights"]):
            logger.write("Selected module %s weight %.4f" % (module_name, float(weight)))

        selected_invalid_cases = selected_result["eval_outputs"]["invalid"]
        if getattr(args, "save_case_study", False):
            case_output_dir = os.path.join(
                getattr(args, "case_study_dir", "LoRA/case_study"),
                f"{args.DATASET}_{args.SAVE_NUM}_score_fusion",
            )
            selected_records = annotate_selected_expert_case_records(
                selected_result["eval_outputs"]["records"],
                selected_result["name"],
            )
            save_paths = save_case_study_records(
                records=selected_records,
                output_dir=case_output_dir,
                stem="eval_predictions",
                title=f"{args.DATASET} Expert Selection Case Study",
            )
            logger.write("Saved case study files:")
            for path in save_paths.values():
                logger.write("\t%s" % path)

        if selected_invalid_cases:
            logger.write('\tThere are %d invalid examples' % len(selected_invalid_cases))
        logger.write('AUC %.2f, Acc %.2f' % (selected_result["eval_auc"], selected_result["eval_acc"]))
        return selected_result, model, tokenizer

    if getattr(args, "composition_method", "static") == "moe":
        from moe_lora import (
            collect_moe_aux_loss,
            count_moe_router_modules,
            install_moe_lora_adapters,
            save_moe_router_state,
            train_moe_routers,
        )

        logger.write("LoRA composition method: adaptive layer-wise MoE-LoRA")

        if number_of_loras == 1:
            logger.write("Single LoRA MoE reduces to the raw adapter weight 1.0; skip router training.")
            model_copy = copy.deepcopy(model)
            set_peft_model_state_dict(model_copy, cache[lora_module_list[0]])
            model_copy = model_copy.merge_and_unload()

            if args.DATASET in ZH_BENCHMARKS:
                case_output_dir = None
                if getattr(args, "save_case_study", False):
                    case_output_dir = os.path.join(
                        getattr(args, "case_study_dir", "LoRA/case_study"),
                        f"{args.DATASET}_{args.SAVE_NUM}_moe",
                    )
                invalid,auc,acc=zh_modular_generation(
                    model_copy,
                    tokenizer,
                    args,
                    save_case_study=getattr(args, "save_case_study", False),
                    output_dir=case_output_dir,
                )
            else:
                invalid,auc,acc=hfm_generation(model_copy,tokenizer,args)
            if len(invalid)>0:
                logger.write('\tThere are %d invalid examples' % (len(invalid)))
            logger.write('AUC %.2f, Acc %.2f' % (auc,acc))
            return [1.0], model, tokenizer

        if batch_size is None:
            batch_size = args.batch_size
        replaced = install_moe_lora_adapters(
            model,
            cache,
            router_temperature=getattr(args, "moe_temperature", 1.0),
            top_k=getattr(args, "moe_top_k", 0),
            router_type=getattr(args, "moe_router_type", "linear"),
            router_hidden_size=getattr(args, "moe_router_hidden_size", 128),
            router_granularity=getattr(args, "moe_router_granularity", "token"),
        )
        router_count = count_moe_router_modules(model)
        logger.write(
            "Installed adaptive MoE-LoRA into %d LoRA layers with %d router instances"
            % (replaced, router_count)
        )
        logger.write(
            "MoE router type: %s, hidden size: %s, top-k: %s, granularity: %s"
            % (
                getattr(args, "moe_router_type", "linear"),
                getattr(args, "moe_router_hidden_size", 128),
                getattr(args, "moe_top_k", 0),
                getattr(args, "moe_router_granularity", "token"),
            )
        )
        train_moe_routers(
            model=model,
            example_dataset=dataset,
            batch_size=batch_size,
            epochs=getattr(args, "moe_router_epochs", 20),
            learning_rate=getattr(args, "moe_router_lr", 1e-3),
            max_steps=getattr(args, "moe_router_max_steps", 0),
            load_balance_coeff=getattr(args, "moe_load_balance_coeff", 0.0),
            logger=logger,
        )
        if getattr(args, "save_moe_router", False):
            router_path = os.path.join(
                getattr(args, "moe_router_dir", "LoRA/moe_routers"),
                "%s_%s_router.pt" % (args.DATASET, args.SAVE_NUM),
            )
            save_moe_router_state(model, router_path)
            logger.write("Saved MoE-LoRA router parameters to: %s" % router_path)
        if getattr(args, "save_individual_lora_outputs", False):
            logger.write("save_individual_lora_outputs is skipped for MoE-LoRA because experts are routed inside each layer.")

        if args.DATASET in ZH_BENCHMARKS:
            case_output_dir = None
            if getattr(args, "save_case_study", False):
                case_output_dir = os.path.join(
                    getattr(args, "case_study_dir", "LoRA/case_study"),
                    f"{args.DATASET}_{args.SAVE_NUM}_moe",
                )
            invalid,auc,acc=zh_modular_generation(
                model,
                tokenizer,
                args,
                save_case_study=getattr(args, "save_case_study", False),
                output_dir=case_output_dir,
            )
        else:
            invalid,auc,acc=hfm_generation(model,tokenizer,args)
        if len(invalid)>0:
            logger.write('\tThere are %d invalid examples' % (len(invalid)))
        logger.write('AUC %.2f, Acc %.2f' % (auc,acc))
        return None, model, tokenizer

    if number_of_loras == 1:
        model_copy = copy.deepcopy(model)
        module_name = lora_module_list[0]
        logger.write("Single LoRA static composition: skip weight search and use raw adapter weight 1.0")
        set_peft_model_state_dict(model_copy, cache[module_name])
        model_copy = model_copy.merge_and_unload()
        logger.write('Module name: %s, module weight %f' % (module_name, 1.0))

        if args.DATASET in ZH_BENCHMARKS:
            case_output_dir = None
            if getattr(args, "save_case_study", False):
                case_output_dir = os.path.join(
                    getattr(args, "case_study_dir", "LoRA/case_study"),
                    f"{args.DATASET}_{args.SAVE_NUM}",
                )
            invalid,auc,acc=zh_modular_generation(
                model_copy,
                tokenizer,
                args,
                save_case_study=getattr(args, "save_case_study", False),
                output_dir=case_output_dir,
            )
        else:
            invalid,auc,acc=hfm_generation(model_copy,tokenizer,args)
        if len(invalid)>0:
            logger.write('\tThere are %d invalid examples' % (len(invalid)))
        logger.write('AUC %.2f, Acc %.2f' % (auc,acc))
        return [1.0], model, tokenizer

    recommendation_value = optimize_static_prior_weights(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        example_dataset=dataset,
        batch_size=batch_size,
        max_inference_step=max_inference_step,
        get_loss=get_loss,
        get_regular=get_regular,
        logger=logger,
    )
    #lora_module_list.pop(-1)
    if getattr(args, "save_individual_lora_outputs", False):
        model_copy = copy.deepcopy(model)
        save_individual_lora_predictions(
            model_copy,
            tokenizer,
            args,
            lora_module_list,
            cache,
            logger,
        )
    final_lora = get_final_weights(recommendation_value, lora_module_list, cache)
    model_copy = copy.deepcopy(model)
    # set the final weights
    set_peft_model_state_dict(model_copy, final_lora)
    model_copy = model_copy.merge_and_unload()
    for i,name in enumerate(lora_module_list):
        logger.write('Module name: %s, module weight %f' % (name,recommendation_value[i]))
    #print (recommendation.value)
    if args.DATASET in ZH_BENCHMARKS:
        case_output_dir = None
        if getattr(args, "save_case_study", False):
            case_output_dir = os.path.join(
                getattr(args, "case_study_dir", "LoRA/case_study"),
                f"{args.DATASET}_{args.SAVE_NUM}",
            )
        invalid,auc,acc=zh_modular_generation(
            model_copy,
            tokenizer,
            args,
            save_case_study=getattr(args, "save_case_study", False),
            output_dir=case_output_dir,
        )
    else:
        invalid,auc,acc=hfm_generation(model_copy,tokenizer,args)
    if len(invalid)>0:
        logger.write('\tThere are %d invalid examples' % (len(invalid)))
    logger.write('AUC %.2f, Acc %.2f' % (auc,acc))
    return recommendation_value, model, tokenizer
