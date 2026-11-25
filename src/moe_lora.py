import os
import re
from collections.abc import Mapping
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _module_tail(module_name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix):]
    return module_name


def _find_lora_weight(state_dict: Dict[str, torch.Tensor], module_name: str, kind: str) -> torch.Tensor:
    module_names = [module_name, _module_tail(module_name)]
    suffixes = []
    for name in module_names:
        suffixes.extend(
            [
                f"{name}.{kind}.weight",
                f"{name}.{kind}.default.weight",
            ]
        )

    for suffix in suffixes:
        if suffix in state_dict:
            return state_dict[suffix]
    for key, value in state_dict.items():
        if any(key.endswith(suffix) for suffix in suffixes):
            return value

    escaped = [re.escape(name) for name in module_names]
    pattern = re.compile(rf"({'|'.join(escaped)})\.{kind}(?:\.[^.]+)?\.weight$")
    for key, value in state_dict.items():
        if pattern.search(key):
            return value

    raise KeyError(f"Cannot find {kind} weights for LoRA module: {module_name}")


def _get_first_mapping_value(value, default=None):
    if isinstance(value, Mapping):
        return next(iter(value.values()), default)
    return value if value is not None else default


def _first_parameter(module: nn.Module):
    try:
        return next(module.parameters())
    except StopIteration:
        return None


def _get_lora_scaling(peft_lora_layer, default: float = 1.0) -> float:
    scaling = getattr(peft_lora_layer, "scaling", None)
    scaling = _get_first_mapping_value(scaling, default)
    try:
        return float(scaling)
    except (TypeError, ValueError):
        return default


def _get_lora_dropout_p(peft_lora_layer) -> float:
    dropout = getattr(peft_lora_layer, "lora_dropout", None)
    dropout = _get_first_mapping_value(dropout, None)
    return float(getattr(dropout, "p", 0.0) or 0.0)


def _get_base_layer(peft_lora_layer: nn.Module) -> nn.Module:
    base_layer = getattr(peft_lora_layer, "base_layer", None)
    if base_layer is None and hasattr(peft_lora_layer, "get_base_layer"):
        base_layer = peft_lora_layer.get_base_layer()
    if base_layer is None:
        raise ValueError("PEFT LoRA layer does not expose base_layer.")
    return base_layer


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _build_router(
    in_features: int,
    num_experts: int,
    router_type: str,
    router_hidden_size: int,
    device,
) -> nn.Module:
    router_type = (router_type or "linear").lower()
    if router_type == "linear":
        router = nn.Linear(in_features, num_experts, bias=True, device=device, dtype=torch.float32)
        nn.init.normal_(router.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(router.bias)
        return router

    if router_type == "mlp":
        hidden_size = int(router_hidden_size or 0)
        if hidden_size <= 0:
            hidden_size = min(256, in_features)
        router = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=True, device=device, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_size, num_experts, bias=True, device=device, dtype=torch.float32),
        )
        nn.init.xavier_uniform_(router[0].weight)
        nn.init.zeros_(router[0].bias)
        nn.init.normal_(router[2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(router[2].bias)
        return router

    raise ValueError(f"Unsupported MoE router type: {router_type}. Use 'linear' or 'mlp'.")


def _apply_top_k(probabilities: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = int(top_k or 0)
    if top_k <= 0 or top_k >= probabilities.shape[-1]:
        return probabilities
    values, indices = torch.topk(probabilities, k=top_k, dim=-1)
    mask = torch.zeros_like(probabilities)
    mask.scatter_(-1, indices, values)
    return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class AdaptiveMoeLoraLinear(nn.Module):
    """Layer-wise MoE-LoRA with frozen experts and a trainable router."""

    def __init__(
        self,
        peft_lora_layer: nn.Module,
        expert_names: List[str],
        expert_a: List[torch.Tensor],
        expert_b: List[torch.Tensor],
        scaling: float,
        dropout_p: float,
        router_temperature: float = 1.0,
        top_k: int = 0,
        router_type: str = "linear",
        router_hidden_size: int = 128,
        router_granularity: str = "token",
    ) -> None:
        super().__init__()
        self.base_layer = _get_base_layer(peft_lora_layer)
        self.expert_names = list(expert_names)
        self.num_experts = len(expert_names)
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.top_k = int(top_k or 0)
        self.router_type = (router_type or "linear").lower()
        self.router_hidden_size = int(router_hidden_size or 0)
        self.router_granularity = (router_granularity or "token").lower()
        if self.router_granularity not in {"token", "sample"}:
            raise ValueError(
                f"Unsupported MoE router granularity: {self.router_granularity}. "
                "Use 'token' or 'sample'."
            )
        self.scaling = float(scaling)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        in_features = int(expert_a[0].shape[1])
        out_features = int(expert_b[0].shape[0])
        base_param = _first_parameter(self.base_layer)
        device = base_param.device if base_param is not None else expert_a[0].device
        lora_dtype = base_param.dtype if base_param is not None and base_param.is_floating_point() else expert_a[0].dtype

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        for a_weight, b_weight in zip(expert_a, expert_b):
            rank = int(a_weight.shape[0])
            a_layer = nn.Linear(in_features, rank, bias=False, device=device, dtype=lora_dtype)
            b_layer = nn.Linear(rank, out_features, bias=False, device=device, dtype=lora_dtype)
            a_layer.weight.data.copy_(a_weight.to(device=device, dtype=lora_dtype))
            b_layer.weight.data.copy_(b_weight.to(device=device, dtype=lora_dtype))
            a_layer.requires_grad_(False)
            b_layer.requires_grad_(False)
            self.lora_A.append(a_layer)
            self.lora_B.append(b_layer)

        self.router = _build_router(
            in_features=in_features,
            num_experts=self.num_experts,
            router_type=self.router_type,
            router_hidden_size=self.router_hidden_size,
            device=device,
        )

        self.base_layer.requires_grad_(False)
        self._gate_aux_loss: Optional[torch.Tensor] = None

    def _pool_sample_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() >= 3:
            return hidden_states[:, -1, :]
        return hidden_states

    def _get_router_inputs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.router_granularity == "sample":
            # Revision experiments supply context from a separate prompt-only pass.
            # It must not be pooled from teacher-forced answer tokens.
            context = getattr(self, "prompt_router_context", None)
            if context is not None:
                return context
            return self._pool_sample_hidden_states(hidden_states)
        return hidden_states

    def _expand_gates(self, gates: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.router_granularity == "sample" and hidden_states.dim() >= 3:
            return gates.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        return gates

    def _gates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_inputs = self._get_router_inputs(hidden_states)
        logits = self.router(router_inputs.float()) / self.router_temperature
        gates = F.softmax(logits, dim=-1)
        gates = _apply_top_k(gates, self.top_k)
        return self._expand_gates(gates, hidden_states)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        gates = self._gates(x)
        if self.training:
            # Load balance auxiliary loss (Switch Transformer style):
            # encourage uniform expert usage across tokens.
            mean_gate = gates.mean(dim=list(range(gates.dim() - 1)))  # [num_experts]
            self._gate_aux_loss = self.num_experts * (mean_gate * mean_gate).sum()
        dropped = self.dropout(x)
        moe_update = None
        for idx, (lora_a, lora_b) in enumerate(zip(self.lora_A, self.lora_B)):
            expert_input = dropped.to(dtype=lora_a.weight.dtype)
            expert_output = lora_b(lora_a(expert_input)) * self.scaling
            expert_output = expert_output * gates[..., idx:idx + 1].to(dtype=expert_output.dtype)
            moe_update = expert_output if moe_update is None else moe_update + expert_output
        return result + moe_update.to(dtype=result.dtype)


def _iter_peft_lora_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if hasattr(module, "base_layer") or hasattr(module, "get_base_layer"):
                yield name, module


def install_moe_lora_adapters(
    model: nn.Module,
    cache: Dict[str, Dict[str, torch.Tensor]],
    router_temperature: float = 1.0,
    top_k: int = 0,
    router_type: str = "linear",
    router_hidden_size: int = 128,
    router_granularity: str = "token",
) -> int:
    expert_names = list(cache.keys())
    lora_layers = list(_iter_peft_lora_layers(model))
    if not lora_layers:
        raise ValueError("No PEFT LoRA layers found. Cannot install MoE-LoRA routers.")

    replaced = 0
    for module_name, peft_lora_layer in lora_layers:
        expert_a = []
        expert_b = []
        for expert_name in expert_names:
            state = cache[expert_name]
            expert_a.append(_find_lora_weight(state, module_name, "lora_A"))
            expert_b.append(_find_lora_weight(state, module_name, "lora_B"))
        moe_layer = AdaptiveMoeLoraLinear(
            peft_lora_layer=peft_lora_layer,
            expert_names=expert_names,
            expert_a=expert_a,
            expert_b=expert_b,
            scaling=_get_lora_scaling(peft_lora_layer, default=1.0),
            dropout_p=_get_lora_dropout_p(peft_lora_layer),
            router_temperature=router_temperature,
            top_k=top_k,
            router_type=router_type,
            router_hidden_size=router_hidden_size,
            router_granularity=router_granularity,
        )
        _set_module(model, module_name, moe_layer)
        replaced += 1
    return replaced


def iter_moe_router_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, AdaptiveMoeLoraLinear):
            yield module


def count_moe_router_modules(model: nn.Module) -> int:
    return sum(1 for _ in iter_moe_router_modules(model))


def iter_moe_router_parameters(model: nn.Module):
    for module in iter_moe_router_modules(model):
        yield from module.router.parameters()


def collect_moe_aux_loss(model: nn.Module) -> Optional[torch.Tensor]:
    """Collect and average the load-balance auxiliary losses from all MoE layers."""
    total: Optional[torch.Tensor] = None
    count = 0
    for module in iter_moe_router_modules(model):
        if module._gate_aux_loss is not None:
            total = module._gate_aux_loss if total is None else total + module._gate_aux_loss
            count += 1
    if total is None or count == 0:
        return None
    return total / count


def train_moe_routers(
    model: nn.Module,
    example_dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_steps: int,
    logger,
    load_balance_coeff: float = 0.0,
) -> None:
    for param in model.parameters():
        param.requires_grad = False
    router_params = list(iter_moe_router_parameters(model))
    if not router_params:
        raise ValueError("No MoE router parameters found.")
    for param in router_params:
        param.requires_grad = True

    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(router_params, lr=learning_rate)
    dataloader = DataLoader(example_dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = 0
    for epoch in range(int(epochs)):
        for batch in dataloader:
            if max_steps > 0 and step >= max_steps:
                break
            if hasattr(batch, "to"):
                batch = batch.to(device)
            else:
                batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if load_balance_coeff > 0.0:
                aux_loss = collect_moe_aux_loss(model)
                if aux_loss is not None:
                    loss = loss + load_balance_coeff * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
            optimizer.step()
            if logger is not None:
                logger.write("MoE router step %d, epoch %d, loss %.4f" % (step, epoch, float(loss.detach().cpu())))
            step += 1
        if max_steps > 0 and step >= max_steps:
            break
    model.eval()


def save_moe_router_state(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    router_state = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveMoeLoraLinear):
            router_state[name] = {
                "router_state": module.router.state_dict(),
                "expert_names": list(module.expert_names),
                "router_granularity": module.router_granularity,
                "router_type": module.router_type,
                "top_k": module.top_k,
                "temperature": module.router_temperature,
            }
    torch.save(router_state, path)


def load_moe_router_state(model: nn.Module, path: str, strict: bool = True) -> int:
    """Load router parameters into an already installed MoE-LoRA model."""
    try:
        router_state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        router_state = torch.load(path, map_location="cpu")

    moe_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, AdaptiveMoeLoraLinear)
    }
    checkpoint_names = set(router_state)
    model_names = set(moe_modules)
    missing_in_model = sorted(checkpoint_names - model_names)
    missing_in_checkpoint = sorted(model_names - checkpoint_names)
    if strict and (missing_in_model or missing_in_checkpoint):
        raise RuntimeError(
            "MoE router checkpoint does not match the installed adapters: "
            f"missing_in_model={missing_in_model}, "
            f"missing_in_checkpoint={missing_in_checkpoint}"
        )

    loaded = 0
    for name in sorted(checkpoint_names & model_names):
        payload = router_state[name]
        module = moe_modules[name]
        expected_metadata = {
            "expert_names": list(module.expert_names),
            "router_granularity": module.router_granularity,
            "router_type": module.router_type,
            "top_k": module.top_k,
            "temperature": module.router_temperature,
        }
        mismatches = {
            key: {"checkpoint": payload.get(key), "model": expected}
            for key, expected in expected_metadata.items()
            if key in payload and payload.get(key) != expected
        }
        if strict and mismatches:
            raise RuntimeError(
                f"MoE router metadata mismatch at layer {name}: {mismatches}"
            )
        module.router.load_state_dict(payload["router_state"], strict=True)
        loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No matching MoE router layers found in checkpoint: {path}")
    return loaded
