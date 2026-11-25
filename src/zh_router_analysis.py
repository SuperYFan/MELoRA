"""
Router activation analysis for MoE-LoRA.

For each test sample, runs a single forward pass (no generation), hooks into
every AdaptiveMoeLoraLinear._gates() call, and records the mean-over-tokens
gate weight vector per layer.  Produces:
  1. Bar chart: mean expert activation by class label (averaged over all layers)
  2. Heatmap: per-layer expert activation, separate panels per class
  3. Scatter (PCA): per-sample routing vectors coloured by label
  4. Raw JSON with all collected gate weights

Usage (on server):
  python src/zh_router_analysis.py \
      --dataset zh-toxicn-mm \
      --base_model models/Qwen2.5-7B \
      --lora_modules zh_hate_semantics/checkpoint-1200,zh_met_m3_toxicn_reasoning/checkpoint-1812,zh_visual_expression/checkpoint-1332 \
      --lora_dir ./lora_modules \
      --data_root data/zh_modular/benchmarks \
      --output_dir tmp_case_study/router_analysis \
      --seed 1111 --num_shots 16 --max_samples 500
"""

import argparse
import json
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── Matplotlib backend (headless server) ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Constants ─────────────────────────────────────────────────────────────────
POS_WORD = "\u662f"   # 是  (label 1 = positive / hate / sarcastic)
NEG_WORD = "\u5426"   # 否  (label 0 = negative / benign)

EXPERT_LABELS = {
    "zh_hate_semantics":      "Hate\nSemantics",
    "zh_met_m3_toxicn_reasoning":  "MET-Meme\nReasoning",
    "zh_visual_expression":   "Visual\nExpression",
}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["zh-toxicn-mm", "zh-sarcnet", "zh-misogyny-meme"])
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_modules", required=True,
                   help="Comma-separated module paths, same order as expert_names")
    p.add_argument("--lora_dir", default="./lora_modules")
    p.add_argument("--data_root", default="")
    p.add_argument("--output_dir", default="tmp_case_study/router_analysis")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--num_shots", type=int, default=16,
                   help="Calibration shots for router training")
    p.add_argument("--moe_router_epochs", type=int, default=20)
    p.add_argument("--moe_router_lr", type=float, default=1e-3)
    p.add_argument("--moe_load_balance_coeff", type=float, default=0.01)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--moe_router_hidden_size", type=int, default=128)
    p.add_argument("--router_checkpoint", default="",
                   help="Saved router .pt from the prediction run. When set, skip router retraining.")
    p.add_argument("--cutoff_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit test samples for speed (0 = all)")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Data loading ───────────────────────────────────────────────────────────────
BENCH_DIR_MAP = {
    "zh-toxicn-mm":      "toxicn_mm",
    "zh-sarcnet":        "sarcnet",
    "zh-misogyny-meme":  "misogyny_meme",
}

def load_test_data(data_root: str, dataset: str, max_samples: int = 0):
    bench_dir = BENCH_DIR_MAP[dataset]
    candidates = [
        Path(data_root) / "zh_modular" / "benchmarks" / bench_dir / "test.json",
        Path(data_root) / bench_dir / "test.json",
        Path(data_root) / dataset / "test.json",
        Path(data_root) / "benchmarks" / bench_dir / "test.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        tried = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Cannot find test.json. Tried:\n  {tried}")
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    data = []
    for row in rows:
        if row.get("label") not in (0, 1):
            continue
        metadata = row.get("metadata") or {}
        data.append({
            "case_id": str(row.get("id", "")),
            "label": int(row["label"]),
            "input": row.get("input", ""),
            "original_input": row.get("original_input", ""),
            "meme_description": metadata.get("image_description", ""),
        })
    if max_samples > 0:
        rng = random.Random(42)
        # Balanced subsample
        pos = [r for r in data if r["label"] == 1]
        neg = [r for r in data if r["label"] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        n = max_samples // 2
        data = pos[:n] + neg[:n]
        rng.shuffle(data)
    print(f"Loaded {len(data)} test samples  (label=1: {sum(r['label']==1 for r in data)}, "
          f"label=0: {sum(r['label']==0 for r in data)})")
    return data


# ── Prompt builder (reuse same format as zh_explanation_case_study) ────────────
def make_prompt(data_point: dict) -> str:
    input_text = data_point.get("input", "").strip()
    if not input_text:
        input_text = data_point.get("original_input", "").strip()
    return (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:请判断该内容是否有害，只输出是或否。\n\n"
        f"### Input:{input_text}\n\n"
        "### Response:"
    )


# ── Model loading (mirrors load_moe_model in zh_explanation_case_study.py) ─────
def load_moe_model(args, lora_module_list):
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from lora_learning import load_base_model_and_lora_modules
    from zh_fewshot_dataset import ZH_FewShot_Data
    from moe_lora import (
        install_moe_lora_adapters,
        load_moe_router_state,
        train_moe_routers,
    )

    mock_args = types.SimpleNamespace(
        base_model=args.base_model,
        load_8bit=False,
        batch_size=args.batch_size,
        cutoff_len=args.cutoff_len,
        num_shots=args.num_shots,
        DATASET=args.dataset,
        PATH=args.data_root,
        SEED=args.seed,
        train_on_inputs=False,
        POS_WORD=POS_WORD,
        NEG_WORD=NEG_WORD,
    )

    print("Loading base model and LoRA modules ...")
    model, tokenizer, cache = load_base_model_and_lora_modules(
        mock_args, lora_module_list, args.base_model
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = None
    if not args.router_checkpoint:
        print("Loading calibration dataset ...")
        dataset = ZH_FewShot_Data(mock_args, tokenizer)

    print("Installing MoE-LoRA routers ...")
    install_moe_lora_adapters(
        model, cache,
        router_temperature=1.0,
        top_k=args.moe_top_k,
        router_type="mlp",
        router_hidden_size=args.moe_router_hidden_size,
        router_granularity="token",
    )

    class _SilentLogger:
        def write(self, *a, **kw): pass

    if args.router_checkpoint:
        checkpoint = Path(args.router_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Router checkpoint not found: {checkpoint}")
        loaded = load_moe_router_state(model, str(checkpoint), strict=True)
        print(f"Loaded {loaded} router layers from: {checkpoint}")
    else:
        print("Training MoE routers on calibration set ...")
        if dataset is None:
            raise RuntimeError("Calibration dataset was not initialized.")
        train_moe_routers(
            model=model,
            example_dataset=dataset,
            batch_size=args.batch_size,
            epochs=args.moe_router_epochs,
            learning_rate=args.moe_router_lr,
            max_steps=0,
            load_balance_coeff=args.moe_load_balance_coeff,
            logger=_SilentLogger(),
        )
    model.eval()
    return model, tokenizer


# ── Gate collection ────────────────────────────────────────────────────────────
def collect_gate_weights(model, tokenizer, data, device, cutoff_len=512):
    """
    For each sample, run a forward pass and collect mean-over-tokens gate
    weights from every AdaptiveMoeLoraLinear layer.

    Returns:
        records: list of dicts  {label, case_id, gate_per_layer: [[num_experts] x num_layers]}
        expert_names: list of str
        layer_names: list of str
    """
    from moe_lora import AdaptiveMoeLoraLinear

    # Enumerate MoE layers once for stable ordering
    moe_layers = [(name, mod) for name, mod in model.named_modules()
                  if isinstance(mod, AdaptiveMoeLoraLinear)]
    if not moe_layers:
        raise RuntimeError("No AdaptiveMoeLoraLinear layers found in model.")

    expert_names = moe_layers[0][1].expert_names
    layer_names  = [name for name, _ in moe_layers]
    num_layers   = len(moe_layers)
    num_experts  = len(expert_names)
    print(f"Found {num_layers} MoE layers  |  experts: {expert_names}")

    # Per-sample accumulator; populated by hooks
    _sample_gates = []   # list of [num_layers x num_experts] (accumulated per forward call)
    _layer_buf   = [None] * num_layers   # scratch buffer reset before each sample

    def make_hook(layer_idx):
        def hook(module, inp, output):
            # output of forward() is the projected hidden state, NOT the gates.
            # We recompute gates from inp[0] (the input hidden states).
            with torch.no_grad():
                x = inp[0]
                gates = module._gates(x)   # [batch=1, seq_len, num_experts]  (token granularity)
                # Mean over sequence positions, keep batch=1 → [num_experts]
                mean_gates = gates.mean(dim=list(range(gates.dim() - 1)))  # [num_experts]
                _layer_buf[layer_idx] = mean_gates.float().cpu().numpy()
        return hook

    hooks = [mod.register_forward_hook(make_hook(i)) for i, (_, mod) in enumerate(moe_layers)]

    records = []
    total = len(data)
    try:
        for idx, row in enumerate(data):
            prompt = make_prompt(row)
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cutoff_len,
                padding=False,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Reset buffer
            for i in range(num_layers):
                _layer_buf[i] = None

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            gate_per_layer = []
            for i in range(num_layers):
                g = _layer_buf[i]
                if g is None:
                    g = np.zeros(num_experts, dtype=np.float32)
                gate_per_layer.append(g.tolist())

            records.append({
                "case_id":       row["case_id"],
                "label":         row["label"],
                "gate_per_layer": gate_per_layer,   # [num_layers, num_experts]
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"  processed {idx+1}/{total}")
    finally:
        for h in hooks:
            h.remove()

    return records, expert_names, layer_names


# ── Plotting ───────────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "Non-hate / Negative", 1: "Hate / Positive"}
LABEL_COLORS = {0: "#4C72B0", 1: "#DD8452"}   # blue / orange


def _short_expert_name(full_name: str) -> str:
    for k, v in EXPERT_LABELS.items():
        if k in full_name:
            return v
    # Fallback: use last path component
    return full_name.split("/")[-1]


def plot_mean_activation_by_class(records, expert_names, dataset, out_path):
    """
    Figure 1: Grouped bar chart — mean expert weight, split by label class.
    """
    short_names = [_short_expert_name(n) for n in expert_names]
    num_experts = len(expert_names)

    label_gates = defaultdict(list)   # label -> list of [num_experts] (averaged over layers)
    for rec in records:
        layer_mean = np.mean(rec["gate_per_layer"], axis=0)   # [num_experts]
        label_gates[rec["label"]].append(layer_mean)

    means = {}
    sems  = {}
    for lbl, vecs in label_gates.items():
        arr = np.array(vecs)      # [N, num_experts]
        means[lbl] = arr.mean(axis=0)
        sems[lbl]  = arr.std(axis=0) / np.sqrt(len(arr))

    fig, ax = plt.subplots(figsize=(max(6, num_experts * 1.8 + 2), 4.5))
    x = np.arange(num_experts)
    width = 0.35
    labels_sorted = sorted(means.keys())
    offsets = np.linspace(-width * (len(labels_sorted) - 1) / 2,
                           width * (len(labels_sorted) - 1) / 2,
                           len(labels_sorted))

    for lbl, offset in zip(labels_sorted, offsets):
        ax.bar(
            x + offset,
            means[lbl],
            width,
            yerr=sems[lbl],
            capsize=4,
            label=LABEL_NAMES.get(lbl, f"Label {lbl}"),
            color=LABEL_COLORS.get(lbl, "grey"),
            alpha=0.85,
        )

    ax.set_xlabel("Expert Module", fontsize=11)
    ax.set_ylabel("Mean Gate Weight", fontsize=11)
    ax.set_title(f"Expert Activation by Class — {dataset}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_layerwise_heatmap(records, expert_names, layer_names, dataset, out_path,
                           max_layers=32):
    """
    Figure 2: Heatmap — expert activation over layer depth, one subplot per class.
    Rows = experts, Columns = layer groups (binned if > max_layers).
    """
    short_names = [_short_expert_name(n) for n in expert_names]
    num_layers  = len(layer_names)
    num_experts = len(expert_names)

    # Bin layers for readability
    n_bins = min(num_layers, max_layers)
    bin_size = max(1, num_layers // n_bins)
    n_bins = (num_layers + bin_size - 1) // bin_size

    label_gates = defaultdict(list)
    for rec in records:
        label_gates[rec["label"]].append(np.array(rec["gate_per_layer"]))  # [num_layers, E]

    labels_sorted = sorted(label_gates.keys())
    fig, axes = plt.subplots(1, len(labels_sorted),
                             figsize=(n_bins * 0.45 + 3, num_experts * 0.9 + 2),
                             sharey=True)
    if len(labels_sorted) == 1:
        axes = [axes]

    vmin, vmax = 0.0, 1.0

    for ax, lbl in zip(axes, labels_sorted):
        arrs  = np.stack(label_gates[lbl], axis=0)  # [N, num_layers, E]
        mean  = arrs.mean(axis=0)                    # [num_layers, E]

        # Bin layers
        binned = np.zeros((n_bins, num_experts))
        for b in range(n_bins):
            start = b * bin_size
            end   = min(start + bin_size, num_layers)
            binned[b] = mean[start:end].mean(axis=0)

        # [experts, bins] for heatmap (experts on y-axis)
        sns.heatmap(
            binned.T,
            ax=ax,
            vmin=vmin, vmax=vmax,
            cmap="YlOrRd",
            linewidths=0.4,
            linecolor="white",
            cbar=(ax == axes[-1]),
            yticklabels=short_names,
            xticklabels=[str(b * bin_size) for b in range(n_bins)],
        )
        ax.set_title(LABEL_NAMES.get(lbl, f"Label {lbl}"), fontsize=10)
        ax.set_xlabel("Layer group (start)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"Expert Activation per Layer — {dataset}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pca_scatter(records, expert_names, dataset, out_path):
    """
    Figure 3: PCA scatter — each point = one sample, coloured by label.
    Feature vector = flattened gate weights across all layers.
    """
    from sklearn.decomposition import PCA

    X = np.array([np.array(rec["gate_per_layer"]).flatten() for rec in records])
    y = np.array([rec["label"] for rec in records])

    n_components = min(2, X.shape[1], X.shape[0] - 1)
    if n_components < 2:
        print("  Skipping PCA scatter (not enough dimensions).")
        return

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lbl in sorted(set(y)):
        mask = y == lbl
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=LABEL_COLORS.get(lbl, "grey"),
            label=LABEL_NAMES.get(lbl, f"Label {lbl}"),
            alpha=0.45, s=18, linewidths=0,
        )
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=10)
    ax.set_title(f"Routing Vector PCA — {dataset}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary(records, expert_names):
    """Print per-class mean gate weights to stdout."""
    short_names = [_short_expert_name(n) for n in expert_names]
    label_gates = defaultdict(list)
    for rec in records:
        layer_mean = np.mean(rec["gate_per_layer"], axis=0)
        label_gates[rec["label"]].append(layer_mean)

    print("\n=== Mean Expert Activation by Class ===")
    header = f"{'Class':30s}" + "".join(f"  {n:>15}" for n in short_names)
    print(header)
    print("-" * len(header))
    for lbl in sorted(label_gates.keys()):
        arr  = np.array(label_gates[lbl])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        row  = f"{LABEL_NAMES.get(lbl, f'Label {lbl}'):<30s}"
        for m, s in zip(mean, std):
            row += f"  {m:.4f}±{s:.4f}"
        print(row)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve LoRA module paths
    lora_module_list = []
    for m in args.lora_modules.split(","):
        m = m.strip()
        if not m:
            continue
        p = Path(m)
        if p.is_absolute() or p.exists():
            lora_module_list.append(str(p))
        else:
            lora_module_list.append(str(Path(args.lora_dir) / m))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model, tokenizer = load_moe_model(args, lora_module_list)
    model = model.to(device)
    model.eval()

    # Load data
    data = load_test_data(args.data_root, args.dataset, args.max_samples)

    # Collect gate weights
    print("\nCollecting gate weights ...")
    records, expert_names, layer_names = collect_gate_weights(
        model, tokenizer, data, device, cutoff_len=args.cutoff_len
    )

    # Save raw data
    raw_path = out_dir / "router_gate_weights.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump({
            "dataset":      args.dataset,
            "seed":         args.seed,
            "num_shots_per_class": args.num_shots,
            "moe_top_k": args.moe_top_k,
            "moe_load_balance_coeff": args.moe_load_balance_coeff,
            "router_checkpoint": args.router_checkpoint,
            "expert_names": expert_names,
            "layer_names":  layer_names,
            "records":      records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nRaw gate weights saved: {raw_path}")

    # Print summary
    print_summary(records, expert_names)

    # Plot 1: Mean activation by class
    plot_mean_activation_by_class(
        records, expert_names, args.dataset,
        out_dir / "fig1_mean_activation_by_class.png"
    )

    # Plot 2: Layer-wise heatmap
    plot_layerwise_heatmap(
        records, expert_names, layer_names, args.dataset,
        out_dir / "fig2_layerwise_heatmap.png"
    )

    # Plot 3: PCA scatter
    try:
        plot_pca_scatter(
            records, expert_names, args.dataset,
            out_dir / "fig3_pca_scatter.png"
        )
    except ImportError:
        print("  scikit-learn not found, skipping PCA scatter.")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
