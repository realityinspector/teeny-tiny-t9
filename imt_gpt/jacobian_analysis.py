#!/usr/bin/env python3
"""
jacobian_analysis.py — Measure conditioning at step 0 for each init method.

Key question: Does spectral init produce better-conditioned Jacobians,
explaining both faster convergence and LR robustness?

Measures (all at step 0, before any training):
1. Per-layer weight condition numbers (σ_max / σ_min)
2. Forward pass activation norms at each layer
3. Backward pass gradient norms at each layer
4. Effective rank of each weight matrix
5. Overall loss and gradient norm on first batch

Usage:
    python -m imt_gpt.jacobian_analysis
"""
import json
import sys
import os
import gc
import math
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imt_gpt.config import TrainConfig, get_device, set_process_memory_limit
from imt_gpt.train import create_model, load_data, _clear_memory
from imt_gpt.baselines import make_init_fn


def weight_conditioning(model):
    """Compute per-layer weight matrix conditioning.

    Returns dict of layer_name -> {
        condition_number: σ_max / σ_min,
        spectral_norm: σ_max,
        effective_rank: exp(entropy of normalized SVs),
        frob_norm: Frobenius norm,
    }
    """
    results = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            W = param.data
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)

            try:
                s = torch.linalg.svdvals(W.float().cpu())
                s_max = s[0].item()
                s_min = s[-1].item()
                cond = s_max / max(s_min, 1e-10)

                # Effective rank: exp(entropy of normalized singular values)
                s_norm = s / s.sum()
                s_norm = s_norm[s_norm > 1e-10]
                entropy = -(s_norm * torch.log(s_norm)).sum().item()
                eff_rank = math.exp(entropy)

                results[name] = {
                    "condition_number": cond,
                    "spectral_norm": s_max,
                    "min_sv": s_min,
                    "effective_rank": eff_rank,
                    "frob_norm": torch.norm(W, 'fro').item(),
                    "shape": list(param.shape),
                }
            except Exception as e:
                results[name] = {"error": str(e)}
    return results


def activation_norms(model, batch, device):
    """Forward pass: measure activation norms at each layer.

    Uses hooks to capture intermediate activations.
    """
    norms = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                norms[name] = {
                    "norm": out.float().norm().item(),
                    "mean": out.float().mean().item(),
                    "std": out.float().std().item(),
                    "max_abs": out.float().abs().max().item(),
                }
        return hook_fn

    # Register hooks on transformer blocks and key sublayers
    for name, module in model.named_modules():
        if any(k in name for k in [".attn", ".mlp", ".ln_", "transformer.h."]):
            if name.count(".") <= 3:  # don't go too deep
                hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"][:1].to(device)
        labels = batch["labels"][:1].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()
        del input_ids, labels, outputs

    for h in hooks:
        h.remove()

    return norms, loss


def gradient_norms(model, batch, device):
    """Backward pass: measure per-layer gradient norms on first batch."""
    model.train()
    model.zero_grad()

    input_ids = batch["input_ids"][:1].to(device)
    labels = batch["labels"][:1].to(device)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    gnorms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorms[name] = {
                "grad_norm": param.grad.float().norm().item(),
                "grad_max": param.grad.float().abs().max().item(),
            }

    total_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

    del input_ids, labels, outputs, loss
    model.zero_grad()

    return gnorms, float(total_gnorm)


def analyze_init(method_name, init_fn, config, tokenizer_data, device):
    """Full conditioning analysis for one init method."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {method_name}")
    print(f"{'='*60}")

    model = create_model(config, init_fn=init_fn)
    model = model.to(device).float()

    # 1. Weight conditioning
    print("  Computing weight conditioning...")
    wcond = weight_conditioning(model)

    # Summarize by layer group
    group_conds = defaultdict(list)
    for name, info in wcond.items():
        if "error" in info:
            continue
        # Group by type
        if "c_attn" in name or "c_proj" in name and "mlp" not in name:
            group_conds["attention"].append(info["condition_number"])
        elif "c_fc" in name:
            group_conds["ffn_up"].append(info["condition_number"])
        elif "mlp" in name and "c_proj" in name:
            group_conds["ffn_down"].append(info["condition_number"])
        elif "wte" in name or "wpe" in name:
            group_conds["embedding"].append(info["condition_number"])
        else:
            group_conds["other"].append(info["condition_number"])

    for group, conds in sorted(group_conds.items()):
        print(f"    {group:12s}: median cond={np.median(conds):8.1f}  "
              f"max={np.max(conds):10.1f}  n={len(conds)}")

    # 2. Get a batch for forward/backward
    from torch.utils.data import DataLoader
    train_dl = DataLoader(tokenizer_data["train"], batch_size=2, shuffle=False)
    batch = next(iter(train_dl))

    # 3. Activation norms
    print("  Computing activation norms...")
    act_norms, init_loss = activation_norms(model, batch, device)
    print(f"    Init loss: {init_loss:.4f}  Init PPL: {math.exp(min(init_loss, 20)):.1f}")

    # Layer-by-layer activation summary
    block_norms = {}
    for name, info in sorted(act_norms.items()):
        if name.startswith("transformer.h.") and name.count(".") == 2:
            layer_idx = name.split(".")[2]
            block_norms[f"layer_{layer_idx}"] = info["std"]

    if block_norms:
        stds = list(block_norms.values())
        print(f"    Activation std: first={stds[0]:.4f}  last={stds[-1]:.4f}  "
              f"ratio={stds[-1]/max(stds[0], 1e-10):.2f}")

    # 4. Gradient norms
    print("  Computing gradient norms...")
    gnorms, total_gnorm = gradient_norms(model, batch, device)
    print(f"    Total gradient norm: {total_gnorm:.4f}")

    # Per-layer gradient summary
    layer_gnorms = defaultdict(list)
    for name, info in gnorms.items():
        if "transformer.h." in name:
            parts = name.split(".")
            idx = parts[parts.index("h") + 1]
            layer_gnorms[int(idx)].append(info["grad_norm"])

    if layer_gnorms:
        per_layer = {k: sum(v) for k, v in sorted(layer_gnorms.items())}
        vals = list(per_layer.values())
        print(f"    Layer grad norms: first={vals[0]:.4f}  last={vals[-1]:.4f}  "
              f"ratio={vals[-1]/max(vals[0], 1e-10):.2f}")

    del model
    _clear_memory(device)

    return {
        "method": method_name,
        "weight_conditioning": {
            name: {k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in info.items()}
            for name, info in wcond.items()
        },
        "group_condition_summary": {
            group: {
                "median": float(np.median(conds)),
                "mean": float(np.mean(conds)),
                "max": float(np.max(conds)),
                "min": float(np.min(conds)),
                "n": len(conds),
            }
            for group, conds in group_conds.items()
        },
        "activation_norms": act_norms,
        "block_activation_stds": block_norms,
        "gradient_norms_per_param": {
            name: info for name, info in gnorms.items()
        },
        "init_loss": init_loss,
        "init_ppl": math.exp(min(init_loss, 20)),
        "total_gradient_norm": total_gnorm,
    }


def main():
    device = get_device()
    set_process_memory_limit(max_gb=10.0)

    config = TrainConfig(
        max_steps=1,
        eval_steps=[],
        seed=42,
        device=device,
    )

    # Seed everything
    torch.manual_seed(42)
    np.random.seed(42)

    # Load extracted spectra for shaped init
    spectra_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "extracted_spectra.json"
    )
    with open(spectra_path) as f:
        spectra_data = json.load(f)

    spectra_coeffs = spectra_data["spectra_coeffs"]

    # Load data once
    print("Loading data...")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_data(config, tokenizer)

    # Methods to test
    methods = {
        "standard": make_init_fn("standard"),
        "xavier": make_init_fn("xavier"),
        "orthogonal": make_init_fn("orthogonal"),
        "imt_flat": make_init_fn("imt_flat"),
        "imt_extracted": make_init_fn("imt_shaped", spectra_coeffs=spectra_coeffs),
    }

    results = {}
    for name, init_fn in methods.items():
        # Reset seed before each init for fair comparison
        torch.manual_seed(42)
        np.random.seed(42)
        result = analyze_init(name, init_fn, config, dataset, device)
        results[name] = result
        _clear_memory(device)

    # Save results
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "jacobian_analysis.json"
    )

    # Convert any non-serializable types
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  CONDITIONING COMPARISON (step 0, seed=42)")
    print(f"{'='*80}")
    print(f"\n{'Method':<16} {'Init PPL':>10} {'Total gnorm':>12} "
          f"{'Attn cond':>12} {'FFN cond':>12} {'Embed cond':>12}")
    print("-" * 80)

    for name in ["standard", "xavier", "orthogonal", "imt_flat", "imt_extracted"]:
        r = results[name]
        gs = r["group_condition_summary"]
        attn = gs.get("attention", {}).get("median", 0)
        ffn = gs.get("ffn_up", {}).get("median", 0)
        emb = gs.get("embedding", {}).get("median", 0)
        print(f"{name:<16} {r['init_ppl']:>10.1f} {r['total_gradient_norm']:>12.2f} "
              f"{attn:>12.1f} {ffn:>12.1f} {emb:>12.1f}")

    # Activation propagation
    print(f"\n{'Method':<16} {'Act std L0':>12} {'Act std L11':>12} {'Ratio':>8}")
    print("-" * 56)
    for name in ["standard", "xavier", "orthogonal", "imt_flat", "imt_extracted"]:
        r = results[name]
        stds = r.get("block_activation_stds", {})
        if stds:
            vals = list(stds.values())
            first, last = vals[0], vals[-1]
            ratio = last / max(first, 1e-10)
            print(f"{name:<16} {first:>12.4f} {last:>12.4f} {ratio:>8.2f}")


if __name__ == "__main__":
    main()
