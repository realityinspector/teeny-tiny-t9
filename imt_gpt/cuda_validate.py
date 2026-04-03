#!/usr/bin/env python3
"""
cuda_validate.py — Run Prism validation on CUDA at real scale.

Three experiments:
1. GPT-2 small (124M) at real batch/seq, 5K steps, 3 seeds
2. GPT-2 medium (355M) self-extraction, 5K steps
3. Longer training: GPT-2 small at 10K steps

Usage:
    python cuda_validate.py [--small|--medium|--long|--all]
"""
import json
import sys
import os
import time
import gc
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imt_gpt.config import TrainConfig, install_signal_handlers
from imt_gpt.train import train, _clear_memory
from imt_gpt.baselines import make_init_fn
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn

sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = "imt_gpt/results/cuda"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single(name, config, init_fn):
    """Run a single training job and save results."""
    result_path = f"{RESULTS_DIR}/{name}.json"
    if os.path.exists(result_path):
        print(f"  [{name}] SKIP — already done")
        return json.load(open(result_path))

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  device={config.device} batch={config.batch_size}x{config.grad_accum_steps} "
          f"seq={config.max_length} steps={config.max_steps}")
    print(f"{'='*60}")

    result = train(config, init_fn=init_fn, init_name=name, verbose=True)

    data = {
        "name": name,
        "final_ppl": result["final_ppl"],
        "elapsed": result["elapsed"],
        "checkpoints": {str(k): v for k, v in result["checkpoints"].items()},
        "grad_norm_samples": result["grad_norms"][::max(1, len(result["grad_norms"]) // 50)],
        "seed": config.seed,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum_steps,
        "max_length": config.max_length,
        "max_steps": config.max_steps,
        "lr": config.lr,
        "device": config.device,
    }

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {result_path}: PPL={result['final_ppl']:.2f}")
    return data


def make_prism_init(model_name="gpt2", align=True, device="cuda"):
    """Create the Prism init function (Spectral Imprint + EigenTransfer)."""
    # Load extracted spectra (group-averaged DCT)
    spectra_path = "imt_gpt/results/extracted_spectra.json"
    if not os.path.exists(spectra_path):
        # Extract fresh
        from imt_gpt.spectral_init import classify_param, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED
        from transformers import GPT2LMHeadModel
        print(f"  Extracting spectra from {model_name}...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        group_svs = {ATTN_PROJ: [], FFN_UP: [], FFN_DOWN: [], EMBED: []}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() < 2:
                    continue
                group = classify_param(name)
                if group and group in group_svs:
                    W = param.data.float()
                    if W.dim() > 2:
                        W = W.reshape(W.shape[0], -1)
                    s = torch.linalg.svdvals(W)
                    s_norm = s / s.max() if s.max() > 0 else s
                    group_svs[group].append(s_norm.cpu().numpy())
        del model
        gc.collect()

        spectra_coeffs = {}
        for group, sv_list in group_svs.items():
            if not sv_list:
                continue
            max_len = max(len(s) for s in sv_list)
            interp = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(s)), s) for s in sv_list]
            avg = np.mean(interp, axis=0)
            clipped = np.clip(avg, 0.01, None)
            target = np.log(np.exp(clipped) - 1.0 + 1e-10)
            n = len(avg)
            t = np.linspace(0, np.pi, n, endpoint=False)
            basis = np.zeros((n, 8))
            for i in range(8):
                basis[:, i] = np.cos((i + 0.5) * t)
            coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
            spectra_coeffs[group] = coeffs.tolist()

        extracted = {"spectra_coeffs": spectra_coeffs}
        with open(spectra_path, "w") as f:
            json.dump(extracted, f, indent=2)
    else:
        with open(spectra_path) as f:
            extracted = json.load(f)

    if align:
        # EigenTransfer: extract pretrained directions
        print(f"  Extracting directions from {model_name} for EigenTransfer...")
        dirs = extract_per_layer(model_name, include_directions=True, device="cpu")
        return make_hybrid_init_fn(
            extracted["spectra_coeffs"], dirs,
            lam=1.0, align_mode="UV", align_strength=0.5
        )
    else:
        return make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])


def run_small_validation():
    """Experiment 1: GPT-2 small at real batch/seq, 5K steps, 3 seeds."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: GPT-2 small at real scale (3 seeds)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps"

    for seed in [42, 137, 512]:
        # Orthogonal baseline
        config_ortho = TrainConfig(
            max_steps=5000,
            eval_steps=[500, 1000, 2000, 3000, 4000, 5000],
            warmup_steps=500,
            log_every=100,
            lr=6.25e-5,
            seed=seed,
            device=device,
            batch_size=16,
            grad_accum_steps=4,  # effective batch 64
            max_length=1024,
            memory_pressure_threshold=5,
        )
        run_single(f"small_ortho_s{seed}", config_ortho, make_init_fn("orthogonal"))
        _clear_memory(device)

        # Prism: Spectral Imprint + EigenTransfer + 2x LR
        config_prism = TrainConfig(
            max_steps=5000,
            eval_steps=[500, 1000, 2000, 3000, 4000, 5000],
            warmup_steps=500,
            log_every=100,
            lr=1.25e-4,  # 2x LR
            seed=seed,
            device=device,
            batch_size=16,
            grad_accum_steps=4,  # effective batch 64
            max_length=1024,
            memory_pressure_threshold=5,
        )
        prism_init = make_prism_init("gpt2", align=True, device=device)
        run_single(f"small_prism_s{seed}", config_prism, prism_init)
        _clear_memory(device)


def run_medium_validation():
    """Experiment 2: GPT-2 medium (355M) self-extraction, 5K steps."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: GPT-2 medium (355M)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps"

    # Need to extract spectra from medium and apply to medium
    # Override the model name in config
    from transformers import GPT2Config

    medium_config_args = dict(
        max_steps=5000,
        eval_steps=[500, 1000, 2000, 3000, 4000, 5000],
        warmup_steps=500,
        log_every=100,
        seed=42,
        device=device,
        batch_size=8,
        grad_accum_steps=8,  # effective batch 64
        max_length=1024,
        model_name="gpt2-medium",
        memory_pressure_threshold=5,
    )

    # Orthogonal baseline
    config_ortho = TrainConfig(**medium_config_args, lr=6.25e-5)
    run_single("medium_ortho_s42", config_ortho, make_init_fn("orthogonal"))
    _clear_memory(device)

    # Prism (self-extracted from medium)
    # First extract spectra from gpt2-medium
    medium_spectra_path = f"{RESULTS_DIR}/medium_extracted_spectra.json"
    if not os.path.exists(medium_spectra_path):
        from imt_gpt.spectral_init import classify_param, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED
        from transformers import GPT2LMHeadModel
        print("  Extracting spectra from gpt2-medium...")
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        group_svs = {ATTN_PROJ: [], FFN_UP: [], FFN_DOWN: [], EMBED: []}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() < 2:
                    continue
                group = classify_param(name)
                if group and group in group_svs:
                    W = param.data.float()
                    if W.dim() > 2:
                        W = W.reshape(W.shape[0], -1)
                    s = torch.linalg.svdvals(W)
                    s_norm = s / s.max() if s.max() > 0 else s
                    group_svs[group].append(s_norm.cpu().numpy())
        del model
        gc.collect()

        spectra_coeffs = {}
        for group, sv_list in group_svs.items():
            if not sv_list:
                continue
            max_len = max(len(s) for s in sv_list)
            interp = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(s)), s) for s in sv_list]
            avg = np.mean(interp, axis=0)
            clipped = np.clip(avg, 0.01, None)
            target = np.log(np.exp(clipped) - 1.0 + 1e-10)
            n = len(avg)
            t = np.linspace(0, np.pi, n, endpoint=False)
            basis = np.zeros((n, 8))
            for i in range(8):
                basis[:, i] = np.cos((i + 0.5) * t)
            coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
            spectra_coeffs[group] = coeffs.tolist()

        with open(medium_spectra_path, "w") as f:
            json.dump({"spectra_coeffs": spectra_coeffs}, f, indent=2)
    else:
        with open(medium_spectra_path) as f:
            spectra_coeffs = json.load(f)["spectra_coeffs"]

    # Extract directions from medium
    print("  Extracting directions from gpt2-medium...")
    dirs = extract_per_layer("gpt2-medium", include_directions=True, device="cpu")

    def medium_prism_init(model):
        from imt_gpt.pretrained_extract import apply_per_layer_spectral
        # Use group-averaged spectra + UV alignment from medium
        from imt_gpt.pretrained_extract import make_hybrid_init_fn as _make
        init_fn = _make(spectra_coeffs, dirs, lam=1.0, align_mode="UV", align_strength=0.5)
        init_fn(model)

    config_prism = TrainConfig(**medium_config_args, lr=1.25e-4)
    run_single("medium_prism_s42", config_prism, medium_prism_init)
    _clear_memory(device)


def run_long_validation():
    """Experiment 3: GPT-2 small, 10K steps — does the advantage persist?"""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: GPT-2 small, 10K steps")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps"

    long_config_args = dict(
        max_steps=10000,
        eval_steps=[1000, 2000, 3000, 5000, 7500, 10000],
        warmup_steps=1000,
        log_every=200,
        seed=42,
        device=device,
        batch_size=16,
        grad_accum_steps=4,  # effective batch 64
        max_length=1024,
        memory_pressure_threshold=5,
    )

    # Orthogonal
    config_ortho = TrainConfig(**long_config_args, lr=6.25e-5)
    run_single("long_ortho_s42", config_ortho, make_init_fn("orthogonal"))
    _clear_memory(device)

    # Prism
    config_prism = TrainConfig(**long_config_args, lr=1.25e-4)
    prism_init = make_prism_init("gpt2", align=True, device=device)
    run_single("long_prism_s42", config_prism, prism_init)
    _clear_memory(device)


def print_summary():
    """Print all CUDA validation results."""
    print("\n" + "=" * 60)
    print("  CUDA VALIDATION SUMMARY")
    print("=" * 60)

    import glob
    results = {}
    for f in sorted(glob.glob(f"{RESULTS_DIR}/*.json")):
        if "spectra" in f:
            continue
        with open(f) as fp:
            d = json.load(fp)
        results[d["name"]] = d

    if not results:
        print("  No results yet.")
        return

    print(f"\n{'Name':>30s}  {'PPL@last':>9s}  {'Elapsed':>8s}  {'Device':>6s}  {'Batch':>6s}  {'Steps':>6s}")
    print("-" * 75)
    for name, d in sorted(results.items()):
        ppl = d["final_ppl"]
        elapsed = d["elapsed"]
        batch = d.get("batch_size", "?") * d.get("grad_accum", 1)
        steps = d.get("max_steps", "?")
        dev = d.get("device", "?")
        print(f"{name:>30s}  {ppl:>9.1f}  {elapsed:>7.0f}s  {dev:>6s}  {batch:>6}  {steps:>6}")

    # Compute ratios for matched pairs
    print("\n  Speedup ratios:")
    for prefix in ["small", "medium", "long"]:
        for seed in [42, 137, 512]:
            ortho_key = f"{prefix}_ortho_s{seed}"
            prism_key = f"{prefix}_prism_s{seed}"
            if ortho_key in results and prism_key in results:
                # Find the last common checkpoint
                o_cp = results[ortho_key]["checkpoints"]
                p_cp = results[prism_key]["checkpoints"]
                common = sorted(set(o_cp.keys()) & set(p_cp.keys()), key=int)
                if common:
                    last = common[-1]
                    ratio = o_cp[last] / p_cp[last]
                    print(f"    {prefix} s{seed} @ step {last}: "
                          f"ortho={o_cp[last]:.1f} prism={p_cp[last]:.1f} "
                          f"ratio={ratio:.2f}x")


if __name__ == "__main__":
    install_signal_handlers()

    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"

    if mode == "--small":
        run_small_validation()
    elif mode == "--medium":
        run_medium_validation()
    elif mode == "--long":
        run_long_validation()
    elif mode == "--summary":
        print_summary()
    elif mode == "--all":
        run_small_validation()
        run_medium_validation()
        run_long_validation()
    else:
        print(f"Usage: python cuda_validate.py [--small|--medium|--long|--all|--summary]")

    print_summary()
