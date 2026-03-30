#!/usr/bin/env python3
"""
run.py — Main entry point for IMT-GPT experiments.

Usage:
    python -m imt_gpt.run test              # Quick smoke test (20 steps, <30s)
    python -m imt_gpt.run baselines         # Run all baseline inits (5K steps each)
    python -m imt_gpt.run baselines --quick # Quick baselines (200 steps each)
    python -m imt_gpt.run extract           # Extract spectra from pretrained GPT-2
    python -m imt_gpt.run search            # CMA-ES spectral search
    python -m imt_gpt.run eval [genome]     # Full eval with optional IMT genome
    python -m imt_gpt.run plot              # Generate plots from saved results
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cmd_test():
    """Quick smoke test: ~20 steps, tiny config. Should use <2GB, finish <30s."""
    from imt_gpt.config import TrainConfig, get_device
    from imt_gpt.train import train

    config = TrainConfig(
        batch_size=1,
        grad_accum_steps=1,
        max_length=128,       # tiny sequences
        max_steps=20,
        log_every=5,
        eval_every=20,
        eval_steps=[20],
        warmup_steps=5,
        device=get_device(),
    )
    result = train(config, init_name="smoke_test", verbose=True)
    print(f"\nSmoke test passed! Final PPL: {result['final_ppl']:.2f}")
    return result


def cmd_baselines(quick=False):
    """Run all baseline initializations."""
    from imt_gpt.config import TrainConfig, get_device
    from imt_gpt.eval import run_baselines

    if quick:
        config = TrainConfig(
            max_steps=200,
            eval_steps=[100, 200],
            warmup_steps=50,
            log_every=25,
            device=get_device(),
        )
    else:
        config = TrainConfig(device=get_device())
    return run_baselines(config=config)


def cmd_search():
    """Run CMA-ES spectral search."""
    from imt_gpt.config import TrainConfig, SearchConfig, get_device
    from imt_gpt.search import run_search

    train_config = TrainConfig(device=get_device())
    search_config = SearchConfig()
    return run_search(search_config=search_config, train_config=train_config)


def cmd_eval(genome_path=None, extracted=False, quick=False):
    """Run full evaluation with all methods."""
    from imt_gpt.config import TrainConfig, get_device
    from imt_gpt.eval import run_all

    if quick:
        config = TrainConfig(
            max_steps=200,
            eval_steps=[100, 200],
            warmup_steps=50,
            log_every=25,
            device=get_device(),
        )
    else:
        config = TrainConfig(device=get_device())

    extracted_path = "imt_gpt/results/extracted_spectra.json" if extracted else None
    return run_all(genome_path=genome_path, extracted_path=extracted_path,
                   config=config)


def cmd_extract():
    """Extract spectral shapes from pretrained GPT-2 (Open Question #5).

    Skip CMA-ES entirely: reverse-engineer the spectrum from a known-good model.
    Then test whether training from scratch with that spectrum beats baselines.
    """
    from imt_gpt.spectral_init import classify_param, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED
    import torch
    import numpy as np
    import json
    import os
    from transformers import GPT2LMHeadModel

    print("Loading pretrained GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Collect singular value distributions per group
    group_svs = {ATTN_PROJ: [], FFN_UP: [], FFN_DOWN: [], EMBED: []}

    print("Extracting singular values...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_param(name)
            if group is None or group not in group_svs:
                continue

            W = param.data
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)

            s = torch.linalg.svdvals(W)
            # Normalize to [0, 1]
            s_norm = s / s.max()
            group_svs[group].append(s_norm.cpu().numpy())
            print(f"  {name:50s} {str(list(param.shape)):20s} -> {group} "
                  f"({len(s)} SVs)")

    del model
    import gc; gc.collect()

    # Average spectra per group, fit DCT coefficients
    from scipy.fft import dct as scipy_dct
    spectra_coeffs = {}

    print("\nFitting DCT coefficients to average spectra...")
    for group, sv_list in group_svs.items():
        if not sv_list:
            print(f"  {group}: no matrices found!")
            continue

        # Interpolate all to common length and average
        max_len = max(len(s) for s in sv_list)
        interpolated = []
        for s in sv_list:
            x_old = np.linspace(0, 1, len(s))
            x_new = np.linspace(0, 1, max_len)
            interpolated.append(np.interp(x_new, x_old, s))

        avg_spectrum = np.mean(interpolated, axis=0)

        # Fit 8 DCT coefficients via least squares
        # Our dct_expand uses: spectrum[j] = sum_i coeffs[i] * cos((i+0.5)*t[j])
        # where t = linspace(0, pi, n)
        n = len(avg_spectrum)
        t = np.linspace(0, np.pi, n, endpoint=False)
        # We need to invert through softplus, so work in log space
        # softplus(x) = log(1 + exp(x)), inverse: log(exp(y) - 1)
        avg_clipped = np.clip(avg_spectrum, 0.01, None)
        # Normalize by max before inverse softplus
        target = np.log(np.exp(avg_clipped) - 1.0 + 1e-10)

        # Build basis matrix
        n_dct = 8
        basis = np.zeros((n, n_dct))
        for i in range(n_dct):
            basis[:, i] = np.cos((i + 0.5) * t)

        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
        spectra_coeffs[group] = coeffs.tolist()

        print(f"  {group}: {len(sv_list)} matrices, avg {max_len} SVs, "
              f"coeffs={[f'{c:.3f}' for c in coeffs]}")

    # Save
    os.makedirs("imt_gpt/results", exist_ok=True)
    result = {
        "method": "pretrained_extraction",
        "spectra_coeffs": spectra_coeffs,
        "lam": 1.0,
        "description": "Spectra extracted from pretrained GPT-2 weights",
    }
    path = "imt_gpt/results/extracted_spectra.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print("Use with: python -m imt_gpt.run eval --extracted")


def cmd_plot():
    """Generate plots from saved results."""
    from imt_gpt.plot import plot_all_from_dir
    plot_all_from_dir()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "test":
        cmd_test()
    elif cmd == "baselines":
        quick = "--quick" in sys.argv
        cmd_baselines(quick=quick)
    elif cmd == "search":
        cmd_search()
    elif cmd == "eval":
        extracted = "--extracted" in sys.argv
        quick = "--quick" in sys.argv
        genome = None
        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                genome = arg
        cmd_eval(genome, extracted=extracted, quick=quick)
    elif cmd == "plot":
        cmd_plot()
    elif cmd == "extract":
        cmd_extract()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
