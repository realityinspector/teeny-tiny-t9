#!/usr/bin/env python3
"""
dct_ablation.py — Test how many DCT coefficients are needed.

Re-extracts spectra from pretrained GPT-2 at n_dct={2, 4, 8, 16, 32},
then trains 1000 steps with each. Tests how compressible the spectral
fingerprint is.

Usage:
    python -m imt_gpt.dct_ablation
"""
import json
import sys
import os
import time
import gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imt_gpt.spectral_init import (
    ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED, classify_param, dct_expand
)


def extract_spectra_at_n_dct(n_dct):
    """Extract and compress pretrained GPT-2 spectra to n_dct coefficients.

    Uses the SAME cosine basis + inverse-softplus + lstsq method as
    the original extraction in run.py, just with variable n_dct.
    """
    from transformers import GPT2LMHeadModel

    print(f"  Extracting spectra with n_dct={n_dct}...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Collect SVs per group
    group_svs = {ATTN_PROJ: [], FFN_UP: [], FFN_DOWN: [], EMBED: []}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_param(name)
            if group is None or group not in group_svs:
                continue

            W = param.data.float()
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)

            s = torch.linalg.svdvals(W)
            s_norm = s / s.max() if s.max() > 0 else s
            group_svs[group].append(s_norm.cpu().numpy())

    # Average spectrum per group, then compress via lstsq on custom cosine basis
    spectra_coeffs = {}
    for group, sv_list in group_svs.items():
        if not sv_list:
            continue

        # Interpolate all to same length (max in group) and average
        max_len = max(len(s) for s in sv_list)
        interpolated = []
        for s in sv_list:
            interp = np.interp(
                np.linspace(0, 1, max_len),
                np.linspace(0, 1, len(s)),
                s
            )
            interpolated.append(interp)
        avg_spectrum = np.mean(interpolated, axis=0)

        # Inverse softplus to get target in the cosine basis domain
        # (same as run.py extraction)
        avg_clipped = np.clip(avg_spectrum, 0.01, None)
        target = np.log(np.exp(avg_clipped) - 1.0 + 1e-10)

        # Build cosine basis matrix (matches dct_expand)
        n = len(avg_spectrum)
        t = np.linspace(0, np.pi, n, endpoint=False)
        basis = np.zeros((n, n_dct))
        for i in range(n_dct):
            basis[:, i] = np.cos((i + 0.5) * t)

        # Least squares fit
        coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)

        # Verify reconstruction quality using dct_expand
        reconstructed = dct_expand(coeffs, n)
        mse = np.mean((avg_spectrum - reconstructed) ** 2)
        corr = np.corrcoef(avg_spectrum, reconstructed)[0, 1]
        print(f"    {group:12s}: {len(sv_list)} matrices, "
              f"n_sv={max_len}, MSE={mse:.6f}, corr={corr:.4f}")

        spectra_coeffs[group] = coeffs.tolist()

    del model
    gc.collect()
    return spectra_coeffs


def run_training(name, spectra_coeffs, seed=42, steps=1000, lr=6.25e-5):
    """Run a single training job and return final PPL."""
    from imt_gpt.config import TrainConfig, get_device, set_process_memory_limit
    from imt_gpt.train import train, _clear_memory
    from imt_gpt.baselines import make_init_fn

    device = get_device()
    set_process_memory_limit(max_gb=10.0)

    config = TrainConfig(
        max_steps=steps,
        eval_steps=sorted(set([250, 500, 750, steps])),
        warmup_steps=100,
        log_every=100,
        lr=lr,
        seed=seed,
        device=device,
    )

    init_fn = make_init_fn("imt_shaped", spectra_coeffs=spectra_coeffs)
    result = train(config, init_fn=init_fn, init_name=name, verbose=True)

    _clear_memory(device)
    return result


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)

    n_dct_values = [2, 4, 8, 16, 32]
    results = {}

    for n_dct in n_dct_values:
        result_path = f"imt_gpt/results/dct_ablation_n{n_dct}.json"
        if os.path.exists(result_path):
            print(f"\n  [n_dct={n_dct}] SKIP — already done")
            with open(result_path) as f:
                results[n_dct] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"  DCT ABLATION: n_dct={n_dct}")
        print(f"{'='*60}")

        # Extract spectra at this compression level
        spectra = extract_spectra_at_n_dct(n_dct)

        # Train
        name = f"dct_n{n_dct}_s42"
        result = run_training(name, spectra, seed=42, steps=1000)

        result_data = {
            "n_dct": n_dct,
            "final_ppl": result["final_ppl"],
            "elapsed": result["elapsed"],
            "checkpoints": {str(k): v for k, v in result["checkpoints"].items()},
            "spectra_coeffs": spectra,
        }

        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"  Saved {result_path}: PPL={result['final_ppl']:.2f}")
        results[n_dct] = result_data

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DCT ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'n_dct':>6} {'PPL@250':>10} {'PPL@500':>10} {'PPL@1000':>10}")
    print("-" * 40)
    for n_dct in n_dct_values:
        if n_dct in results:
            r = results[n_dct]
            cp = r["checkpoints"]
            p250 = cp.get("250", "?")
            p500 = cp.get("500", "?")
            p1000 = cp.get("1000", r["final_ppl"])
            print(f"{n_dct:>6} {p250:>10.1f} {p500:>10.1f} {p1000:>10.1f}")


if __name__ == "__main__":
    main()
