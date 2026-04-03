#!/usr/bin/env python3
"""
lr_sweep.py — Test spectral init vs orthogonal across learning rates.

Sweeps LR from 6.25e-5 (current) to 6e-4 (standard GPT-2 pretraining).
Tests whether spectral init is more LR-robust than orthogonal.

We already know:
- Orthogonal diverges at 3x LR (1.875e-4)
- Both work at 1x LR (6.25e-5)
- Spectral init: PPL 839, Orthogonal: PPL 1871

Key question: Does spectral init work at the standard pretraining LR
where orthogonal fails?

Usage:
    python -m imt_gpt.lr_sweep
"""
import json
import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, "imt_gpt", ".venv", "bin", "python")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run_single(name, init_method, lr, seed=42, steps=1000, warmup=100,
               extra_kwargs="{}"):
    """Run a single training job as a subprocess."""
    script = f'''
import json, sys, os, numpy as np
sys.path.insert(0, "{PROJECT_ROOT}")
os.chdir("{PROJECT_ROOT}")

from imt_gpt.config import TrainConfig, get_device, install_signal_handlers, set_process_memory_limit
from imt_gpt.train import train, _clear_memory
from imt_gpt.baselines import make_init_fn

install_signal_handlers()
set_process_memory_limit(max_gb=10.0)
device = get_device()

config = TrainConfig(
    max_steps={steps},
    eval_steps=sorted(set([250, 500, 750, {steps}])),
    warmup_steps={warmup},
    log_every=100,
    lr={lr},
    seed={seed},
    device=device,
)

extra = {extra_kwargs}
init_fn = make_init_fn("{init_method}", **extra)
result = train(config, init_fn=init_fn, init_name="{name}", verbose=True)

path = "imt_gpt/results/lr_sweep_{name}.json"
with open(path, "w") as f:
    json.dump({{
        "init_name": "{name}",
        "final_ppl": result["final_ppl"],
        "elapsed": result["elapsed"],
        "checkpoints": {{str(k): v for k, v in result["checkpoints"].items()}},
        "loss_samples": result["losses"][::max(1, len(result["losses"]) // 30)],
        "grad_norm_samples": result["grad_norms"][::max(1, len(result["grad_norms"]) // 30)],
        "stopped_early": result.get("stopped_early", False),
        "seed": {seed},
        "lr": {lr},
        "steps": {steps},
        "method": "{init_method}",
    }}, f, indent=2)
print(f"Saved {{path}}: PPL={{result['final_ppl']:.2f}}", flush=True)
'''
    print(f"\n{'='*60}", flush=True)
    print(f"  LR SWEEP: {name} (lr={lr})", flush=True)
    print(f"  method={init_method} steps={steps} seed={seed}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=False, text=True, timeout=7200,
    )
    elapsed = time.time() - t0
    print(f"  [{name}] exit={result.returncode} wall={elapsed:.0f}s", flush=True)
    return result.returncode


def main():
    os.chdir(PROJECT_ROOT)

    # Load extracted spectra for shaped init
    with open("imt_gpt/results/extracted_spectra.json") as f:
        spectra_data = json.load(f)
    spectra_kwargs = json.dumps({
        "spectra_coeffs": spectra_data["spectra_coeffs"]
    })

    # LR sweep: test at multiples of the current 6.25e-5
    # Already have: ortho at 1x (PPL 1871) and 3x (diverges)
    # Already have: extracted at 1x (PPL 839)
    lrs = {
        "1x": 6.25e-5,     # current (fine-tuning range)
        "2x": 1.25e-4,     # midpoint
        "3x": 1.875e-4,    # already tested for ortho (diverges)
        "4x": 2.5e-4,      # standard pretraining range start
        "6x": 3.75e-4,     # mid pretraining range
    }

    tests = []
    for lr_name, lr in lrs.items():
        tests.append((f"extracted_{lr_name}_s42", "imt_shaped", lr, spectra_kwargs))
        tests.append((f"ortho_{lr_name}_s42", "orthogonal", lr, "{}"))

    # Skip already-done tests
    completed = 0
    skipped = 0
    for name, method, lr, kwargs in tests:
        result_path = f"imt_gpt/results/lr_sweep_{name}.json"
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    existing = json.load(f)
                ppl = existing.get("final_ppl", "?")
                print(f"  [{name}] SKIP — already done (PPL={ppl})", flush=True)
                skipped += 1
                completed += 1
                continue
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            rc = run_single(name, method, lr, extra_kwargs=kwargs)
            if rc == 0:
                completed += 1
        except subprocess.TimeoutExpired:
            print(f"  [{name}] TIMEOUT", flush=True)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  LR SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'LR':>12} {'extracted':>12} {'orthogonal':>12} {'Ratio':>8}")
    print("-" * 48)

    for lr_name, lr in lrs.items():
        ext_path = f"imt_gpt/results/lr_sweep_extracted_{lr_name}_s42.json"
        ort_path = f"imt_gpt/results/lr_sweep_ortho_{lr_name}_s42.json"
        ext_ppl = "?"
        ort_ppl = "?"
        if os.path.exists(ext_path):
            with open(ext_path) as f:
                ext_ppl = json.load(f)["final_ppl"]
        if os.path.exists(ort_path):
            with open(ort_path) as f:
                ort_ppl = json.load(f)["final_ppl"]

        ratio = ""
        if isinstance(ext_ppl, float) and isinstance(ort_ppl, float) and ext_ppl > 0:
            ratio = f"{ort_ppl/ext_ppl:.1f}x"
        print(f"{lr:>12.2e} {str(ext_ppl):>12} {str(ort_ppl):>12} {ratio:>8}")


if __name__ == "__main__":
    main()
