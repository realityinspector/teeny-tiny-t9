#!/usr/bin/env python3
"""
run_diagnostics.py — Address reviewer critique of spectral init results.

Tests:
1. imt_scaled_flat: flat spectrum + pretrained Frobenius norms (shape vs scale)
2. Orthogonal with 3x LR: is the baseline LR-starved?
3. Orthogonal with 2000 steps: does it just need more time?
4. 3-seed runs for error bars on key methods

All results include gradient norms for spike analysis.
Runs each method in a subprocess for clean memory between runs.
"""
import json
import subprocess
import sys
import os
import time

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "imt_gpt", ".venv", "bin", "python")
# Fix: use the venv python from the project root
PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, "imt_gpt", ".venv", "bin", "python")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run_single(name, init_method, steps, lr, warmup, seed=None, extra_kwargs="{}"):
    """Run a single training job as a subprocess for clean memory."""
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
    eval_steps=sorted(set([max(1,{steps}//4), max(1,{steps}//2), max(1,3*{steps}//4), {steps}])),
    warmup_steps={warmup},
    log_every=max(1, {steps}//10),
    lr={lr},
    seed={seed},
    device=device,
)

extra = {extra_kwargs}
init_fn = make_init_fn("{init_method}", **extra)
result = train(config, init_fn=init_fn, init_name="{name}", verbose=True)

path = "imt_gpt/results/diag_{name}.json"
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
    }}, f, indent=2)
print(f"Saved {{path}}: PPL={{result['final_ppl']:.2f}}", flush=True)
'''
    print(f"\n{'='*60}", flush=True)
    print(f"  DIAGNOSTIC: {name}", flush=True)
    print(f"  method={init_method} steps={steps} lr={lr} warmup={warmup} seed={seed}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=False, text=True, timeout=7200,
    )
    elapsed = time.time() - t0
    print(f"  [{name}] exit={result.returncode} wall={elapsed:.0f}s", flush=True)
    return result.returncode


def load_extracted():
    """Load extracted spectra and Frobenius norms."""
    path = os.path.join(PROJECT_ROOT, "imt_gpt/results/extracted_spectra.json")
    with open(path) as f:
        data = json.load(f)
    return data


def main():
    os.chdir(PROJECT_ROOT)
    data = load_extracted()
    has_frobs = "group_frob_norms" in data

    print(f"\nDiagnostic suite starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"Extracted spectra has Frobenius norms: {has_frobs}", flush=True)

    # Build kwargs strings for subprocess
    spectra_kwargs = json.dumps({
        "spectra_coeffs": {k: list(v) if isinstance(v, list) else v
                          for k, v in data["spectra_coeffs"].items()}
    })
    frob_kwargs = json.dumps({
        "group_frob_norms": data.get("group_frob_norms", {})
    }) if has_frobs else "{}"

    tests = []

    # === TEST 1: imt_scaled_flat (the missing confound) ===
    if has_frobs:
        tests.append(("scaled_flat_s42", "imt_scaled_flat", 1000, 6.25e-5, 100, 42, frob_kwargs))

    # === TEST 2: Orthogonal LR ablation ===
    # 2a: Same LR, same schedule (control — reproduce the baseline)
    tests.append(("ortho_control_s42", "orthogonal", 1000, 6.25e-5, 100, 42, "{}"))
    # 2b: 3x LR (is it LR-starved?)
    tests.append(("ortho_3xlr_s42", "orthogonal", 1000, 1.875e-4, 100, 42, "{}"))
    # 2c: 2000 steps (does it just need more time?)
    tests.append(("ortho_2k_s42", "orthogonal", 2000, 6.25e-5, 200, 42, "{}"))

    # === TEST 3: Seeded runs for error bars (3 seeds) ===
    for seed in [42, 137, 512]:
        tests.append((f"extracted_s{seed}", "imt_shaped", 1000, 6.25e-5, 100, seed, spectra_kwargs))
        tests.append((f"ortho_s{seed}", "orthogonal", 1000, 6.25e-5, 100, seed, "{}"))
        tests.append((f"flat_s{seed}", "imt_flat", 1000, 6.25e-5, 100, seed, "{}"))

    # Remove duplicates (ortho_s42 = ortho_control_s42)
    seen = set()
    unique_tests = []
    for t in tests:
        if t[0] not in seen:
            seen.add(t[0])
            unique_tests.append(t)
    tests = unique_tests

    print(f"\n{len(tests)} tests queued:", flush=True)
    for t in tests:
        print(f"  {t[0]}: {t[1]} steps={t[2]} lr={t[3]} seed={t[5]}", flush=True)

    # Run sequentially (memory safety — one model at a time)
    completed = 0
    for name, method, steps, lr, warmup, seed, kwargs in tests:
        try:
            rc = run_single(name, method, steps, lr, warmup, seed, kwargs)
            if rc == 0:
                completed += 1
        except subprocess.TimeoutExpired:
            print(f"  [{name}] TIMEOUT", flush=True)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  DIAGNOSTICS COMPLETE: {completed}/{len(tests)} succeeded", flush=True)
    print(f"  Results in imt_gpt/results/diag_*.json", flush=True)
    print(f"{'='*60}", flush=True)

    # Touch sentinel
    with open(os.path.join(PROJECT_ROOT, "imt_gpt/results/.diag_done"), "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{completed}/{len(tests)} tests\n")


if __name__ == "__main__":
    main()
