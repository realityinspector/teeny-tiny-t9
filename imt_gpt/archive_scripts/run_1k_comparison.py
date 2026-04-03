#!/usr/bin/env python3
"""
run_1k_comparison.py — 1000-step comparison of top init methods.

Standalone script designed to run via nohup for long training.
Writes results incrementally so progress is observable.
"""
import json
import numpy as np
import sys
import os
import gc
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force unbuffered output for nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from imt_gpt.config import (
    TrainConfig, get_device, install_signal_handlers,
    check_memory_safe, set_process_memory_limit,
)
from imt_gpt.train import train, _clear_memory
from imt_gpt.baselines import make_init_fn


def main():
    install_signal_handlers()
    set_process_memory_limit(max_gb=10.0)
    device = get_device()
    output_dir = "imt_gpt/results"
    os.makedirs(output_dir, exist_ok=True)

    # Load extracted spectra
    with open(os.path.join(output_dir, "extracted_spectra.json")) as f:
        data = json.load(f)
    spectra = {k: np.array(v) for k, v in data["spectra_coeffs"].items()}

    # 1000-step config
    config = TrainConfig(
        max_steps=1000,
        eval_steps=[100, 250, 500, 750, 1000],
        warmup_steps=100,
        log_every=50,
        device=device,
    )

    methods = [
        ("orthogonal", make_init_fn("orthogonal")),
        ("imt_flat", make_init_fn("imt_flat")),
        ("imt_extracted", make_init_fn("imt_shaped", spectra_coeffs=spectra, lam=1.0)),
    ]

    results = {}
    for name, init_fn in methods:
        safe, level, detail = check_memory_safe(config.memory_pressure_threshold)
        print(f"\n>>> Pre-flight for {name}: {detail}", flush=True)
        if not safe:
            print(f"Memory too low ({detail}), stopping.", flush=True)
            break

        t0 = time.time()
        try:
            result = train(config, init_fn=init_fn, init_name=name, verbose=True)
            results[name] = result

            # Write result immediately
            path = os.path.join(output_dir, f"cmp1k_{name}.json")
            with open(path, "w") as f:
                json.dump({
                    "init_name": name,
                    "final_ppl": result["final_ppl"],
                    "elapsed": result["elapsed"],
                    "checkpoints": {str(k): v for k, v in result["checkpoints"].items()},
                    "loss_samples": result["losses"][::max(1, len(result["losses"]) // 30)],
                }, f, indent=2)
            print(f"  Saved {path} (PPL={result['final_ppl']:.2f}, {result['elapsed']:.0f}s)", flush=True)

        except Exception as e:
            print(f"  Error training {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        finally:
            _clear_memory(device)
            gc.collect()
            elapsed = time.time() - t0
            print(f"  Method {name} total wall time: {elapsed:.0f}s", flush=True)

    # Summary
    if results:
        print(f"\n{'='*70}", flush=True)
        print(f"  1000-STEP COMPARISON", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  {'Method':<20s} {'Final PPL':>10s} {'Time':>8s}  Checkpoints", flush=True)
        print(f"  {'-'*60}", flush=True)
        for name, r in sorted(results.items(), key=lambda x: x[1]["final_ppl"]):
            ckpts = " ".join(f"{s}:{r['checkpoints'][s]:.0f}"
                             for s in sorted(r["checkpoints"].keys()))
            print(f"  {name:<20s} {r['final_ppl']:>10.2f} {r['elapsed']:>7.0f}s  {ckpts}", flush=True)

        # Save combined
        combined_path = os.path.join(output_dir, "comparison_1k.json")
        combined = {}
        for name, r in results.items():
            combined[name] = {
                "final_ppl": r["final_ppl"],
                "elapsed": r["elapsed"],
                "checkpoints": {str(k): v for k, v in r["checkpoints"].items()},
            }
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved {combined_path}", flush=True)

    # Touch a sentinel file to signal completion
    with open(os.path.join(output_dir, ".1k_done"), "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("\n=== 1000-step comparison COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
