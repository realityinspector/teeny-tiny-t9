#!/usr/bin/env python3
"""
compare.py — Quick comparison of all init methods on the same config.
Runs each method once, prints a table, generates plots.

Usage:
    python -m imt_gpt.compare              # Quick 200-step comparison
    python -m imt_gpt.compare --steps 1000 # Custom step count
"""
import sys
import os
import json
import gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_comparison(max_steps=200, output_dir="imt_gpt/results"):
    from imt_gpt.config import (
        TrainConfig, get_device, check_memory_safe,
        preflight_memory_check, is_shutdown_requested,
        install_signal_handlers,
    )
    from imt_gpt.train import train, _clear_memory
    from imt_gpt.baselines import make_init_fn

    device = get_device()
    install_signal_handlers()

    # Eval steps at 25%, 50%, 75%, 100%
    eval_at = sorted(set([
        max(1, max_steps // 4),
        max(1, max_steps // 2),
        max(1, 3 * max_steps // 4),
        max_steps,
    ]))

    config = TrainConfig(
        max_steps=max_steps,
        eval_steps=eval_at,
        warmup_steps=min(50, max_steps // 4),
        log_every=max(1, max_steps // 8),
        device=device,
    )

    # Load extracted spectra if available
    extracted_path = os.path.join(output_dir, "extracted_spectra.json")
    extracted_spectra = None
    if os.path.exists(extracted_path):
        with open(extracted_path) as f:
            data = json.load(f)
        extracted_spectra = {k: np.array(v) for k, v in data["spectra_coeffs"].items()}

    # Methods to test
    methods = [
        ("standard", make_init_fn("standard")),
        ("xavier", make_init_fn("xavier")),
        ("orthogonal", make_init_fn("orthogonal")),
        ("imt_flat", make_init_fn("imt_flat")),
    ]
    if extracted_spectra is not None:
        methods.append((
            "imt_extracted",
            make_init_fn("imt_shaped", spectra_coeffs=extracted_spectra, lam=1.0)
        ))

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, init_fn in methods:
        # Check for shutdown or memory pressure
        if is_shutdown_requested():
            print(f"\n  Shutdown requested, skipping remaining methods.")
            break

        safe, _, detail = check_memory_safe(config.memory_pressure_threshold)
        if not safe:
            print(f"\n  Memory too low ({detail}), skipping remaining methods.")
            break

        try:
            result = train(config, init_fn=init_fn, init_name=name, verbose=True)
            results[name] = result

            with open(os.path.join(output_dir, f"cmp_{name}.json"), "w") as f:
                json.dump({
                    "init_name": name,
                    "final_ppl": result["final_ppl"],
                    "elapsed": result["elapsed"],
                    "checkpoints": {str(k): v for k, v in result["checkpoints"].items()},
                    "loss_samples": result["losses"][::max(1, len(result["losses"]) // 20)],
                }, f, indent=2)
        except MemoryError as e:
            print(f"\n  Memory error during {name}: {e}")
            break
        except Exception as e:
            print(f"\n  Error during {name}: {e}")
            continue
        finally:
            _clear_memory(device)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  INIT COMPARISON ({max_steps} steps, seq={config.max_length}, "
          f"eff_batch={config.batch_size * config.grad_accum_steps})")
    print(f"{'='*70}")
    print(f"  {'Method':<20s} {'Final PPL':>10s} {'Time':>8s}  Checkpoints")
    print(f"  {'-'*65}")

    for name, r in sorted(results.items(), key=lambda x: x[1]["final_ppl"]):
        ckpts = " ".join(f"{s}:{r['checkpoints'][s]:.0f}"
                         for s in sorted(r["checkpoints"].keys()))
        print(f"  {name:<20s} {r['final_ppl']:>10.2f} {r['elapsed']:>7.0f}s  {ckpts}")

    # Save combined
    combined_path = os.path.join(output_dir, "comparison.json")
    combined = {}
    for name, r in results.items():
        combined[name] = {
            "final_ppl": r["final_ppl"],
            "elapsed": r["elapsed"],
            "checkpoints": {str(k): v for k, v in r["checkpoints"].items()},
        }
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Saved: {combined_path}")

    # Generate plots
    try:
        from imt_gpt.plot import plot_loss_curves, plot_perplexity_checkpoints
        plot_loss_curves(results, output_dir)
        plot_perplexity_checkpoints(results, output_dir)
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    return results


if __name__ == "__main__":
    steps = 200
    for arg in sys.argv[1:]:
        if arg.startswith("--steps"):
            steps = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg.isdigit():
            steps = int(arg)

    run_comparison(max_steps=steps)
