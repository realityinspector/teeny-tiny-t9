"""
eval.py — Run all baselines + IMT shaped init, collect convergence data.

SAFETY: Pre-flight checks before each run, skip remaining if memory critical.
"""
import os
import json
import numpy as np

from imt_gpt.config import (
    TrainConfig, get_device, check_memory_safe,
    preflight_memory_check, is_shutdown_requested,
)
from imt_gpt.train import train, _clear_memory
from imt_gpt.baselines import make_init_fn, BASELINE_METHODS
from imt_gpt.spectral_init import decode_search_vector


def run_baselines(config: TrainConfig = None, output_dir: str = "imt_gpt/results",
                  verbose: bool = True):
    """Train all baseline init methods and collect convergence data."""
    if config is None:
        config = TrainConfig()
        config.device = get_device()

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    device = config.device or get_device()
    for method in BASELINE_METHODS:
        # Check for shutdown
        if is_shutdown_requested():
            if verbose:
                print(f"\n  Shutdown requested, skipping remaining baselines.")
            break

        # Pre-flight memory check before each baseline
        safe, level, detail = check_memory_safe(config.memory_pressure_threshold)
        if not safe:
            if verbose:
                print(f"\n  Memory too low for {method} ({detail}), skipping remaining.")
            break

        if verbose:
            print(f"\n--- Running baseline: {method} ---")

        try:
            preflight_memory_check(config, label=f"baseline:{method}")
            init_fn = make_init_fn(method)
            result = train(config, init_fn=init_fn, init_name=method, verbose=verbose)
            results[method] = result
            save_result(result, os.path.join(output_dir, f"baseline_{method}.json"))
        except MemoryError as e:
            if verbose:
                print(f"\n  Memory error during {method}: {e}")
                print(f"  Skipping remaining baselines.")
            break
        except Exception as e:
            if verbose:
                print(f"\n  Error during {method}: {e}")
            continue
        finally:
            _clear_memory(device)

    return results


def run_imt_shaped(genome_path: str, config: TrainConfig = None,
                   output_dir: str = "imt_gpt/results", verbose: bool = True):
    """Train with IMT shaped init from a CMA-ES search result."""
    if config is None:
        config = TrainConfig()
        config.device = get_device()

    os.makedirs(output_dir, exist_ok=True)

    with open(genome_path) as f:
        search_result = json.load(f)

    genome = np.array(search_result["best_genome"])
    decoded = decode_search_vector(genome)
    spectra_coeffs = {}
    for k, v in decoded["spectra_coeffs"].items():
        spectra_coeffs[k] = np.array(v)
    lam = decoded["lam"]

    init_fn = make_init_fn("imt_shaped", spectra_coeffs=spectra_coeffs, lam=lam)
    result = train(config, init_fn=init_fn, init_name="imt_shaped", verbose=verbose)

    save_result(result, os.path.join(output_dir, "imt_shaped.json"))
    return result


def run_extracted(extracted_path: str, config: TrainConfig = None,
                  output_dir: str = "imt_gpt/results", verbose: bool = True):
    """Train with spectra extracted from pretrained GPT-2."""
    if config is None:
        config = TrainConfig()
        config.device = get_device()

    os.makedirs(output_dir, exist_ok=True)

    with open(extracted_path) as f:
        data = json.load(f)

    spectra_coeffs = {}
    for k, v in data["spectra_coeffs"].items():
        spectra_coeffs[k] = np.array(v)
    lam = data.get("lam", 1.0)

    init_fn = make_init_fn("imt_shaped", spectra_coeffs=spectra_coeffs, lam=lam)
    result = train(config, init_fn=init_fn, init_name="imt_extracted", verbose=verbose)

    save_result(result, os.path.join(output_dir, "imt_extracted.json"))
    return result


def run_all(genome_path: str = None, extracted_path: str = None,
            config: TrainConfig = None,
            output_dir: str = "imt_gpt/results", verbose: bool = True):
    """Run all baselines + IMT shaped (if genome available) + extracted."""
    results = run_baselines(config=config, output_dir=output_dir, verbose=verbose)

    if genome_path and os.path.exists(genome_path):
        result = run_imt_shaped(genome_path, config=config,
                                output_dir=output_dir, verbose=verbose)
        results["imt_shaped"] = result

    if extracted_path and os.path.exists(extracted_path):
        result = run_extracted(extracted_path, config=config,
                               output_dir=output_dir, verbose=verbose)
        results["imt_extracted"] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"  CONVERGENCE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Method':<20s} {'Final PPL':>10s} {'Time':>8s}")
    print(f"  {'-'*38}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["final_ppl"]):
        print(f"  {name:<20s} {r['final_ppl']:>10.2f} {r['elapsed']:>7.0f}s")

    if results:
        # Checkpoint comparison
        first = next(iter(results.values()))
        if first["checkpoints"]:
            print(f"\n  Perplexity at checkpoints:")
            steps = sorted(first["checkpoints"].keys())
            header = f"  {'Method':<20s}" + "".join(f"  step {s:>5d}" for s in steps)
            print(header)
            for name, r in sorted(results.items(), key=lambda x: x[1]["final_ppl"]):
                vals = "".join(
                    f"  {r['checkpoints'].get(s, float('nan')):>10.2f}" for s in steps
                )
                print(f"  {name:<20s}{vals}")

    # Save combined results
    save_all_results(results, os.path.join(output_dir, "comparison.json"))
    return results


def save_result(result, path):
    """Save a single training result to JSON."""
    serializable = {
        "init_name": result["init_name"],
        "final_ppl": result["final_ppl"],
        "elapsed": result["elapsed"],
        "n_params": result["n_params"],
        "checkpoints": {str(k): v for k, v in result["checkpoints"].items()},
        "loss_samples": result["losses"][::50],  # sample every 50 steps
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_all_results(results, path):
    """Save combined comparison results."""
    combined = {}
    for name, r in results.items():
        combined[name] = {
            "final_ppl": r["final_ppl"],
            "elapsed": r["elapsed"],
            "checkpoints": {str(k): v for k, v in r["checkpoints"].items()},
        }
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)


if __name__ == "__main__":
    import sys
    genome_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_all(genome_path=genome_path)
