"""
search.py — CMA-ES search over spectral shapes for transformer init.

33D search space: 4 spectra x 8 DCT coefficients + 1 global lambda.
Fitness: -log(val_perplexity) at step N.

SAFETY: Pre-flight memory checks, explicit cleanup between candidates,
auto-abort on memory pressure, clean shutdown support.
"""
import sys
import os
import time
import json
import math
import numpy as np

# Add parent dir for imt_auto imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imt_auto.imt.cmaes import CMAES
from imt_gpt.config import (
    TrainConfig, SearchConfig, get_device,
    check_memory_safe, preflight_memory_check,
    install_signal_handlers, is_shutdown_requested,
)
from imt_gpt.spectral_init import decode_search_vector, apply_spectral_init
from imt_gpt.train import train, _clear_memory


def fitness_fn(genome: np.ndarray, search_config: SearchConfig,
               train_config: TrainConfig) -> float:
    """Evaluate one genome: train for fitness_steps, return neg log perplexity.

    Higher is better (CMA-ES maximizes).
    Explicitly cleans up all GPU memory after each evaluation.
    """
    device = train_config.device or get_device()

    # Pre-flight: skip this candidate if memory is too low
    safe, level, detail = check_memory_safe(train_config.memory_pressure_threshold)
    if not safe:
        print(f"  [fitness skip: memory too low ({detail})]")
        _clear_memory(device)
        return -20.0

    # Check for shutdown
    if is_shutdown_requested():
        return -20.0

    decoded = decode_search_vector(genome, n_dct=search_config.n_dct_coeffs)
    spectra_coeffs = decoded["spectra_coeffs"]
    lam = decoded["lam"]

    def init_fn(model):
        apply_spectral_init(model, spectra_coeffs, lam=lam)

    # Short training run with safe config
    eval_config = TrainConfig(
        batch_size=train_config.batch_size,
        grad_accum_steps=train_config.grad_accum_steps,
        max_length=train_config.max_length,
        max_steps=search_config.fitness_steps,
        lr=train_config.lr,
        warmup_steps=min(train_config.warmup_steps, search_config.fitness_steps // 5),
        weight_decay=train_config.weight_decay,
        grad_clip=train_config.grad_clip,
        betas=train_config.betas,
        device=train_config.device,
        log_every=search_config.fitness_steps + 1,  # suppress logging
        eval_every=search_config.fitness_steps + 1,
        eval_steps=[search_config.fitness_steps],
        memory_check_interval=50,  # check often during search
        memory_pressure_threshold=train_config.memory_pressure_threshold,
    )

    try:
        result = train(eval_config, init_fn=init_fn, init_name="cmaes_candidate",
                       verbose=False, return_checkpoints=True)
        ppl = result["final_ppl"]
        if math.isnan(ppl) or math.isinf(ppl) or ppl > 1e6:
            return -20.0
        return -math.log(ppl)
    except (MemoryError, RuntimeError) as e:
        print(f"  [fitness error: {e}]")
        return -20.0
    except Exception as e:
        print(f"  [fitness error: {e}]")
        return -20.0
    finally:
        # SAFETY: always clean up, even on error
        _clear_memory(device)


def run_search(search_config: SearchConfig = None, train_config: TrainConfig = None,
               output_dir: str = "imt_gpt/results", verbose: bool = True):
    """Run CMA-ES search for optimal spectral initialization."""
    if search_config is None:
        search_config = SearchConfig()
    if train_config is None:
        train_config = TrainConfig()
        train_config.device = get_device()

    # Install safety infrastructure
    install_signal_handlers()

    dim = search_config.search_dim
    cma = CMAES(dim=dim, pop_size=search_config.population, sigma=search_config.sigma)

    os.makedirs(output_dir, exist_ok=True)

    best_genome = None
    best_fitness = -float("inf")
    history = []

    if verbose:
        safe, level, detail = check_memory_safe(train_config.memory_pressure_threshold)
        print(f"\n{'='*60}")
        print(f"  CMA-ES SPECTRAL SEARCH")
        print(f"  Dimensions: {dim} | Pop: {search_config.population} "
              f"| Gens: {search_config.generations}")
        print(f"  Fitness steps: {search_config.fitness_steps}")
        print(f"  batch={train_config.batch_size} x accum={train_config.grad_accum_steps}"
              f" | seq_len={train_config.max_length}")
        print(f"  memory: {detail} {'OK' if safe else 'LOW'}")
        print(f"{'='*60}\n")

    for gen in range(search_config.generations):
        # Check for shutdown
        if is_shutdown_requested():
            if verbose:
                print(f"\n  Clean shutdown at generation {gen}")
            break

        # Pre-flight memory check before each generation
        safe, level, detail = check_memory_safe(train_config.memory_pressure_threshold)
        if not safe:
            if verbose:
                print(f"\n  Memory abort at generation {gen}: {detail}")
            break

        t0 = time.time()
        solutions = cma.ask()
        fitnesses = []

        for i, genome in enumerate(solutions):
            # Check for shutdown between candidates
            if is_shutdown_requested():
                fitnesses.append(-20.0)
                continue

            genome = np.array(genome)
            f = fitness_fn(genome, search_config, train_config)
            fitnesses.append(f)

            if f > best_fitness:
                best_fitness = f
                best_genome = genome.copy()
                if verbose:
                    ppl = math.exp(-f)
                    print(f"  gen {gen:3d} ind {i:2d} | NEW BEST ppl={ppl:.2f} "
                          f"fitness={f:.4f}")

        cma.tell(solutions, fitnesses)

        gen_best = max(fitnesses)
        gen_mean = np.mean(fitnesses)
        gen_ppl = math.exp(-gen_best) if gen_best > -20 else float("inf")
        elapsed = time.time() - t0

        record = {
            "gen": gen,
            "best_fitness": float(gen_best),
            "mean_fitness": float(gen_mean),
            "best_ppl": float(gen_ppl),
            "global_best_fitness": float(best_fitness),
            "global_best_ppl": float(math.exp(-best_fitness)),
            "sigma": float(cma.sigma),
            "elapsed": float(elapsed),
        }
        history.append(record)

        if verbose:
            print(f"  gen {gen:3d} | best_ppl={gen_ppl:.2f} "
                  f"mean_f={gen_mean:.4f} | sigma={cma.sigma:.4f} | "
                  f"{elapsed:.1f}s")

        # Save checkpoint every 5 generations
        if (gen + 1) % 5 == 0 and best_genome is not None:
            checkpoint = {
                "gen": gen,
                "best_genome": best_genome.tolist(),
                "best_fitness": float(best_fitness),
                "best_ppl": float(math.exp(-best_fitness)),
                "history": history,
            }
            ckpt_path = os.path.join(output_dir, f"search_gen{gen+1:04d}.json")
            with open(ckpt_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

    if best_genome is None:
        print("  No valid results found.")
        return None, -float("inf"), history

    # Final save
    final = {
        "best_genome": best_genome.tolist(),
        "best_fitness": float(best_fitness),
        "best_ppl": float(math.exp(-best_fitness)),
        "decoded": decode_search_vector(best_genome, n_dct=search_config.n_dct_coeffs),
        "history": history,
        "config": {
            "dim": dim,
            "population": search_config.population,
            "generations": search_config.generations,
            "fitness_steps": search_config.fitness_steps,
        },
    }
    for k, v in final["decoded"]["spectra_coeffs"].items():
        if isinstance(v, np.ndarray):
            final["decoded"]["spectra_coeffs"][k] = v.tolist()

    final_path = os.path.join(output_dir, "search_final.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)

    if verbose:
        ppl = math.exp(-best_fitness)
        print(f"\n{'='*60}")
        print(f"  SEARCH COMPLETE")
        print(f"  Best perplexity: {ppl:.2f}")
        print(f"  Lambda: {final['decoded']['lam']:.4f}")
        print(f"  Saved: {final_path}")
        print(f"{'='*60}\n")

    return best_genome, best_fitness, history


if __name__ == "__main__":
    run_search()
