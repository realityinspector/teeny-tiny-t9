"""
plot.py — Loss curves, spectrum visualizations, convergence comparison.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imt_gpt.spectral_init import dct_expand


def plot_loss_curves(results: dict, output_dir: str = "imt_gpt/results"):
    """Plot training loss curves for all methods."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, r in results.items():
        losses = r["losses"]
        # Smooth with rolling average
        window = 50
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(losses)), smoothed, label=name, alpha=0.8)
        else:
            ax.plot(losses, label=name, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss: IMT Spectral Init vs Baselines")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_perplexity_checkpoints(results: dict, output_dir: str = "imt_gpt/results"):
    """Plot perplexity at checkpoint steps."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, r in results.items():
        checkpoints = r["checkpoints"]
        if checkpoints:
            steps = sorted(checkpoints.keys())
            ppls = [checkpoints[s] for s in steps]
            ax.plot(steps, ppls, "o-", label=name, markersize=8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Convergence: Perplexity at Checkpoints")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    path = os.path.join(output_dir, "perplexity_checkpoints.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_spectra(spectra_coeffs: dict, output_dir: str = "imt_gpt/results"):
    """Visualize the spectral shapes for each matrix group."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    groups = list(spectra_coeffs.keys())
    for idx, (group, coeffs) in enumerate(spectra_coeffs.items()):
        ax = axes[idx // 2][idx % 2]
        coeffs = np.array(coeffs)

        # Expand to different target sizes
        for n, label in [(64, "64 SVs"), (768, "768 SVs"), (3072, "3072 SVs")]:
            spectrum = dct_expand(coeffs, n)
            ax.plot(np.linspace(0, 1, n), spectrum, label=label, alpha=0.7)

        ax.set_title(f"{group}")
        ax.set_xlabel("Singular value index (normalized)")
        ax.set_ylabel("Magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("IMT Spectral Shapes (DCT-expanded)", fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, "spectra.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_search_history(history_path: str, output_dir: str = "imt_gpt/results"):
    """Plot CMA-ES search progress."""
    os.makedirs(output_dir, exist_ok=True)

    with open(history_path) as f:
        data = json.load(f)
    history = data["history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gens = [h["gen"] for h in history]
    best_ppls = [h["global_best_ppl"] for h in history]
    sigmas = [h["sigma"] for h in history]

    ax1.plot(gens, best_ppls, "b-", linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Perplexity")
    ax1.set_title("CMA-ES Search Progress")
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, sigmas, "r-", linewidth=2)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Step Size (sigma)")
    ax2.set_title("CMA-ES Step Size")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "search_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_all_from_dir(output_dir: str = "imt_gpt/results"):
    """Load saved results and generate all plots."""
    # Load comparison data
    comp_path = os.path.join(output_dir, "comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)
        # Convert checkpoint keys back to int
        for name in comparison:
            comparison[name]["checkpoints"] = {
                int(k): v for k, v in comparison[name]["checkpoints"].items()
            }
        plot_perplexity_checkpoints(comparison, output_dir)

    # Load search results for spectra plot
    search_path = os.path.join(output_dir, "search_final.json")
    if os.path.exists(search_path):
        with open(search_path) as f:
            search = json.load(f)
        spectra = search["decoded"]["spectra_coeffs"]
        plot_spectra(spectra, output_dir)
        plot_search_history(search_path, output_dir)


if __name__ == "__main__":
    plot_all_from_dir()
