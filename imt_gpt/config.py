"""Training + search hyperparameters for IMT-GPT experiments.

SAFETY: Tuned for M1 MacBook Pro 16GB. Do NOT raise batch_size or
max_length without checking memory first.
"""
import gc
import signal
import subprocess
import resource
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # Model
    model_name: str = "gpt2"  # GPT-2 small (same arch as GPT-1)

    # Data
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 256  # SAFE: 256 tokens, not 1024 (4x less memory)

    # Training — effective batch = batch_size * grad_accum_steps = 2*4 = 8
    batch_size: int = 2        # SAFE: small per-step batch for M1 16GB
    grad_accum_steps: int = 4  # accumulate to effective batch of 8
    max_steps: int = 5000      # measured in optimizer steps (not micro-batches)
    lr: float = 6.25e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.999)

    # Logging
    log_every: int = 50
    eval_every: int = 500
    eval_steps: list = field(default_factory=lambda: [100, 500, 1000, 2000, 5000])

    # Device
    device: Optional[str] = None  # auto-detect

    # Reproducibility
    seed: Optional[int] = None  # set for reproducible runs

    # Memory safety
    max_memory_gb: float = 10.0  # abort if PyTorch uses more than this
    memory_check_interval: int = 25  # check every N optimizer steps
    memory_pressure_threshold: int = 15  # abort below this level (0-100)


@dataclass
class SearchConfig:
    # CMA-ES
    population: int = 8   # SAFE: reduced from 16 (each trains a model)
    generations: int = 30  # reduced from 50 for initial safety
    sigma: float = 0.5

    # Spectrum parameterization
    n_dct_coeffs: int = 8
    n_spectra: int = 4  # attention, ffn_up, ffn_down, embeddings
    # Total search dims: n_spectra * n_dct_coeffs + 1 (global lambda) = 33

    # Fitness evaluation
    fitness_steps: int = 500   # SAFE: reduced from 1000
    fitness_seeds: int = 1

    @property
    def search_dim(self):
        return self.n_spectra * self.n_dct_coeffs + 1


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_level():
    """Get macOS memory pressure level (0-100, higher = more free).

    Returns None if not on macOS or can't determine.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def get_process_rss_gb():
    """Get this process's RSS (resident set size) in GB."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on macOS
        return ru.ru_maxrss / (1024 ** 3)
    except Exception:
        return 0.0


def check_memory_safe(threshold: int = 15):
    """Check if it's safe to continue training.

    Args:
        threshold: minimum memory pressure level (0-100).
            15 = abort when <15% free. Default is conservative for M1 16GB.

    Returns:
        (safe: bool, level: int or None, detail: str)
    """
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        level = get_memory_level()
        if level is None:
            return True, None, "unknown"
        rss = get_process_rss_gb()
        detail = f"level={level}% rss={rss:.1f}GB"
        return level > threshold, level, detail

    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        detail = f"alloc={allocated:.1f}GB reserved={reserved:.1f}GB"
        return allocated < 10.0, None, detail

    return True, None, "cpu"


def preflight_memory_check(config, label="training"):
    """Check memory BEFORE starting a training run. Raises if unsafe.

    Estimates memory needed and compares to available.
    """
    level = get_memory_level()
    if level is not None and level < config.memory_pressure_threshold:
        raise MemoryError(
            f"System memory too low to start {label}: "
            f"level={level}% (need >{config.memory_pressure_threshold}%). "
            f"Close other apps or reduce config."
        )

    # Estimate: GPT-2 small ~500MB model, ~1.5GB optimizer states,
    # ~0.5-2GB activations depending on batch/seq
    est_model_gb = 0.5
    est_optimizer_gb = 1.5  # AdamW stores m and v
    est_activations_gb = (
        config.batch_size * config.max_length * 768 * 12 * 4  # rough
    ) / (1024 ** 3)
    est_total = est_model_gb + est_optimizer_gb + est_activations_gb

    if level is not None:
        # level is % free of 16GB total
        free_gb = level / 100.0 * 16.0
        if est_total > free_gb * 0.7:  # leave 30% headroom
            print(f"  WARNING: {label} estimated {est_total:.1f}GB, "
                  f"~{free_gb:.1f}GB free. Proceeding cautiously.")


def set_process_memory_limit(max_gb: float = 12.0):
    """Set a hard RSS limit on this process to prevent system-wide OOM.

    If the process exceeds this, Python gets a MemoryError instead of
    the kernel killing random processes or the system freezing.
    """
    max_bytes = int(max_gb * 1024 ** 3)
    try:
        # RLIMIT_RSS is advisory on modern macOS but still helps
        soft, hard = resource.getrlimit(resource.RLIMIT_RSS)
        resource.setrlimit(resource.RLIMIT_RSS, (max_bytes, hard))
    except (ValueError, resource.error):
        pass  # some systems don't support this

    # Also limit virtual memory (more enforceable)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Be generous — virtual memory is overcommitted
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes * 2, hard))
    except (ValueError, resource.error):
        pass


# Global flag for clean shutdown
_shutdown_requested = False


def request_shutdown():
    """Signal handler sets this flag for clean training abort."""
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested():
    """Check if a clean shutdown was requested (Ctrl-C)."""
    return _shutdown_requested


def install_signal_handlers():
    """Install Ctrl-C handler that requests clean shutdown instead of crashing.

    First Ctrl-C: sets shutdown flag (training loop exits cleanly, frees memory).
    Second Ctrl-C: hard exit (in case clean shutdown is stuck).
    """
    original_handler = signal.getsignal(signal.SIGINT)
    hit_count = [0]

    def handler(signum, frame):
        hit_count[0] += 1
        if hit_count[0] == 1:
            print("\n  [Ctrl-C] Requesting clean shutdown... "
                  "(press again to force-quit)")
            request_shutdown()
        else:
            print("\n  [Ctrl-C] Force quit.")
            # Attempt cleanup
            import torch
            gc.collect()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    return original_handler
