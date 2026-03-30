"""
baselines.py — Baseline initialization methods for GPT-2 small.
Standard N(0,0.02), Xavier, orthogonal, and flat-spectrum ablation.
"""
import torch
import torch.nn as nn
import numpy as np

from imt_gpt.spectral_init import apply_spectral_init, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED


def init_standard(model):
    """GPT-2 default: N(0, 0.02) for all weights, zeros for biases.

    This is the standard GPT-2 initialization with residual scaling.
    HuggingFace already does this in GPT2LMHeadModel.__init__,
    so this is a no-op — included for API consistency.
    """
    # GPT2LMHeadModel already initializes with N(0, 0.02)
    # Just verify it's correct
    pass


def init_xavier(model):
    """Xavier/Glorot initialization: variance = 2/(fan_in + fan_out)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def init_orthogonal(model):
    """Orthogonal initialization: Q factor of random Gaussian matrix."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def init_imt_flat(model, lam=1.0):
    """Flat spectrum with norm matching (ablation).

    All singular values equal, but norm-matched to Xavier scale.
    Tests whether spectral SHAPE matters or just the overall SCALE.
    """
    n_dct = 8
    # Flat spectrum: all DCT coefficients = 0 except DC component
    flat_coeffs = np.zeros(n_dct)
    flat_coeffs[0] = 1.0  # DC only -> flat after DCT expansion

    spectra = {
        ATTN_PROJ: flat_coeffs,
        FFN_UP: flat_coeffs,
        FFN_DOWN: flat_coeffs,
        EMBED: flat_coeffs,
    }
    apply_spectral_init(model, spectra, lam=lam)


def init_imt_shaped(model, spectra_coeffs, lam=1.0):
    """IMT shaped spectrum from CMA-ES search or manual design."""
    apply_spectral_init(model, spectra_coeffs, lam=lam)


def make_init_fn(method: str, **kwargs):
    """Factory: return an init_fn(model) for the given method name."""
    if method == "standard":
        return init_standard
    elif method == "xavier":
        return init_xavier
    elif method == "orthogonal":
        return init_orthogonal
    elif method == "imt_flat":
        lam = kwargs.get("lam", 1.0)
        return lambda model: init_imt_flat(model, lam=lam)
    elif method == "imt_shaped":
        spectra = kwargs["spectra_coeffs"]
        lam = kwargs.get("lam", 1.0)
        return lambda model: init_imt_shaped(model, spectra, lam=lam)
    else:
        raise ValueError(f"Unknown init method: {method}")


# All baseline methods for comparison runs
BASELINE_METHODS = ["standard", "xavier", "orthogonal", "imt_flat"]
