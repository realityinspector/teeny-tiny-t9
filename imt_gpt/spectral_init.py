"""
spectral_init.py — SVD-based spectral initialization for transformer weights.

Core idea: decompose each weight matrix via SVD, replace singular values
with a shaped distribution (from DCT coefficients), reconstruct.
Preserves random U/V directions and Frobenius norm.
"""
import torch
import numpy as np
from typing import Dict, Optional


def dct_expand(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Expand DCT coefficients to n spectral values.

    Args:
        coeffs: array of k DCT coefficients
        n: target number of spectral values

    Returns:
        Smooth spectrum of length n
    """
    k = len(coeffs)
    t = np.linspace(0, np.pi, n, endpoint=False)
    spectrum = np.zeros(n)
    for i in range(k):
        spectrum += coeffs[i] * np.cos((i + 0.5) * t)
    # Softplus to ensure positive values
    spectrum = np.log(1.0 + np.exp(np.clip(spectrum, -10, 10)))
    # Normalize to [0, 1]
    s_max = spectrum.max()
    if s_max > 0:
        spectrum = spectrum / s_max
    return spectrum


def spectral_init_weight(W: torch.Tensor, target_spectrum: np.ndarray,
                         lam: float = 1.0,
                         target_frob: Optional[float] = None) -> torch.Tensor:
    """Replace singular values of W with shaped distribution.

    Preserves U, V directions (random). Preserves Frobenius norm
    (or matches target_frob if provided).

    Args:
        W: weight matrix (any shape, treated as 2D)
        target_spectrum: normalized spectrum values in [0, 1]
        lam: blending strength. 0=flat (Xavier-like), 1=full shape
        target_frob: if set, scale to this Frobenius norm instead of W's

    Returns:
        Reshaped weight matrix
    """
    orig_shape = W.shape
    # Handle > 2D tensors by reshaping
    if W.dim() > 2:
        W_2d = W.reshape(W.shape[0], -1)
    else:
        W_2d = W

    U, s, Vt = torch.linalg.svd(W_2d, full_matrices=False)
    frob = target_frob if target_frob is not None else torch.norm(W_2d, 'fro').item()
    n = len(s)

    # Interpolate target spectrum to match matrix rank
    target = np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(target_spectrum)),
        target_spectrum
    )
    target_t = torch.tensor(target, dtype=s.dtype, device=s.device)

    # Blend: lam=0 -> flat (Xavier-like), lam=1 -> full shape
    flat = torch.ones_like(s)
    shaped = torch.clamp(target_t, min=0.01)
    blended = flat + lam * (shaped - flat)
    blended = torch.clamp(blended, min=0.01)

    # Norm-match to target Frobenius norm
    s_new = blended * (frob / torch.norm(blended).item())

    W_new = U @ torch.diag(s_new) @ Vt
    return W_new.reshape(orig_shape)


# Matrix group names for the 4-spectrum grouping
ATTN_PROJ = "attention"   # Q, K, V, O projections
FFN_UP = "ffn_up"         # FFN first linear (expand to 4x)
FFN_DOWN = "ffn_down"     # FFN second linear (project back)
EMBED = "embedding"       # Token + position embeddings


def classify_param(name: str) -> Optional[str]:
    """Classify a transformer parameter name into a spectrum group.

    Returns None for parameters that should not be spectrally initialized
    (biases, layer norms).
    """
    # Skip biases and layer norms
    if "bias" in name or "ln" in name or "layernorm" in name or "norm" in name:
        return None

    # Attention projections (Q, K, V, O)
    if any(k in name for k in ["c_attn", "q_proj", "k_proj", "v_proj"]):
        return ATTN_PROJ
    if any(k in name for k in ["c_proj", "o_proj"]) and "mlp" not in name:
        # c_proj in attention (not MLP) is the output projection
        if "attn" in name:
            return ATTN_PROJ

    # FFN layers
    if any(k in name for k in ["c_fc", "up_proj", "gate_proj"]):
        return FFN_UP
    if "mlp" in name and any(k in name for k in ["c_proj", "down_proj"]):
        return FFN_DOWN

    # Embeddings
    if any(k in name for k in ["wte", "wpe", "embed"]):
        return EMBED

    return None


def decode_search_vector(genome: np.ndarray, n_dct: int = 8) -> Dict:
    """Decode a CMA-ES search vector into per-group spectra + lambda.

    Genome layout (33D default):
        [0:8]   attention DCT coefficients
        [8:16]  ffn_up DCT coefficients
        [16:24] ffn_down DCT coefficients
        [24:32] embedding DCT coefficients
        [32]    global lambda (raw, softplus-transformed)

    Returns:
        Dict with keys: spectra (dict of group -> spectrum), lam (float)
    """
    groups = [ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED]
    spectra = {}
    for i, group in enumerate(groups):
        coeffs = genome[i * n_dct:(i + 1) * n_dct]
        # We'll expand to target size at init time (depends on matrix shape)
        spectra[group] = coeffs

    lam_raw = genome[len(groups) * n_dct]
    lam = float(np.log(1.0 + np.exp(np.clip(lam_raw, -10, 10))))

    return {"spectra_coeffs": spectra, "lam": lam}


def apply_spectral_init(model, spectra_coeffs: Dict[str, np.ndarray],
                        lam: float = 1.0, verbose: bool = False,
                        group_frob_norms: Optional[Dict[str, float]] = None):
    """Apply spectral initialization to all eligible weight matrices in a model.

    Args:
        model: HuggingFace GPT-2 model
        spectra_coeffs: dict of group_name -> DCT coefficients
        lam: global blending strength
        verbose: print per-layer info
        group_frob_norms: if set, per-group target Frobenius norms
            (overrides the default of preserving each matrix's own norm)
    """
    n_shaped = 0
    n_skipped = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                n_skipped += 1
                continue

            group = classify_param(name)
            if group is None:
                n_skipped += 1
                continue

            if group not in spectra_coeffs:
                n_skipped += 1
                continue

            coeffs = spectra_coeffs[group]
            # Expand DCT to match the smaller dimension of the weight matrix
            if param.dim() > 2:
                n_sv = min(param.shape[0], np.prod(param.shape[1:]))
            else:
                n_sv = min(param.shape)

            target_spectrum = dct_expand(coeffs, n_sv)
            tf = group_frob_norms.get(group) if group_frob_norms else None
            param.data = spectral_init_weight(param.data, target_spectrum,
                                              lam=lam, target_frob=tf)
            n_shaped += 1

            if verbose:
                print(f"  {name:50s} {str(list(param.shape)):20s} -> {group}")

    if verbose:
        print(f"\nShaped: {n_shaped} matrices, Skipped: {n_skipped} params")

    return n_shaped
