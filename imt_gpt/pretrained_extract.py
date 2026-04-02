"""
pretrained_extract.py — Extract per-layer SVD data from pretrained models.

Extracts singular values AND vectors per weight matrix, enabling:
1. Per-layer spectra (not just group-averaged)
2. Directional alignment (use pretrained U/V, not random)
3. Cross-model transfer (extract from medium, apply to small)

The extracted data is a dict keyed by parameter name, each containing:
  - svs: normalized singular values
  - U: left singular vectors (optional, for directional alignment)
  - V: right singular vectors (optional)
  - group: which spectral group this belongs to
  - shape: original weight shape
"""
import gc
import json
import os
import numpy as np
import torch
from imt_gpt.spectral_init import classify_param, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED


def extract_per_layer(model_name="gpt2", include_directions=False, device="cpu"):
    """Extract per-layer SVD data from a pretrained model.

    Args:
        model_name: HuggingFace model name ("gpt2", "gpt2-medium", etc.)
        include_directions: if True, also store U and V matrices
            (large — ~2GB for GPT-2 small, ~8GB for medium)
        device: device for SVD computation

    Returns:
        dict with per-parameter SVD data
    """
    from transformers import GPT2LMHeadModel

    print(f"  Extracting from {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)

    layers = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_param(name)
            if group is None:
                continue

            W = param.data.float().to(device)
            orig_shape = W.shape
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)

            U, s, Vt = torch.linalg.svd(W, full_matrices=False)
            s_norm = s / s.max() if s.max() > 0 else s

            entry = {
                "svs": s_norm.cpu().numpy(),
                "group": group,
                "shape": list(orig_shape),
                "frob": torch.norm(W, 'fro').item(),
            }

            if include_directions:
                entry["U"] = U.cpu().numpy()
                entry["Vt"] = Vt.cpu().numpy()

            layers[name] = entry
            print(f"    {name:50s} {str(list(orig_shape)):20s} {group:12s} "
                  f"top3_sv=[{s_norm[0]:.3f},{s_norm[1]:.3f},{s_norm[2]:.3f}]")

    del model
    gc.collect()
    return layers


def apply_per_layer_spectral(model, extracted, lam=1.0, align_mode="none",
                             align_strength=0.0, noise_std=0.0, verbose=False):
    """Apply per-layer spectral init with optional directional alignment.

    Args:
        model: fresh GPT-2 model to initialize
        extracted: output of extract_per_layer()
        lam: spectral blending strength (0=flat, 1=full shape)
        align_mode: how to use pretrained directions
            "none" — random U/V (current approach, but per-layer spectra)
            "V" — align right singular vectors with pretrained V
            "U" — align left singular vectors with pretrained U
            "UV" — align both
        align_strength: 0.0=fully random, 1.0=fully pretrained directions
            Values in between do geodesic interpolation on the Stiefel manifold
            (approximated as linear blend + re-orthogonalization)
        noise_std: add Gaussian noise to the extracted singular values
            before applying. Tests robustness of the spectral signal.
        verbose: print per-layer info
    """
    n_shaped = 0
    n_skipped = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                n_skipped += 1
                continue

            if name not in extracted:
                n_skipped += 1
                continue

            ext = extracted[name]
            target_svs = ext["svs"].copy()

            # Add noise if requested
            if noise_std > 0:
                noise = np.random.randn(*target_svs.shape) * noise_std
                target_svs = np.clip(target_svs + noise, 0.01, None)
                target_svs = target_svs / target_svs.max()

            orig_shape = param.shape
            W = param.data.float()
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)

            # SVD of the fresh random weight
            U_fresh, s_fresh, Vt_fresh = torch.linalg.svd(W, full_matrices=False)
            frob = torch.norm(W, 'fro').item()
            n = len(s_fresh)

            # Interpolate target spectrum to match matrix rank
            target_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(target_svs)),
                target_svs
            )
            target_t = torch.tensor(target_interp, dtype=s_fresh.dtype, device=s_fresh.device)

            # Blend with flat spectrum
            flat = torch.ones_like(s_fresh)
            shaped = torch.clamp(target_t, min=0.01)
            blended = flat + lam * (shaped - flat)
            blended = torch.clamp(blended, min=0.01)

            # Norm-match
            s_new = blended * (frob / torch.norm(blended).item())

            # Choose U and V
            U_use = U_fresh
            Vt_use = Vt_fresh

            if align_mode != "none" and align_strength > 0:
                if "Vt" in ext and ("V" in align_mode or align_mode == "UV"):
                    Vt_pre = torch.tensor(ext["Vt"], dtype=W.dtype, device=W.device)
                    # Handle dimension mismatch (cross-model transfer)
                    if Vt_pre.shape == Vt_fresh.shape:
                        Vt_use = _blend_orthogonal(Vt_fresh, Vt_pre, align_strength)

                if "U" in ext and ("U" in align_mode or align_mode == "UV"):
                    U_pre = torch.tensor(ext["U"], dtype=W.dtype, device=W.device)
                    if U_pre.shape == U_fresh.shape:
                        U_use = _blend_orthogonal(U_fresh, U_pre, align_strength)

            W_new = U_use @ torch.diag(s_new) @ Vt_use
            param.data = W_new.reshape(orig_shape)
            n_shaped += 1

            if verbose:
                print(f"  {name:50s} {str(list(orig_shape)):20s} "
                      f"align={align_mode} noise={noise_std:.2f}")

    if verbose:
        print(f"\nShaped: {n_shaped} matrices, Skipped: {n_skipped} params")
    return n_shaped


def _blend_orthogonal(A, B, alpha):
    """Blend two orthogonal matrices: (1-alpha)*A + alpha*B, re-orthogonalized.

    This is a cheap approximation to geodesic interpolation on the Stiefel
    manifold. For alpha=0 returns A, alpha=1 returns B. In between, the
    result is the closest orthogonal matrix to the linear blend.
    """
    blended = (1 - alpha) * A + alpha * B
    # Re-orthogonalize via SVD (Procrustes)
    U, _, Vt = torch.linalg.svd(blended, full_matrices=False)
    return U @ Vt


def make_per_layer_init_fn(extracted, lam=1.0, align_mode="none",
                           align_strength=0.0, noise_std=0.0):
    """Factory: return an init_fn(model) for per-layer spectral init."""
    def init_fn(model):
        apply_per_layer_spectral(
            model, extracted, lam=lam, align_mode=align_mode,
            align_strength=align_strength, noise_std=noise_std,
            verbose=True,
        )
    return init_fn
