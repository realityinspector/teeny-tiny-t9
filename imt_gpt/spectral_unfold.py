"""
spectral_unfold.py — Hierarchical unfolding of spectral initialization.

Instead of fixed DCT coefficients per group, a small meta-parameter vector
unfolds through a generator function into layer-specific spectra and
alignment strengths. This is the "modular synthesizer" approach:

    meta_seed (tiny)
      → spectral generator (oscillator)
        → layer-specific spectra (waveforms)
          → V alignment strengths per layer (routing)
            → weight matrices (audio)
              → training dynamics (perception)

The generator is a smooth function of (depth, group_type), so it produces
layer-specific spectra without the noise of raw per-layer extraction.
It lives in the space between "4 group-averaged spectra" (too coarse)
and "50 per-layer spectra" (too noisy).

Meta-parameter layout:
    [0:8]   base DCT coefficients (the group-averaged starting point)
    [8:16]  depth modulation (how spectrum changes from layer 0 to 11)
    [16:20] group offsets (shift per group: attn, ffn_up, ffn_down, embed)
    [20:24] alignment profile (V align strength varies with depth)
    Total: 24 meta-parameters → 50 layer-specific spectra + alignments
"""
import numpy as np
import torch
import gc
from imt_gpt.spectral_init import (
    classify_param, dct_expand, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED
)

GROUP_INDEX = {ATTN_PROJ: 0, FFN_UP: 1, FFN_DOWN: 2, EMBED: 3}
N_LAYERS = 12


def decode_meta(meta, n_dct=8):
    """Decode a meta-parameter vector into generator components.

    Args:
        meta: array of 24 floats (the entire seed)
        n_dct: number of DCT coefficients per spectrum

    Returns:
        dict with base_coeffs, depth_mod, group_offsets, align_profile
    """
    meta = np.array(meta, dtype=np.float64)
    return {
        "base_coeffs": meta[0:n_dct],
        "depth_mod": meta[n_dct:2*n_dct],
        "group_offsets": meta[2*n_dct:2*n_dct+4],
        "align_profile": meta[2*n_dct+4:2*n_dct+8],
    }


def generate_spectrum(layer_idx, group, components, n_dct=8):
    """Generate DCT coefficients for a specific layer and group.

    The spectrum smoothly varies with depth (layer position) and has
    per-group offsets. This is the core unfolding: 24 numbers → 1 spectrum.

    Args:
        layer_idx: 0-11 for transformer layers, -1 for embeddings
        group: one of ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED
        components: output of decode_meta()

    Returns:
        array of n_dct DCT coefficients
    """
    base = components["base_coeffs"]
    depth_mod = components["depth_mod"]
    group_off = components["group_offsets"]

    # Normalized depth: 0.0 (layer 0) to 1.0 (layer 11)
    # Embeddings get depth = -0.1 (before all layers)
    if layer_idx < 0:
        depth = -0.1
    else:
        depth = layer_idx / max(N_LAYERS - 1, 1)

    # Linear depth modulation: spectrum smoothly changes with depth
    coeffs = base + depth * depth_mod

    # Per-group offset (scalar shift to all coefficients)
    g_idx = GROUP_INDEX.get(group, 0)
    coeffs = coeffs + group_off[g_idx]

    return coeffs


def generate_alignment(layer_idx, group, components):
    """Generate V alignment strength for a specific layer.

    The alignment profile controls how much pretrained direction
    information is injected, varying smoothly with depth.

    Returns: float in [0, 1] via sigmoid
    """
    profile = components["align_profile"]

    if layer_idx < 0:
        depth = -0.1
    else:
        depth = layer_idx / max(N_LAYERS - 1, 1)

    g_idx = GROUP_INDEX.get(group, 0)

    # Alignment = sigmoid(profile[0] + profile[1]*depth + profile[2]*group + profile[3]*depth*group)
    raw = profile[0] + profile[1] * depth + profile[2] * (g_idx / 3.0) + profile[3] * depth * (g_idx / 3.0)
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))


def _blend_orthogonal(A, B, alpha):
    """Blend two orthogonal matrices, re-orthogonalized via SVD."""
    blended = (1 - alpha) * A + alpha * B
    try:
        U, _, Vt = torch.linalg.svd(blended, full_matrices=False)
        return U @ Vt
    except torch._C._LinAlgError:
        return A  # fall back to random if SVD fails


def apply_unfolded_init(model, meta, pretrained_directions=None,
                        lam=1.0, verbose=False):
    """Apply unfolded spectral initialization.

    The meta vector unfolds into layer-specific spectra and alignment
    strengths, which then shape the weight matrices.

    Args:
        model: fresh GPT-2 model to initialize
        meta: array of 24 meta-parameters (the seed)
        pretrained_directions: output of extract_per_layer(include_directions=True)
            If None, no directional alignment (spectra only).
        lam: global spectral blending strength
        verbose: print per-layer info
    """
    components = decode_meta(meta)
    n_shaped = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue

            group = classify_param(name)
            if group is None:
                continue

            # Determine layer index
            import re
            m = re.match(r"transformer\.h\.(\d+)\.", name)
            layer_idx = int(m.group(1)) if m else -1

            # Generate layer-specific DCT coefficients
            coeffs = generate_spectrum(layer_idx, group, components)

            # Expand to full spectrum
            orig_shape = param.shape
            W = param.data.float()
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)

            U_fresh, s_fresh, Vt_fresh = torch.linalg.svd(W, full_matrices=False)
            frob = torch.norm(W, 'fro').item()
            n = len(s_fresh)

            target_spectrum = dct_expand(coeffs, n)
            target_t = torch.tensor(target_spectrum, dtype=s_fresh.dtype,
                                    device=s_fresh.device)

            flat = torch.ones_like(s_fresh)
            shaped = torch.clamp(target_t, min=0.01)
            blended = flat + lam * (shaped - flat)
            blended = torch.clamp(blended, min=0.01)
            s_new = blended * (frob / torch.norm(blended).item())

            # Directional alignment (strength varies by layer)
            U_use = U_fresh
            Vt_use = Vt_fresh

            if pretrained_directions and name in pretrained_directions:
                align_strength = generate_alignment(layer_idx, group, components)
                ext = pretrained_directions[name]

                if "Vt" in ext and align_strength > 0.01:
                    Vt_pre = torch.tensor(ext["Vt"], dtype=W.dtype, device=W.device)
                    if Vt_pre.shape == Vt_fresh.shape:
                        Vt_use = _blend_orthogonal(Vt_fresh, Vt_pre, align_strength)

            W_new = U_use @ torch.diag(s_new) @ Vt_use
            param.data = W_new.reshape(orig_shape)
            n_shaped += 1

            if verbose:
                align_str = f"{generate_alignment(layer_idx, group, components):.2f}" if pretrained_directions else "n/a"
                print(f"  L{layer_idx:>2d} {group:>12s} {str(list(orig_shape)):>20s} "
                      f"align={align_str}")

    if verbose:
        print(f"\nUnfolded init: {n_shaped} matrices from {len(meta)} meta-parameters")

    return n_shaped


def make_unfolded_init_fn(meta, pretrained_directions=None, lam=1.0):
    """Factory: return an init_fn(model) for unfolded spectral init."""
    def init_fn(model):
        apply_unfolded_init(model, meta, pretrained_directions=pretrained_directions,
                            lam=lam, verbose=True)
    return init_fn


def seed_from_pretrained(model_name="gpt2"):
    """Extract a meta-parameter seed from a pretrained model.

    Fits the 24-parameter generator to best approximate the pretrained
    model's actual per-layer spectra. This gives us a principled starting
    point for the meta-parameters — the unfolded version of what we already
    know works.

    Returns:
        array of 24 meta-parameters
    """
    from imt_gpt.pretrained_extract import extract_per_layer

    print(f"  Extracting per-layer spectra from {model_name}...")
    layers = extract_per_layer(model_name)

    # Collect per-layer DCT coefficients
    layer_data = []  # (layer_idx, group, coeffs)
    for name, ext in layers.items():
        group = ext["group"]
        svs = ext["svs"]

        import re
        m = re.match(r"transformer\.h\.(\d+)\.", name)
        layer_idx = int(m.group(1)) if m else -1

        # Fit DCT coefficients to this layer's spectrum
        n = len(svs)
        clipped = np.clip(svs, 0.01, None)
        target = np.log(np.exp(clipped) - 1.0 + 1e-10)

        t = np.linspace(0, np.pi, n, endpoint=False)
        basis = np.zeros((n, 8))
        for i in range(8):
            basis[:, i] = np.cos((i + 0.5) * t)
        coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)

        layer_data.append((layer_idx, group, coeffs))

    # Now fit the 24 meta-parameters to best reproduce all per-layer coefficients
    # via least squares on the generator function
    from scipy.optimize import minimize

    def loss(meta):
        components = decode_meta(meta)
        err = 0
        for layer_idx, group, true_coeffs in layer_data:
            pred_coeffs = generate_spectrum(layer_idx, group, components)
            err += np.sum((true_coeffs - pred_coeffs) ** 2)
        return err

    # Initialize from group averages
    group_coeffs = {g: [] for g in GROUP_INDEX}
    for _, group, coeffs in layer_data:
        group_coeffs[group].append(coeffs)

    # Base = overall average
    all_coeffs = [c for _, _, c in layer_data]
    base_init = np.mean(all_coeffs, axis=0)

    # Depth modulation = regression slope
    depth_init = np.zeros(8)
    for layer_idx, group, coeffs in layer_data:
        if layer_idx >= 0:
            depth = layer_idx / 11.0
            depth_init += (coeffs - base_init) * depth
    depth_init /= max(len([d for d in layer_data if d[0] >= 0]), 1)

    # Group offsets = mean deviation per group
    group_off_init = np.zeros(4)
    for g, idx in GROUP_INDEX.items():
        if group_coeffs[g]:
            avg = np.mean(group_coeffs[g], axis=0)
            group_off_init[idx] = np.mean(avg - base_init)

    # Alignment profile = start with uniform 0.5
    align_init = np.array([0.0, 0.0, 0.0, 0.0])  # sigmoid(0)=0.5

    meta_init = np.concatenate([base_init, depth_init, group_off_init, align_init])

    result = minimize(loss, meta_init, method='L-BFGS-B', options={'maxiter': 500})
    meta_opt = result.x

    # Report fit quality
    components = decode_meta(meta_opt)
    errors = []
    for layer_idx, group, true_coeffs in layer_data:
        pred = generate_spectrum(layer_idx, group, components)
        err = np.sqrt(np.mean((true_coeffs - pred) ** 2))
        errors.append(err)
    print(f"  Meta-fit: {len(meta_opt)} params, mean RMSE={np.mean(errors):.4f}, "
          f"max RMSE={np.max(errors):.4f}")

    return meta_opt


def search_meta(n_generations=30, population=12, fitness_steps=750):
    """CMA-ES search over meta-parameters.

    The search optimizes the 24 meta-parameters to minimize PPL at
    fitness_steps. This is the outer loop of the unfolding: searching
    the space of generator configurations.

    Note: This is expensive (each eval = one training run). Use
    seed_from_pretrained() for a good starting point, then search
    around it.
    """
    # Placeholder — implement when ready for the search phase
    raise NotImplementedError(
        "CMA-ES search over meta-parameters. Start from seed_from_pretrained() "
        "and search a neighborhood. Each eval = one training run at fitness_steps."
    )
