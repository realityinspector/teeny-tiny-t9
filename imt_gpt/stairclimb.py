#!/usr/bin/env python3
"""
stairclimb.py — Autoresearch loop for spectral init.

Systematically test hypotheses to improve on the 2.3x baseline.
Each iteration: form hypothesis, run experiment, keep if better.

Metric: PPL at step 750 (most reliable evaluation point).
Baseline: extracted_s42 PPL@750 = 818 (2.33x vs orthogonal 1,904).

Usage:
    python -m imt_gpt.stairclimb              # run next untested hypothesis
    python -m imt_gpt.stairclimb --list       # show all hypotheses + results
    python -m imt_gpt.stairclimb --run NAME   # run specific hypothesis
"""
import json
import sys
import os
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, "imt_gpt", ".venv", "bin", "python")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "imt_gpt", "results")
LEDGER = os.path.join(RESULTS_DIR, "stairclimb_ledger.json")

# Baseline to beat
BASELINE_PPL_750 = 818.0  # extracted_s42 step 750
BASELINE_NAME = "extracted_s42"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def load_ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            return json.load(f)
    return {"baseline": BASELINE_PPL_750, "best": BASELINE_PPL_750, "experiments": []}


def save_ledger(ledger):
    with open(LEDGER, "w") as f:
        json.dump(ledger, f, indent=2)


def run_experiment(name, script_body, timeout=10800):
    """Run a training experiment as a subprocess. Returns result dict or None."""
    print(f"\n{'='*60}")
    print(f"  STAIRCLIMB: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, "-c", script_body],
            capture_output=False, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        print(f"  [{name}] exit={result.returncode} wall={elapsed:.0f}s")

        # Read back the result file
        result_path = os.path.join(RESULTS_DIR, f"sc_{name}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                return json.load(f)
        else:
            print(f"  [{name}] NO RESULT FILE")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [{name}] TIMEOUT after {timeout}s")
        return None


def make_script(name, init_code, config_overrides=""):
    """Generate a training script with custom init and config."""
    # Build config kwargs, letting overrides replace defaults
    defaults = {
        "max_steps": 1000,
        "eval_steps": [250, 500, 750, 1000],
        "warmup_steps": 100,
        "log_every": 100,
        "lr": 6.25e-5,
        "seed": 42,
    }
    # Parse overrides like "warmup_steps=200, lr=1.25e-4,"
    if config_overrides.strip():
        for part in config_overrides.split(","):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                try:
                    defaults[key] = eval(val.strip())
                except Exception:
                    defaults[key] = val.strip()
    config_lines = ",\n    ".join(f"{k}={repr(v)}" for k, v in defaults.items())
    return f'''
import json, sys, os, numpy as np, math
sys.path.insert(0, "{PROJECT_ROOT}")
os.chdir("{PROJECT_ROOT}")

from imt_gpt.config import TrainConfig, get_device, install_signal_handlers, set_process_memory_limit
from imt_gpt.train import train, _clear_memory

install_signal_handlers()
set_process_memory_limit(max_gb=10.0)
device = get_device()

# Load extracted spectra
with open("imt_gpt/results/extracted_spectra.json") as f:
    extracted = json.load(f)

config = TrainConfig(
    {config_lines},
    device=device,
)

{init_code}

result = train(config, init_fn=init_fn, init_name="{name}", verbose=True)

# Save
path = "imt_gpt/results/sc_{name}.json"
with open(path, "w") as f:
    json.dump({{
        "name": "{name}",
        "final_ppl": result["final_ppl"],
        "elapsed": result["elapsed"],
        "checkpoints": {{str(k): v for k, v in result["checkpoints"].items()}},
        "grad_norm_samples": result["grad_norms"][::max(1, len(result["grad_norms"]) // 30)],
        "seed": 42,
    }}, f, indent=2)

ppl_750 = result["checkpoints"].get(750, "N/A")
print(f"\\n=== RESULT: {{path}} PPL@750={{ppl_750}} PPL@1000={{result['final_ppl']:.1f}} ===")
'''


# ============================================================
# HYPOTHESIS REGISTRY
# Each hypothesis: (name, description, init_code, config_overrides)
# ============================================================
HYPOTHESES = [
    # --- Spike mitigation ---
    (
        "spike_skip_50x",
        "Skip optimizer steps where gnorm > 50x running median. Should prevent late spikes that destroyed s137.",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'spike_skip_mult=50.0,',
    ),
    (
        "spike_skip_10x",
        "More aggressive spike skip: 10x threshold.",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'spike_skip_mult=10.0,',
    ),

    # --- Lambda blending ---
    (
        "lam_0.5",
        "Half-strength spectral shape. Less anisotropy = less sharp basin = more stable?",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"], lam=0.5)
''',
        '',
    ),
    (
        "lam_2.0",
        "Double-strength spectral shape. More anisotropy = sharper basin but faster convergence?",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"], lam=2.0)
''',
        '',
    ),

    # --- Warmup tuning ---
    (
        "warmup_200",
        "Longer warmup (200 vs 100). Slower LR ramp = fewer early spikes?",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'warmup_steps=200,',
    ),
    (
        "warmup_50",
        "Shorter warmup (50 vs 100). Faster to peak LR = exploit spectral head start sooner?",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'warmup_steps=50,',
    ),

    # --- Grad clip tuning ---
    (
        "clip_0.5",
        "Tighter gradient clipping (0.5 vs 1.0). Gentler updates = less spike damage?",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'grad_clip=0.5,',
    ),
    (
        "clip_0.25",
        "Very tight gradient clipping (0.25). Prevents nearly all spike damage.",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'grad_clip=0.25,',
    ),

    # --- LR ---
    (
        "lr_2x",
        "Double LR (1.25e-4). We know ortho diverges at 3x, but spectral might handle it.",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'lr=1.25e-4,',
    ),

    # --- Skip embedding spectral init ---
    (
        "no_embed_spectral",
        "Spectral init for attn/FFN only, orthogonal for embeddings. Embedding reconstruction is worst (r=0.82).",
        '''
import torch.nn as nn
from imt_gpt.spectral_init import apply_spectral_init, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED, classify_param

def init_fn(model):
    # Spectral for non-embedding groups
    coeffs_no_embed = {k: v for k, v in extracted["spectra_coeffs"].items() if k != "embedding"}
    apply_spectral_init(model, coeffs_no_embed, lam=1.0)
    # Orthogonal for embeddings
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2 and classify_param(name) == EMBED:
                nn.init.orthogonal_(param)

import torch
''',
        '',
    ),

    # --- Combined: spike_skip + clip 0.5 ---
    (
        "spike_skip_50x_clip_0.5",
        "Both spike skip AND tighter clip. Belt + suspenders for stability.",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Combined: spike_skip + warmup 200 ---
    (
        "spike_skip_50x_warmup_200",
        "Spike skip + longer warmup. Two stability interventions.",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'spike_skip_mult=50.0, warmup_steps=200,',
    ),

    # =============================================================
    # ROUND 2: Per-layer spectra + directional alignment + noise
    # =============================================================

    # --- Per-layer spectra (not group-averaged) ---
    (
        "per_layer",
        "Per-layer spectra from pretrained GPT-2 small. Each layer gets its own spectrum instead of group average.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2")
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0)
''',
        '',
    ),
    (
        "per_layer_stable",
        "Per-layer spectra + winning stability config (spike_skip + clip 0.5).",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2")
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Directional alignment ---
    (
        "align_V_0.25",
        "Per-layer spectra + 25% alignment of right singular vectors with pretrained V.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2", include_directions=True)
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, align_mode="V", align_strength=0.25)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "align_V_0.5",
        "Per-layer spectra + 50% V alignment. Half pretrained directions, half random.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2", include_directions=True)
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, align_mode="V", align_strength=0.5)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "align_V_1.0",
        "Per-layer spectra + full V alignment. Pretrained right singular vectors, fresh left.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2", include_directions=True)
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, align_mode="V", align_strength=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "align_UV_0.5",
        "Per-layer spectra + 50% alignment of BOTH U and V. Maximum pretrained structure transfer.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2", include_directions=True)
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, align_mode="UV", align_strength=0.5)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "align_UV_1.0",
        "Per-layer spectra + full UV alignment. Essentially loading pretrained weights with random scaling.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2", include_directions=True)
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, align_mode="UV", align_strength=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Hybrid: group-avg spectra + pretrained directions ---
    # (Per-layer spectra were too noisy. Use the stable DCT spectra
    # but add directional information from pretrained U/V.)
    (
        "hybrid_V_0.25",
        "Group-avg DCT spectra (stable) + 25% pretrained V alignment. Best of both worlds?",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="V", align_strength=0.25)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "hybrid_V_0.5",
        "Group-avg DCT spectra + 50% pretrained V alignment.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="V", align_strength=0.5)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "hybrid_V_1.0",
        "Group-avg DCT spectra + full pretrained V alignment.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="V", align_strength=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "hybrid_UV_0.5",
        "Group-avg DCT spectra + 50% pretrained UV alignment. Maximum directional transfer.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="UV", align_strength=0.5)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Combine winners: lr_2x + UV alignment + stability ---
    (
        "lr_2x_UV_stable",
        "2x LR + 50% UV alignment + spike_skip + clip_0.5. Combining the two best axes.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="UV", align_strength=0.5)
''',
        'lr=1.25e-4, spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "lr_2x_UV",
        "2x LR + 50% UV alignment, no stability interventions. Raw speed test.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="UV", align_strength=0.5)
''',
        'lr=1.25e-4,',
    ),
    (
        "lr_3x",
        "3x LR (1.875e-4). Orthogonal diverges here. Can spectral init handle it?",
        '''
from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=extracted["spectra_coeffs"])
''',
        'lr=1.875e-4, spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "lr_3x_UV_stable",
        "3x LR + UV alignment + stability. Push both axes simultaneously.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_hybrid_init_fn
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_hybrid_init_fn(extracted["spectra_coeffs"], dirs, lam=1.0, align_mode="UV", align_strength=0.5)
''',
        'lr=1.875e-4, spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # =============================================================
    # ROUND 3: Unfolded spectral init (24 meta-params → 50 spectra)
    # =============================================================
    (
        "unfold_seed",
        "24 meta-params fitted to pretrained GPT-2, unfolded into layer-specific spectra. No alignment.",
        '''
from imt_gpt.spectral_unfold import seed_from_pretrained, make_unfolded_init_fn
meta = seed_from_pretrained("gpt2")
init_fn = make_unfolded_init_fn(meta, lam=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "unfold_aligned",
        "24 meta-params + pretrained V directions. The full unfolding: seed → spectra + alignment → weights.",
        '''
from imt_gpt.spectral_unfold import seed_from_pretrained, make_unfolded_init_fn
from imt_gpt.pretrained_extract import extract_per_layer
meta = seed_from_pretrained("gpt2")
dirs = extract_per_layer("gpt2", include_directions=True)
init_fn = make_unfolded_init_fn(meta, pretrained_directions=dirs, lam=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Noise ablations (how robust is the signal?) ---
    (
        "noise_0.1",
        "Add N(0,0.1) noise to per-layer spectra. Tests robustness of spectral signal.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2")
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, noise_std=0.1)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "noise_0.3",
        "Add N(0,0.3) noise. Heavy corruption — if this still works, the signal is coarse.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2")
init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0, noise_std=0.3)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),
    (
        "shuffled_layers",
        "Per-layer spectra but randomly shuffled across layers. Tests whether layer-specific information matters.",
        '''
import random as _rng
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn
extracted_layers = extract_per_layer("gpt2")

# Shuffle spectra across layers within each group
_rng.seed(42)
by_group = {}
for name, ext in extracted_layers.items():
    g = ext["group"]
    by_group.setdefault(g, []).append(name)
for g, names in by_group.items():
    svs_list = [extracted_layers[n]["svs"] for n in names]
    _rng.shuffle(svs_list)
    for n, svs in zip(names, svs_list):
        extracted_layers[n]["svs"] = svs

init_fn = make_per_layer_init_fn(extracted_layers, lam=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Medium model extraction ---
    (
        "medium_per_layer",
        "Per-layer spectra from GPT-2 medium (355M), applied to small. Cross-scale transfer.",
        '''
from imt_gpt.pretrained_extract import extract_per_layer, make_per_layer_init_fn

# Extract from medium — layers won't match 1:1 (24 vs 12 layers),
# so we map by relative depth: medium layer i maps to small layer i//2
extracted_medium = extract_per_layer("gpt2-medium")

# Build mapping: for each small model param name, find best medium match
# Medium has 24 layers, small has 12. Map medium layer 2i and 2i+1 -> small layer i
import re
mapped = {}
for name_med, ext in extracted_medium.items():
    # Try to map to small model name
    m = re.match("transformer\\.h\\.(\\d+)\\.(.*)", name_med)
    if m:
        layer_med = int(m.group(1))
        rest = m.group(2)
        layer_small = layer_med // 2
        name_small = f"transformer.h.{layer_small}.{rest}"
        # Keep the one from the deeper medium layer (more processed)
        if name_small not in mapped or layer_med % 2 == 1:
            mapped[name_small] = ext
    elif "wte" in name_med or "wpe" in name_med or "ln_f" in name_med:
        # Non-layer params map directly
        mapped[name_med] = ext

init_fn = make_per_layer_init_fn(mapped, lam=1.0)
''',
        'spike_skip_mult=50.0, grad_clip=0.5,',
    ),

    # --- Extract from GPT-2 medium (group-averaged, original approach) ---
    (
        "medium_spectra",
        "Extract spectra from GPT-2 medium (355M) instead of small (124M). Richer task structure?",
        '''
import torch, numpy as np
from transformers import GPT2LMHeadModel
from imt_gpt.spectral_init import classify_param, ATTN_PROJ, FFN_UP, FFN_DOWN, EMBED, apply_spectral_init

# Extract from medium
print("  Extracting spectra from GPT-2 medium...")
medium = GPT2LMHeadModel.from_pretrained("gpt2-medium")
group_svs = {ATTN_PROJ: [], FFN_UP: [], FFN_DOWN: [], EMBED: []}
with torch.no_grad():
    for name, param in medium.named_parameters():
        if param.dim() < 2:
            continue
        group = classify_param(name)
        if group and group in group_svs:
            W = param.data.float()
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            s = torch.linalg.svdvals(W)
            s_norm = s / s.max() if s.max() > 0 else s
            group_svs[group].append(s_norm.cpu().numpy())
del medium
import gc; gc.collect()

# Compress to 8 DCT coefficients (same method)
medium_coeffs = {}
for group, sv_list in group_svs.items():
    if not sv_list:
        continue
    max_len = max(len(s) for s in sv_list)
    interp = [np.interp(np.linspace(0,1,max_len), np.linspace(0,1,len(s)), s) for s in sv_list]
    avg = np.mean(interp, axis=0)
    clipped = np.clip(avg, 0.01, None)
    target = np.log(np.exp(clipped) - 1.0 + 1e-10)
    n = len(avg)
    t = np.linspace(0, np.pi, n, endpoint=False)
    basis = np.zeros((n, 8))
    for i in range(8):
        basis[:, i] = np.cos((i + 0.5) * t)
    coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    medium_coeffs[group] = coeffs.tolist()
    print(f"    {group}: {len(sv_list)} matrices")

from imt_gpt.baselines import make_init_fn
init_fn = make_init_fn("imt_shaped", spectra_coeffs=medium_coeffs)
''',
        '',
    ),
]


def list_hypotheses(ledger):
    print(f"\nBaseline: {BASELINE_NAME} PPL@750 = {BASELINE_PPL_750}")
    print(f"Current best: PPL@750 = {ledger['best']}")
    print(f"\n{'#':>3}  {'Name':>30s}  {'PPL@750':>8s}  {'vs base':>8s}  {'Status'}")
    print("-" * 75)

    done = {e["name"]: e for e in ledger["experiments"]}
    for i, (name, desc, _, _) in enumerate(HYPOTHESES):
        if name in done:
            e = done[name]
            ppl = e.get("ppl_750", "?")
            if isinstance(ppl, (int, float)):
                ratio = f"{BASELINE_PPL_750 / ppl:.2f}x" if ppl > 0 else "N/A"
                better = "BETTER" if ppl < ledger["best"] else ""
                print(f"{i:>3}  {name:>30s}  {ppl:>8.1f}  {ratio:>8s}  done {better}")
            else:
                print(f"{i:>3}  {name:>30s}  {'?':>8s}  {'?':>8s}  error")
        else:
            print(f"{i:>3}  {name:>30s}  {'—':>8s}  {'—':>8s}  pending")
    print()


def run_hypothesis(name, ledger):
    """Run a single hypothesis and update the ledger."""
    hyp = None
    for h in HYPOTHESES:
        if h[0] == name:
            hyp = h
            break
    if hyp is None:
        print(f"Unknown hypothesis: {name}")
        return

    name, desc, init_code, config_overrides = hyp

    # Check if already done
    for e in ledger["experiments"]:
        if e["name"] == name:
            print(f"Already done: {name} (PPL@750={e.get('ppl_750', '?')})")
            return

    print(f"\nHypothesis: {desc}")
    script = make_script(name, init_code, config_overrides)
    result = run_experiment(name, script)

    entry = {
        "name": name,
        "description": desc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if result:
        ppl_750 = result.get("checkpoints", {}).get("750", None)
        ppl_1000 = result.get("final_ppl", None)
        entry["ppl_750"] = ppl_750
        entry["ppl_1000"] = ppl_1000
        entry["elapsed"] = result.get("elapsed", None)

        if ppl_750 and ppl_750 < ledger["best"]:
            print(f"\n  *** NEW BEST: {ppl_750:.1f} (was {ledger['best']:.1f}) ***")
            ledger["best"] = ppl_750
        elif ppl_750:
            print(f"\n  Result: {ppl_750:.1f} (best is {ledger['best']:.1f})")
    else:
        entry["error"] = "no result"

    ledger["experiments"].append(entry)
    save_ledger(ledger)
    return entry


def main():
    os.chdir(PROJECT_ROOT)
    ledger = load_ledger()

    if "--list" in sys.argv:
        list_hypotheses(ledger)
        return

    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        if idx + 1 < len(sys.argv):
            name = sys.argv[idx + 1]
            run_hypothesis(name, ledger)
        else:
            print("Usage: --run NAME")
        return

    if "--all" in sys.argv:
        # Run all pending hypotheses
        done_names = {e["name"] for e in ledger["experiments"]}
        for name, desc, _, _ in HYPOTHESES:
            if name not in done_names:
                run_hypothesis(name, ledger)
                ledger = load_ledger()  # reload after each
        list_hypotheses(ledger)
        return

    # Default: run next pending
    done_names = {e["name"] for e in ledger["experiments"]}
    for name, desc, _, _ in HYPOTHESES:
        if name not in done_names:
            run_hypothesis(name, ledger)
            return

    print("All hypotheses tested!")
    list_hypotheses(ledger)


if __name__ == "__main__":
    main()
