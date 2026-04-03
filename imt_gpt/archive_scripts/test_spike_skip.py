#!/usr/bin/env python3
"""Quick test: does spike-skip fix the s137 instability?

Runs extracted_s137 with spike_skip_mult=50 (skip steps where gnorm > 50x median).
Compare against the existing result (PPL=1530, two late spikes).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imt_gpt.config import TrainConfig, get_device, install_signal_handlers
from imt_gpt.train import train
from imt_gpt.baselines import make_init_fn

install_signal_handlers()
device = get_device()

# Load extracted spectra
spectra = json.load(open("imt_gpt/results/extracted_spectra.json"))

config = TrainConfig(
    max_steps=1000,
    eval_steps=[250, 500, 750, 1000],
    warmup_steps=100,
    log_every=100,
    lr=6.25e-5,
    seed=137,
    device=device,
    spike_skip_mult=50.0,  # Skip steps where gnorm > 50x running median
)

init_fn = make_init_fn("imt_shaped", spectra_coeffs=spectra["spectra_coeffs"])
result = train(config, init_fn=init_fn, init_name="extracted_s137_spikeskip", verbose=True)

print(f"\n=== RESULT ===")
print(f"Final PPL: {result['final_ppl']:.1f}")
print(f"Checkpoints: {result['checkpoints']}")

# Count spikes
gnorms = result["grad_norms"]
big = [g for g in gnorms if g > 100]
print(f"Gnorm spikes >100: {len(big)}")
print(f"Max gnorm: {max(gnorms):.1f}")

# Save
path = "imt_gpt/results/diag_extracted_s137_spikeskip.json"
with open(path, "w") as f:
    json.dump({
        "init_name": "extracted_s137_spikeskip",
        "final_ppl": result["final_ppl"],
        "checkpoints": {str(k): v for k, v in result["checkpoints"].items()},
        "grad_norm_samples": gnorms[::max(1, len(gnorms) // 30)],
        "loss_samples": result["losses"][::max(1, len(result["losses"]) // 30)],
        "seed": 137,
        "spike_skip_mult": 50.0,
    }, f, indent=2)
print(f"Saved {path}")

# Compare
print(f"\n=== COMPARISON ===")
orig = json.load(open("imt_gpt/results/diag_extracted_s137.json"))
print(f"Without spike-skip: PPL={orig['final_ppl']:.1f}")
print(f"With spike-skip:    PPL={result['final_ppl']:.1f}")
