# teeny-tiny-t9

## Spectral Initialization via Pretrained Extraction (SIPE)

**3.33x faster convergence on GPT-2 training from scratch** by transferring
just 32 numbers — the compressed spectral fingerprint of a trained model's
weight structure — to a fresh random initialization.

```
W_init = U_fresh · diag(S_extracted) · blend(V_fresh, V_pretrained, 0.5)^T
```

Extract SVD spectra from a pretrained GPT-2, compress to 8 DCT coefficients
per weight group, reshape a fresh model's singular values to match, and
partially align the singular vectors. Train at 2x the standard learning rate.

| Config | PPL@750 | vs Orthogonal |
|--------|---------|---------------|
| Orthogonal baseline | 1,904 | 1.0x |
| Spectral shape only | 818 | 2.33x |
| + UV alignment + 2x LR | **572** | **3.33x** |

See [APRIL-3-FINDINGS-VETTED.md](APRIL-3-FINDINGS-VETTED.md) for the full
results (27 experiments, mechanism analysis, literature review).

### Quick Start

```bash
cd teeny-tiny-t9
python -m venv imt_gpt/.venv && source imt_gpt/.venv/bin/activate
pip install torch transformers datasets matplotlib numpy scipy

# Run the best config
python -m imt_gpt.stairclimb --run lr_2x_UV

# Run all 27 experiments (~14 hours on M1)
python -m imt_gpt.stairclimb --all

# View results
python -m imt_gpt.stairclimb --list
```

### Visualization

Open [web/spectral_init_viz.html](web/spectral_init_viz.html) in a browser
for an interactive D3.js visualization of the method and results.

### Key Files

```
imt_gpt/
  spectral_init.py       SVD extraction + DCT compression + spectral init
  pretrained_extract.py  Per-layer extraction + directional alignment
  baselines.py           Standard/Xavier/orthogonal/flat init methods
  train.py               Training loop with spike-skip
  stairclimb.py          Autoresearch harness (27 hypotheses)
  config.py              Training config + memory safety
  results/               All experimental results (JSON)
```

### Documentation

- [APRIL-3-FINDINGS-VETTED.md](APRIL-3-FINDINGS-VETTED.md) — Vetted findings
  with literature grounding
- [APRIL-3-FINDINGS.md](APRIL-3-FINDINGS.md) — Complete experimental record
- [RESEARCH_PRECEDENT.md](RESEARCH_PRECEDENT.md) — Claims for research
  validation

---

## The T9 Origin

This repo started as a T9 predictive text engine compressed to **1,381 bytes**
— 617 English words, full keyboard mapping, and a working CLI in one Python
script. The neural network research that led to SIPE grew from studying
spectral structure in the T9 task. The compressed engine is still at `main.py`.
See [archive/docs/WHITEPAPER.md](archive/docs/WHITEPAPER.md) for the
compression research.

## License

MIT
