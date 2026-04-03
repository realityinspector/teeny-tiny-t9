# PRISM

**Prismic Pretraining Acceleration**

**3.33x faster convergence on GPT-2 training from scratch.**

A prism decomposes white light into its spectral components. PRISM
decomposes a pretrained model's weights into their SVD spectrum, then
refracts that spectral structure into a fresh initialization — transferring
the geometric fingerprint of learned representations without transferring
the weights themselves.

## Core Techniques

PRISM is a family of spectral transfer techniques for neural network
initialization:

- **Spectral Imprint** — Extract the singular value distribution from
  pretrained weights, compress to 8 DCT coefficients per group (32 numbers
  total), and reshape a fresh model's SVD spectrum to match. This transfers
  *what the model learned matters* without transferring the learned values.

- **EigenTransfer** — Partially align the fresh model's singular vectors
  (50% blend) with the pretrained model's directions. This transfers
  *which directions in weight space are task-relevant*, giving the optimizer
  a geometric head start.

Together, these enable training at 2x the standard learning rate
(orthogonal init diverges at this LR) and produce 3.33x faster
convergence on GPT-2 small (124M params, WikiText-2).

```
W_init = U_fresh · diag(S_imprint) · blend(V_fresh, V_pretrained, 0.5)^T
```

## Results

| Config | PPL@750 | vs Orthogonal |
|--------|---------|---------------|
| Orthogonal baseline | 1,904 | 1.0x |
| Spectral Imprint only | 818 | 2.33x |
| + EigenTransfer + 2x LR | **572** | **3.33x** |

Discovered through 27 systematic experiments (autoresearch stairclimb).
See [APRIL-3-FINDINGS-VETTED.md](APRIL-3-FINDINGS-VETTED.md) for the
full results, mechanism analysis, and literature review.

## Quick Start

```bash
cd teeny-tiny-t9
python -m venv imt_gpt/.venv && source imt_gpt/.venv/bin/activate
pip install torch transformers datasets matplotlib numpy scipy

# Run the best config (Spectral Imprint + EigenTransfer + 2x LR)
python -m imt_gpt.stairclimb --run lr_2x_UV

# Run all 27 experiments (~14 hours on M1)
python -m imt_gpt.stairclimb --all

# View results
python -m imt_gpt.stairclimb --list
```

## Visualization

Open [web/spectral_init_viz.html](web/spectral_init_viz.html) in a browser
for an interactive D3.js visualization of the method and results.

## Key Files

```
imt_gpt/
  spectral_init.py       Spectral Imprint: SVD extraction + DCT compression
  pretrained_extract.py  EigenTransfer: directional alignment
  baselines.py           Standard/Xavier/orthogonal/flat init methods
  train.py               Training loop with spike-skip
  stairclimb.py          Autoresearch harness (27 hypotheses)
  config.py              Training config + memory safety
  results/               All experimental results (JSON)
```

## Documentation

- [APRIL-3-FINDINGS-VETTED.md](APRIL-3-FINDINGS-VETTED.md) — Vetted findings
  with literature grounding
- [APRIL-3-FINDINGS.md](APRIL-3-FINDINGS.md) — Complete experimental record
- [RESEARCH_PRECEDENT.md](RESEARCH_PRECEDENT.md) — Claims for research
  validation

## Related Work

PRISM connects to several lines of research:

- **Spectral initialization theory**: Saxe et al. (2014) showed SVD structure
  at initialization determines learning dynamics in deep linear networks.
  PRISM operationalizes this by extracting the SVD structure from a trained
  model and transferring it.

- **Frequency-domain priors**: Tancik et al. (2020, Fourier Features) showed
  that frequency-domain initialization is critical for implicit neural
  representations (SIREN, NeRF). PRISM generalizes this: instead of
  hand-designed frequency priors, we extract the actual spectral prior from
  a trained model.

- **SVD-based transfer**: PiSSA (NeurIPS 2024), DoRA (ICML 2024), and
  Conv-SVD (2023) transfer SVD components for fine-tuning. PRISM applies
  spectral transfer to *from-scratch training* — a distinct setting where
  the full spectral shape (not just top-k components) matters.

- **Xavier/Kaiming limitations**: Standard initialization methods (Glorot
  2010, He 2015) achieve variance preservation but not spectral shaping.
  They treat all singular value directions equally. PRISM injects the
  task-relevant anisotropy that trained models naturally develop.

---

## The T9 Origin

This repo started as a T9 predictive text engine compressed to **1,381 bytes**
— 617 English words in one Python script. The spectral initialization research
that became PRISM grew from studying SVD structure in the T9 task's weight
matrices. The compressed engine is still at `main.py`.

## License

Apache 2.0. "PRISM" and "Prismic Pretraining Acceleration" are trademarks.
See [LICENSE](LICENSE).
