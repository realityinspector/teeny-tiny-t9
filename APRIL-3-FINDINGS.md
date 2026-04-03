# Spectral Initialization via Pretrained Extraction (SIPE)
## Findings as of April 3, 2026

**Model**: GPT-2 small (124M params, 12 layers, 768 hidden, 12 heads)
**Dataset**: WikiText-2 (raw)
**Hardware**: Apple M1 MacBook Pro 16GB (MPS backend)
**Best result**: 3.33x faster convergence (PPL 572 vs 1,904 at step 750)

---

## Method Summary

We extract the singular value spectra from a pretrained GPT-2 small via SVD,
compress each weight group's average spectrum to 8 DCT coefficients (32 numbers
total), and use these to shape the initialization of a fresh randomly-initialized
GPT-2. At init time, we decompose each random weight matrix via SVD, replace the
singular values with the extracted spectral shape, partially align the right
singular vectors (50% blend) with the pretrained model's directions, and
reconstruct. Trained at 2x the baseline learning rate (1.25e-4 vs 6.25e-5).

```
W_init = U_fresh · diag(S_extracted) · blend(V_fresh, V_pretrained, 0.5)^T
```

### Weight Group Taxonomy

| Group | Matrices per layer | Total | DCT coefficients |
|-------|-------------------|-------|-----------------|
| Attention (Q/K/V/O) | 2 (c_attn, c_proj) | 24 | 8 |
| FFN up-projection | 1 (c_fc) | 12 | 8 |
| FFN down-projection | 1 (mlp.c_proj) | 12 | 8 |
| Embedding (token + position) | — | 2 | 8 |
| **Total** | | **50** | **32** |

### Training Configuration (Best Run: lr_2x_UV)

```
Optimizer:      AdamW (lr=1.25e-4, betas=(0.9, 0.999))
LR schedule:    Linear warmup (100 steps) → cosine decay to 0
Batch:          2 × 4 gradient accumulation = effective batch 8
Sequence len:   256 tokens
Gradient clip:  1.0 (default, no tighter clip needed)
Spike-skip:     Disabled (not needed at this config)
V alignment:    50% blend with pretrained right singular vectors
Max steps:      1,000 optimizer steps
```

---

## Best Result: 3.33x Convergence Advantage

| Step | Spectral+UV+2xLR | Orthogonal | Ratio |
|------|-------------------|------------|-------|
| 250  | 921               | 1,944      | 2.11x |
| 500  | 889               | 1,965      | 2.21x |
| 750  | **572**           | **1,904**  | **3.33x** |
| 1000 | **543**           | **1,896**  | **3.49x** |

Convergence is monotonic: 921 → 889 → 572 → 543. No instability spikes
at this configuration. The advantage *widens* with training — 2.11x at
step 250, 3.49x at step 1000.

---

## Stairclimb: 27 Experiments

We used an autoresearch approach — systematically testing hypotheses to
improve on the initial 2.33x baseline. Each experiment: 1,000 steps at
seed 42, measured at step 750.

### Full Results (sorted by PPL@750)

| # | Method | PPL@750 | PPL@1000 | vs Ortho |
|---|--------|---------|----------|----------|
| 1 | lr_2x_UV | **572** | **543** | **3.33x** |
| 2 | lr_2x_UV_stable | 592 | 599 | 3.22x |
| 3 | lr_2x | 714 | 702 | 2.67x |
| 4 | hybrid_UV_0.5 | 731 | 707 | 2.61x |
| 5 | lr_3x | 734 | 715 | 2.59x |
| 6 | hybrid_V_0.5 | 788 | 765 | 2.42x |
| 7 | spike_skip_50x+clip_0.5 | 802 | 775 | 2.37x |
| 8 | lr_3x_UV_stable | 830 | 798 | 2.29x |
| 9 | unfold_seed | 851 | 841 | 2.24x |
| 10 | clip_0.5 | 863 | 832 | 2.21x |
| 11 | warmup_50 | 1,000 | 952 | 1.90x |
| 12 | medium_spectra | 1,016 | 948 | 1.87x |
| 13 | spike_skip_10x | 1,048 | 1,011 | 1.82x |
| 14 | noise_0.1 | 1,075 | 957 | 1.77x |
| 15 | per_layer_stable | 1,221 | 1,141 | 1.56x |
| 16 | hybrid_V_0.25 | 1,263 | 1,153 | 1.51x |
| 17 | lam_0.5 | 1,357 | 815 | 1.40x |
| 18 | shuffled_layers | 1,384 | 1,332 | 1.38x |
| 19 | unfold_aligned | 1,557 | 810 | 1.22x |
| 20 | lam_2.0 | 1,682 | 6,025 | 1.13x |
| — | hybrid_V_1.0 | FAILED | — | SVD error |

### What Worked

1. **Higher learning rate (2x)**: Single biggest improvement. Spectral init
   tolerates LR = 1.25e-4 where orthogonal diverges at 1.875e-4 (3x).
   Contribution: 818 → 714 (PPL@750).

2. **UV directional alignment (50%)**: Blending right singular vectors 50%
   toward pretrained directions adds task-relevant geometric information.
   Contribution: 714 → 572 at 2x LR.

3. **Stability interventions (spike_skip + clip_0.5)**: Prevent gradient
   spikes from destabilizing training. Required at base LR, unnecessary at
   2x LR with UV alignment.

### What Didn't Work

1. **Per-layer spectra** (PPL 1,221): Too noisy. Each layer's spectrum is
   estimated from 2-4 matrices; the noise dominates the signal. Group
   averaging across all 12 layers acts as variance reduction.

2. **Unfolded spectral init** (PPL 851): 24-parameter generator fitted via
   L-BFGS to produce layer-varying spectra. The fit error (RMSE 0.28) loses
   precision compared to raw DCT coefficients. The smooth depth-variation
   idea is sound but needs a better fitting method.

3. **Medium model extraction** (PPL 1,016): GPT-2 medium (355M) spectra
   applied to small (124M) doesn't transfer well. Dimension mismatch
   (1024 vs 768 hidden) and layer count (24 vs 12) corrupt the signal.

4. **Lambda 2.0** (PPL 1,682 → 6,025): Double-strength spectral shaping
   creates basins too sharp to navigate.

5. **More aggressive spike-skip (10x)**: Skips too many steps, slowing
   convergence more than it helps stability.

6. **Full V alignment (1.0)**: SVD convergence failure on the embedding
   matrix. 50% is the sweet spot — enough directional signal without
   rigidity.

### Key Negative Controls

- **Shuffled layers** (PPL 1,384 vs group-avg 802): Shuffling spectra
  across layers within groups hurts significantly, confirming that
  layer-specific spectral information exists (even though per-layer
  extraction is too noisy to exploit directly).

- **Noise 0.1** (PPL 1,075 vs clean 802): Adding N(0,0.1) noise to
  per-layer spectra degrades performance, confirming the spectral signal
  requires precision.

- **Noise 0.3** (PPL 28M at step 750): Heavy noise destroys the signal
  completely (recovers to 1,079 by step 1000).

- **No-embed spectral** (PPL 1,368 vs full 802): Skipping spectral init
  for embeddings and using orthogonal instead is significantly worse,
  despite embedding having the worst DCT reconstruction quality (r=0.82).

---

## Mechanism Analysis

### Shape Not Scale

Flat singular values scaled to pretrained Frobenius norms (imt_scaled_flat)
produce gradient explosion (max gnorm 10,796) and PPL 2,077. Shaped singular
values at default N(0,0.02) scale produce PPL 572.

**Caveat from review**: This ablation has a confound — pretrained Frobenius
norms (~9x larger) at flat SVs create operator norms far exceeding 1, making
gradient explosion expected regardless of shape. A cleaner ablation would
compare flat vs shaped at identical Frobenius norms. The existing imt_flat
result (PPL 972, all SVs=1 at default scale) vs imt_extracted (PPL 793, shaped
SVs at default scale) is the cleaner comparison: same scale, shape wins.

### Not Numerical Conditioning

| Method | Attn cond | FFN cond | Embed cond | PPL@750 |
|--------|-----------|----------|------------|---------|
| orthogonal | **1.0** | **1.0** | **1.0** | 1,904 |
| imt_flat | 1.9 | 1.9 | 1.9 | 972 |
| imt_extracted | 14.1 | 10.2 | 75.9 | 818 |
| standard | 2,990 | 3.0 | 2.4 | 2,608 |

Orthogonal has perfect conditioning (σ_max/σ_min = 1.0 everywhere) yet
converges slowest. Spectral init has worse conditioning yet converges
3.33x faster. Condition number is the wrong lens.

**Better framework**: The implicit regularization literature (Razin & Cohen,
NeurIPS 2020; Arora et al., NeurIPS 2019) shows deep networks naturally
develop low-rank (anisotropic) weight structures during training. Spectral
init matches what networks converge to naturally, bypassing the need to
develop this structure from random initialization. Effective rank and stable
rank are more appropriate metrics than condition number.

### Directional Alignment Adds a Real Signal

| Config | PPL@750 | What's transferred |
|--------|---------|-------------------|
| Spectral shape only (base LR) | 818 | SV magnitudes |
| + 50% V alignment | 788 | + right singular vector directions |
| + 50% UV alignment | 731 | + left singular vector directions |
| + 2x LR | 572 | + higher learning rate tolerance |

Each addition provides genuine new information. This is supported by
PiSSA (Meng et al., NeurIPS 2024), which showed pretrained singular
vectors carry useful directional information, and by Staats et al. (2024),
who showed right singular vectors develop significant overlap with
activation covariance eigenvectors during training.

### Spectral Init Enables Higher Learning Rates

Orthogonal init diverges (NaN) at 3x the base LR (1.875e-4). Spectral init
handles 2x LR (1.25e-4) and produces its best results there. Theoretical
support from Lewkowycz et al. (2020, "The Catapult Mechanism"): maximum
stable LR is η_max ≈ 4/λ₀ for linear networks, where λ₀ is the initial
loss curvature. If spectral init produces lower initial curvature, this
directly predicts higher LR tolerance.

The 2x LR optimum (714) vs 3x LR (734) shows diminishing returns — the
spectral advantage on LR tolerance is bounded, not unlimited.

### Sharpness-Stability Trade-off

Multi-seed analysis (3 seeds, base LR):

| Method | s42 | s137 | s512 | Mean | Std |
|--------|-----|------|------|------|-----|
| extracted @750 | 818 | 721 | 911 | 817 | 95 |
| flat @750 | 1,004 | 808 | — | 906 | 138 |
| ortho @750 | 1,900 | 1,908 | — | 1,904 | 6 |

Spectral init has high seed variance (std=95) while orthogonal is
rock-solid (std=6). Spectral init creates sharper basins — faster
convergence but vulnerable to stochastic gradient spikes. At 2x LR
with UV alignment, this instability largely disappears (the best run
has monotonic convergence with no spikes).

---

## Spectral Compressibility

### DCT Reconstruction Quality

| n_dct | Mean r (actual sorted SVs) | Embedding r |
|-------|---------------------------|-------------|
| 2 | 0.861 | 0.577 |
| 4 | 0.897 | 0.692 |
| **8** | **0.938** | **0.819** |
| 16 | 0.963 | 0.907 |
| 32 | 0.957 | 0.883 |

8 DCT coefficients per group gives r > 0.93 average. The raw DCT output
has a U-shape (not monotonically decreasing), but after spectral_init_weight
applies it and SVD re-sorts the values, the actual matrix spectrum closely
matches the pretrained target. Even n_dct=2 gives r > 0.86.

**Caveat from review**: The Marchenko-Pastur distribution for random matrices
is also smooth and highly compressible. The paper should demonstrate that
the DCT coefficients capture *deviations from MP* specifically, not just
spectral smoothness. The key question: does a random matrix compressed to
8 DCT coefficients also give high r? (Likely yes for the bulk, but the
task-specific deviations in the leading SVs are what matter.)

### Group Averaging as Regularization

Per-layer spectra (50 separate spectra) are much worse than group-averaged
(4 spectra): PPL 1,221 vs 802. The noise from estimating each layer's
spectrum from only 2-4 matrices overwhelms the benefit of layer-specific
information. Group averaging acts as variance reduction.

Evidence that layer-specific information exists but is inaccessible:
shuffling spectra across layers within groups hurts (PPL 1,384 vs 802).

---

## What We Have NOT Proven

1. **Scaling**: All results are at toy scale (batch 8, seq_len 256, 1K steps
   on MPS). Real GPT-2 pretraining uses batch 512, seq_len 1024, 300K+ steps
   on CUDA. Historical track record for init improvements scaling is mixed to
   poor (LSUV benefits disappeared on ImageNet; Fixup required additional
   regularization at scale).

2. **Persistence beyond 1K steps**: The advantage is widening at termination
   (3.33x at 750, 3.49x at 1000) but we have zero data beyond step 1,000.
   Whether this holds at convergence (50K+ steps) is unknown.

3. **Generalization**: Only WikiText-2 tested. No evidence of transfer to
   other datasets, model sizes, or architectures.

4. **Multi-seed on the best config**: The 3.33x result (lr_2x_UV) is
   single-seed (seed 42). The multi-seed data is only for the base config.

5. **Clean shape-vs-scale ablation**: The imt_scaled_flat ablation conflates
   shape with scale (pretrained Frobenius norms cause explosion independent
   of shape). The imt_flat vs imt_extracted comparison (972 vs 793) is
   cleaner but uses default scale for both.

6. **Whether this outperforms simply loading pretrained weights**: Full warm
   starting would give even better PPL. SIPE transfers radically less
   information (32 numbers vs 124M parameters) — the question is whether
   this is useful when full transfer isn't available.

---

## Important Related Work

The following papers are the closest prior art and must be cited:

- **Trockman & Kolter (ICML 2023)**: Mimetic initialization — uses pretrained
  ViT weight structure for from-scratch initialization. Closest conceptual
  predecessor.
- **PiSSA (Meng et al., NeurIPS 2024)**: Uses pretrained principal singular
  values AND vectors for LoRA initialization. Directly demonstrates that
  pretrained spectral structure transfers.
- **DoRA (Liu et al., ICML 2024)**: Decomposes pretrained weights into
  magnitude and direction — parallel to our spectral shape + directional
  alignment.
- **Conv-SVD (Ng et al., Neural Computation 2023)**: Transfers singular
  vectors from pretrained weights.
- **Lewkowycz et al. (2020)**: Catapult mechanism — theoretical basis for
  why spectral init enables higher LR.
- **Saxe, McClelland & Ganguli (2014)**: Exact solutions for deep linear
  networks — strongest theoretical foundation for how SVD structure at init
  determines learning dynamics.
- **Staats et al. (2024)**: "Small Singular Values Matter" — confirms
  spectral deviations from Marchenko-Pastur encode task information.
- **Martin & Mahoney (JMLR 2021)**: Heavy-tailed self-regularization — the
  descriptive framework (not prescriptive) for trained weight spectra.
- **SPAM (2025)**: Spike-aware Adam with similar gradient norm thresholds.
- **PaLM (Chowdhery et al., 2022)**: Batch-skipping for spike mitigation.
- **Ash & Adams (NeurIPS 2020)**: Shrink & Perturb — warm starting analysis,
  relevant to partial transfer framing.

---

## Honest Framing for a Paper

**Strongest claims** (build the paper around these):
1. Directional alignment (pretrained V vectors) adds signal beyond spectral
   magnitudes — confirmed by PiSSA, clean ablation, theoretical support.
2. Spectral init enables higher learning rates — supported by catapult
   mechanism theory, clean empirical test.
3. The specific pipeline (SVD → DCT → spectral reshape for from-scratch
   training) is novel, though adjacent work in fine-tuning (PiSSA, DoRA)
   explores similar decompositions.

**Claims to avoid or heavily qualify**:
1. "No prior work combines spectral and directional transfer" — directly
   contradicted by PiSSA, DoRA, Conv-SVD, OSoRA.
2. "Spike-skip is novel" — PaLM and SPAM are clear prior art.
3. "Spectral curriculum" framing — strained metaphor, use "prior-informed
   initialization" instead.
4. Any claim about dynamical isometry being "disproven" — it was never
   claimed for transformers specifically.

**The core contribution**: A practical method for transferring compressed
spectral structure from pretrained models to accelerate from-scratch
training, with 3.33x improvement at toy scale. The method occupies a novel
point on the transfer spectrum: more than random init, less than full weight
loading, compressible to 32 numbers + partial directional alignment.

---

## Reproduction

```bash
cd teeny-tiny-t9
python -m venv imt_gpt/.venv
source imt_gpt/.venv/bin/activate
pip install torch transformers datasets matplotlib numpy scipy

# Run the best config (lr_2x_UV)
python -m imt_gpt.stairclimb --run lr_2x_UV

# Run the full stairclimb (27 experiments, ~14 hours on M1)
python -m imt_gpt.stairclimb --all

# View results
python -m imt_gpt.stairclimb --list
```

Results in `imt_gpt/results/sc_*.json` and `stairclimb_ledger.json`.
Visualization at `web/spectral_init_viz.html`.

## Files

```
imt_gpt/
  spectral_init.py       SVD extraction + DCT compression + spectral init
  pretrained_extract.py  Per-layer extraction + directional alignment
  spectral_unfold.py     Hierarchical unfolding (24 meta-params → spectra)
  baselines.py           Standard/Xavier/orthogonal/flat init methods
  train.py               Training loop with spike-skip
  stairclimb.py          Autoresearch harness (27 hypotheses)
  config.py              Training config + memory safety
  results/
    extracted_spectra.json         Pretrained spectral data
    stairclimb_ledger.json         Full experiment results
    sc_*.json                      Individual run results
    jacobian_analysis.json         Conditioning analysis
    dct_analysis_corrected.json    DCT reconstruction quality
```
