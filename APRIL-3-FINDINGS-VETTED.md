# Spectral Initialization via Pretrained Extraction (SIPE)
## Vetted Findings — April 3, 2026

*Adjusted against external literature review. Claims are grounded in
established theory where possible, novelty claims are scoped precisely,
and known limitations are stated upfront.*

---

## What This Is

A method for transferring compressed SVD structure from a pretrained
transformer to a randomly-initialized one, achieving 3.33x faster early
convergence on GPT-2 small (124M params, WikiText-2, 1,000 steps).

The method sits at a specific point on the transfer spectrum:
- **Random init**: 0 bits of pretrained information → baseline
- **SIPE**: ~32 DCT coefficients + partial V alignment → 3.33x speedup
- **Full warm start**: 124M parameters → fastest, but known plasticity
  issues (Ash & Adams, NeurIPS 2020)

### Core Operation

```
W_init = U_fresh · diag(S_extracted) · blend(V_fresh, V_pretrained, 0.5)^T
```

For each weight matrix in the fresh model:
1. SVD decompose: W_random = U_r · S_r · V_r^T
2. Replace S_r with spectral shape from pretrained model (via DCT expansion)
3. Blend V_r 50% toward pretrained V directions (re-orthogonalized)
4. Keep U_r random
5. Reconstruct

---

## Results at a Glance

| Config | PPL@750 | PPL@1000 | vs Orthogonal |
|--------|---------|----------|---------------|
| Orthogonal baseline | 1,904 | 1,896 | 1.0x |
| Spectral shape only (base LR) | 818 | 793 | 2.33x |
| + stability (spike-skip + clip) | 802 | 775 | 2.37x |
| + V alignment (50%) | 788 | 765 | 2.42x |
| + UV alignment (50%) | 731 | 707 | 2.61x |
| + 2x LR (1.25e-4) | 714 | 702 | 2.67x |
| **UV alignment + 2x LR** | **572** | **543** | **3.33x** |

**Scale caveat**: Effective batch 8, seq_len 256, Apple M1 MPS, 1,000 steps.
Real GPT-2 pretraining uses batch 512, seq_len 1024, 300K+ steps on CUDA.
Historical track record for init improvements scaling to production is mixed
(LSUV benefits disappeared on ImageNet; Fixup required extra regularization).
We have zero data beyond 1,000 steps.

---

## Validated Mechanism Claims

### 1. Spectral shape carries task-relevant information (not just scale)

**Status: Supported, with ablation caveat**

imt_flat (all SVs = 1, default scale): PPL 972. imt_extracted (shaped SVs,
default scale): PPL 793. Same Frobenius norm, shape wins. This is the clean
comparison.

The imt_scaled_flat ablation (flat SVs at pretrained Frobenius norms → PPL
2,077 with gradient explosion) is confounded: pretrained norms are ~9x larger,
and gradient explosion is the expected outcome regardless of shape (Yang, Simon
& Bernstein, 2025, show spectral norm governs feature learning dynamics).

**Theoretical basis**: Saxe, McClelland & Ganguli (2014) showed explicitly
that SVD structure at initialization determines learning dynamics in deep
linear networks, and that pretraining speeds learning by aligning singular
vectors with input-output covariance structure. This is the strongest
theoretical foundation for our findings — more direct than dynamical isometry
(which was developed for very deep unnormalized networks, not transformers)
or Martin & Mahoney's heavy-tail framework (which is descriptive of trained
networks, not prescriptive for initialization).

### 2. Pretrained singular vector directions add signal beyond magnitudes

**Status: Confirmed — strongest claim, well-supported**

Group-averaged DCT spectra alone (base LR): PPL 818. Adding 50% pretrained V
alignment: PPL 788. Adding 50% UV alignment: PPL 731. At 2x LR with UV: PPL
572. Each addition provides genuine new information.

**Literature support**:
- PiSSA (Meng et al., NeurIPS 2024 Spotlight) showed principal singular
  values AND vectors from pretrained weights accelerate LoRA convergence
- Conv-SVD (Ng et al., Neural Computation 2023) transfers singular vectors
  from pretrained convolution weights
- Staats et al. (2024) showed right singular vectors of transformer weights
  develop significant overlap with activation covariance eigenvectors
- The lazy-vs-rich regime framework (Chizat & Bach, NeurIPS 2019) suggests
  V alignment puts the model closer to the feature-learning regime from step 0

**Important scope**: PiSSA, DoRA (Liu et al., ICML 2024), Conv-SVD, and
OSoRA (2025) all combine spectral and directional components from pretrained
models. Our novelty is applying this to *from-scratch training* rather than
fine-tuning. The conceptual combination is established; the application
context is new.

### 3. Spectral init enables higher learning rates

**Status: Confirmed — theoretical basis is solid**

Orthogonal init diverges (NaN) at 3x base LR (1.875e-4). Spectral init
handles 2x LR (1.25e-4) and achieves its best results there (PPL 714 at 2x
vs 818 at 1x). The 3x LR (PPL 734) is past the optimum but doesn't diverge.

**Theoretical basis**: Lewkowycz et al. (2020, "The Catapult Mechanism")
derives η_max ≈ 4/λ₀ for linear networks, where λ₀ is the initial loss
curvature. If spectral init produces a loss landscape with lower initial
curvature than orthogonal init, this directly predicts higher maximum
stable learning rate. This is the most appropriate theoretical framework —
more precise than general SAM/sharpness references.

**Caveat**: The relationship between weight matrix spectrum and loss Hessian
spectrum is indirect (the Hessian depends on both weights and data). We have
not measured the loss Hessian directly.

### 4. Condition number is NOT the explanatory variable

**Status: Correct conclusion, needs better framing**

Orthogonal init has perfect conditioning (cond = 1.0) and lowest gradient
norms yet converges slowest. This rules out condition number as explanatory.

**Better framework**: Per the implicit regularization literature (Razin &
Cohen, NeurIPS 2020; Arora et al., NeurIPS 2019), deep networks naturally
develop low-rank (anisotropic) weight structures during training. Spectral
init pre-loads this structure, reducing the optimization path. Effective rank
(Huh, 2023) and stable rank (Vershynin, 2018) are more appropriate metrics
than condition number for understanding why anisotropic initialization helps.

---

## Negative Results (What Didn't Work)

### Per-layer spectra are worse than group-averaged (PPL 1,221 vs 802)

Each layer's spectrum is estimated from only 2-4 weight matrices. The noise
overwhelms the layer-specific signal. Group averaging across all layers
within a type (attention, FFN up, FFN down, embedding) acts as variance
reduction. Evidence that layer-specific information *exists*: shuffling
spectra across layers hurts (PPL 1,384 vs 802), and the spectral unfolding
generator (24 meta-parameters capturing smooth depth variation) achieves PPL
851 — between group-averaged and per-layer.

### Cross-model extraction degrades (GPT-2 medium → small: PPL 1,016)

Dimension mismatch (1024 vs 768 hidden, 24 vs 12 layers) corrupts the
spectral transfer. This is expected — knowledge distillation also degrades
with large teacher-student capacity gaps (Mirzadeh et al., AAAI 2020).

### Lambda > 1 explodes, < 1 slows convergence

Lambda 2.0 creates basins too sharp to navigate (PPL 6,025). Lambda 0.5
converges more stably but slower (PPL 1,357 at step 750, eventually reaching
815 at step 1000 — it needs more time). Lambda 1.0 is the sweet spot.

---

## Spectral Compressibility

32 DCT coefficients (8 per group) capture pretrained spectral structure with
r > 0.93 mean correlation (actual sorted SVs vs pretrained). Even n_dct=2
gives r > 0.86.

**Caveat from review**: The Marchenko-Pastur distribution for random matrices
is also smooth and compressible. The high correlation may partly reflect
spectral smoothness rather than task-specific structure. The task-specific
signal likely lives in the *deviations* from Marchenko-Pastur in the leading
singular values (confirmed by Staats et al. 2024). A proper analysis should
measure compression quality of the MP-deviation component separately.

---

## What We Claim Is Novel

1. **The specific pipeline applied to from-scratch training**: SVD extraction
   → DCT compression → spectral reshaping + partial V alignment, used to
   initialize a fresh model for training from scratch. Adjacent work (PiSSA,
   DoRA, Conv-SVD) applies similar decompositions in fine-tuning contexts.
   The from-scratch application is the novel contribution.

2. **The empirical characterization of what transfers**: Shape not scale,
   directions add signal, group averaging beats per-layer, 50% alignment is
   the sweet spot, 2x LR is optimal. This systematic map of the transfer
   space is a contribution even where individual components have precedent.

3. **The autoresearch optimization approach**: 27 systematically tested
   hypotheses that turned a 2.33x finding into 3.33x through stacking
   independently discovered improvements (spectral shape + directional
   alignment + LR tolerance).

## What We Do NOT Claim Is Novel

- Combining spectral and directional information (PiSSA, DoRA, Conv-SVD)
- Gradient spike mitigation via step-skipping (PaLM, SPAM)
- DCT compression of neural network weights (FRONT, harmonic CNNs)
- The observation that trained networks have heavy-tailed spectra
  (Martin & Mahoney 2019/2021)

---

## Open Questions for CUDA Validation

1. Does the 3.33x hold at batch 512, seq_len 1024, on CUDA?
2. Does the advantage persist past 1,000 steps? At 10K? At convergence?
3. Multi-seed validation on the best config (lr_2x_UV)?
4. Does this work for GPT-2 medium/large (self-extraction, same architecture)?
5. Clean shape-vs-scale ablation at matched Frobenius norms?
6. Direct measurement of loss Hessian eigenvalues at initialization?
7. Does SIPE outperform mimetic initialization (Trockman & Kolter, 2023)?

---

## Critical References

| Paper | Relevance |
|-------|-----------|
| Saxe, McClelland & Ganguli (2014) | Theoretical foundation: SVD at init determines learning dynamics |
| Trockman & Kolter (ICML 2023) | Mimetic init: closest conceptual predecessor |
| PiSSA (Meng et al., NeurIPS 2024) | Pretrained SV + vectors for LoRA init |
| DoRA (Liu et al., ICML 2024) | Magnitude + direction decomposition |
| Conv-SVD (Ng et al., 2023) | Transfers pretrained singular vectors |
| Lewkowycz et al. (2020) | Catapult mechanism: why init affects max LR |
| Martin & Mahoney (JMLR 2021) | Heavy-tail framework (descriptive, not prescriptive) |
| Staats et al. (2024) | SV deviations from MP encode task info |
| Ash & Adams (NeurIPS 2020) | Shrink & Perturb: warm-starting analysis |
| Razin & Cohen (NeurIPS 2020) | Implicit rank minimization in deep networks |
| PaLM (Chowdhery et al., 2022) | Batch-skipping for spike mitigation |
| SPAM (2025) | Spike-aware Adam (similar threshold) |
| Yang, Simon & Bernstein (2025) | Spectral condition for feature learning |
