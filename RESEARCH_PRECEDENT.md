# Research Precedent: Spectral Initialization via Pretrained Extraction

**Purpose**: Claims below are for a research agent to vet. Each claim has a
type (BINARY yes/no, or SCALE 1-5) and enough context to evaluate.

**Our method**: Extract SVD singular value spectra from a pretrained GPT-2,
compress to 8 DCT coefficients per weight group (32 numbers total), use these
to shape the initialization of a fresh randomly-initialized GPT-2, with 50%
directional alignment of right singular vectors. Train at 2x the standard
learning rate. Achieves 3.33x faster convergence (PPL 572 vs 1904 at step 750)
on WikiText-2 with GPT-2 small (124M params).

---

## 1. Prior Art Claims

### 1.1 Dynamical isometry is the dominant theory of optimal initialization
- **Type**: BINARY
- **Claim**: The prevailing theoretical framework for understanding neural network initialization (Saxe et al. 2014, Pennington et al. 2017, 2018) focuses on dynamical isometry — making all singular values of the input-output Jacobian equal to 1. Our work contradicts this: orthogonal init (perfect isometry, cond=1.0) converges slowest, while anisotropic spectral init (cond=14 for attention, 76 for embeddings) converges 3.33x faster.
- **Question**: Is dynamical isometry actually the dominant framework, and does our finding genuinely contradict it rather than being a special case?

### 1.2 No prior work transfers SVD spectra from pretrained models as initialization
- **Type**: BINARY
- **Claim**: To our knowledge, no prior work has extracted the singular value distribution from a trained model's weight matrices and used it as a spectral template for initializing a new model from scratch. Related work includes: knowledge distillation (transfers activations/logits, not weight structure), lottery ticket hypothesis (transfers sparse masks), Net2Net (transfers exact weights for widening/deepening), and spectral normalization (constrains the largest SV during training). None of these transfer the full SV distribution shape.
- **Question**: Does such prior work exist? Check: weight transfer, SVD-based initialization, spectral methods in deep learning.

### 1.3 Martin & Mahoney's heavy-tail theory predicts our finding
- **Type**: SCALE (1=contradicts, 5=directly predicts)
- **Claim**: Martin & Mahoney (2019, 2021) showed that well-trained neural network weight matrices exhibit heavy-tailed singular value distributions, with the tail exponent (alpha) correlating with generalization quality. Our extracted spectra show exactly this heavy-tail structure. Their framework would predict that initializing with the correct spectral shape should accelerate training, which is what we observe.
- **Question**: How well does their theoretical framework predict our specific finding? Do they discuss initialization implications?

### 1.4 AdamW erases initialization effects within a few hundred steps
- **Type**: BINARY
- **Claim**: A common belief is that adaptive optimizers (Adam, AdamW) quickly erase initialization effects because per-parameter learning rates adapt to the local loss landscape. Our results show the 3.33x advantage persists through 1,000 optimizer steps and is still growing at termination. However, we have not tested beyond 1,000 steps.
- **Question**: Is the "Adam erases init effects" claim actually established in the literature, or is it an informal belief? What do papers like (Loshchilov & Hutter 2019, Zhang et al. 2019) say?

### 1.5 Sharpness-aware optimization literature predicts our stability findings
- **Type**: SCALE (1=unrelated, 5=directly predicts)
- **Claim**: We find that spectral init creates sharper loss basins (faster convergence but vulnerable to gradient spikes), while orthogonal init sits in flat basins (slow but stable). This maps to the sharpness-aware optimization literature (Foret et al. 2021, SAM; Keskar et al. 2017). Our spike-skip mechanism is a crude form of sharpness-aware training.
- **Question**: Does the SAM/sharpness literature specifically discuss initialization-induced sharpness? Or is this a novel connection?

---

## 2. Mechanism Claims

### 2.1 The advantage is from spectral SHAPE, not SCALE
- **Type**: BINARY (our ablation is clean)
- **Claim**: Flat singular values scaled to pretrained Frobenius norms (imt_scaled_flat) produce gradient explosion and PPL 2,077. Shaped singular values at default N(0,0.02) scale produce PPL 572. The spectral distribution shape, not the overall magnitude, carries the signal.
- **Question**: Is this ablation design (shape vs scale) standard in the spectral initialization literature? Are there better controls we should run?

### 2.2 The mechanism is NOT better numerical conditioning
- **Type**: BINARY (our Jacobian analysis is clean)
- **Claim**: Orthogonal init has perfect condition numbers (σ_max/σ_min = 1.0 for all layers) and the lowest gradient norms, yet converges slowest. Spectral init has worse conditioning (cond=14 for attention, cond=76 for embeddings) yet converges 3.33x faster. This rules out numerical conditioning as the mechanism.
- **Question**: Is condition number the right metric for "numerical conditioning" here? Should we also measure the effective rank, stable rank, or some other spectral property?

### 2.3 Directional alignment (pretrained V vectors) adds signal beyond spectral magnitudes
- **Type**: BINARY
- **Claim**: Group-averaged DCT spectra alone: PPL 714. Adding 50% pretrained V alignment: PPL 572. The pretrained model's right singular vectors encode which directions in embedding space matter for language modeling, and transferring this directional information accelerates convergence beyond what spectral magnitudes alone provide.
- **Question**: Is this related to the "feature learning" vs "lazy training" distinction (Chizat & Bach 2018)? Does transferring V vectors put the model in the feature learning regime from step 0?

### 2.4 Spectral init enables higher learning rates that would cause orthogonal init to diverge
- **Type**: BINARY
- **Claim**: Orthogonal init diverges (NaN) at 3x the base LR (1.875e-4). Spectral init handles 2x LR (1.25e-4) and produces better results at this higher LR. The spectral shape creates a loss landscape with better LR tolerance.
- **Question**: Is there precedent for initialization methods that change the maximum stable learning rate? Does the "Edge of Stability" literature (Cohen et al. 2021) predict this?

---

## 3. Compressibility Claims

### 3.1 The spectral fingerprint is compressible to ~32 numbers
- **Type**: SCALE (1=misleading, 5=accurate and novel)
- **Claim**: 4 weight groups × 8 DCT coefficients = 32 numbers capture the spectral structure of a 124M parameter model with r > 0.97 correlation to the true pretrained SV distribution (for non-embedding groups). Even 2 DCT coefficients per group gives r > 0.94.
- **Question**: Is this level of compressibility surprising given the Marchenko-Pastur distribution for random matrices? How does it compare to the effective dimensionality of pretrained weight matrices as studied in intrinsic dimensionality literature (Li et al. 2018)?

### 3.2 Group averaging acts as regularization — per-layer spectra are worse
- **Type**: BINARY
- **Claim**: Applying per-layer spectra (50 different spectra, one per weight matrix) gives PPL 1,221 — much worse than group-averaged spectra (PPL 802). The per-layer approach creates massive gradient instability (gnorms up to 7,700). Averaging across layers within a group smooths noise while preserving the useful coarse spectral signal.
- **Question**: Is this consistent with how spectral methods work in other domains (e.g., graph spectral methods, spectral clustering)? Is there a theoretical basis for why group averaging helps?

---

## 4. Scope and Limitations Claims

### 4.1 Effects this large at toy scale typically shrink at real scale
- **Type**: SCALE (1=usually doesn't shrink, 5=almost always shrinks dramatically)
- **Claim**: Our experiments use effective batch size 8, sequence length 256, on Apple MPS with 1,000 optimizer steps. Real GPT-2 pretraining uses batch 512, seq_len 1024, 300K+ steps on CUDA. Effects observed at toy scale frequently shrink or disappear at real scale. We have no data beyond our toy setting.
- **Question**: What is the historical track record for initialization improvements scaling from toy to real settings? Cite specific examples if possible.

### 4.2 The method requires a pretrained model of the same architecture
- **Type**: BINARY
- **Claim**: Extracting spectra from GPT-2 medium (355M, 24 layers) and applying to GPT-2 small (124M, 12 layers) gave PPL 1,016 — much worse than self-extraction (PPL 572). The spectral fingerprint appears architecture-specific. Cross-architecture transfer is an open question.
- **Question**: Is this consistent with how other transfer learning methods work? Does knowledge distillation also degrade significantly when teacher and student have different architectures?

### 4.3 This is conceptually related to "warm starting" but transfers less information
- **Type**: SCALE (1=unrelated, 5=essentially the same thing)
- **Claim**: Loading pretrained weights directly ("warm starting") would give perfect initialization. Our method transfers only the spectral shape (32 numbers) + partial directional alignment (50% V blend), which is much less information. The question is whether this reduced transfer is useful when you can't or don't want to transfer full weights (different vocab, different tokenizer, different objective).
- **Question**: Where does this fit on the spectrum from random init to full weight transfer? Are there other "partial transfer" methods in the literature?

---

## 5. Novelty Claims

### 5.1 No prior work combines spectral shaping with directional alignment for initialization
- **Type**: BINARY
- **Claim**: Our hybrid approach — using DCT-compressed spectral magnitudes for the singular values AND partially aligning the singular vectors with pretrained directions — is novel. Prior spectral initialization work (if any exists per claim 1.2) focused only on magnitudes.
- **Question**: Verify this claim. Check related work in structured initialization, few-shot weight generation, hypernetworks.

### 5.2 The spike-skip mechanism is a novel training stabilization technique
- **Type**: BINARY
- **Claim**: Skipping optimizer steps where the pre-clip gradient norm exceeds a multiple (50x) of the running median is, to our knowledge, novel. Related work includes gradient clipping (standard), loss spike detection in large-scale training (Chowdhery et al. 2022 PaLM), and stochastic gradient descent with restarts.
- **Question**: Does this mechanism exist under a different name in the literature?

### 5.3 The DCT compression of spectral shapes is a novel representation
- **Type**: BINARY  
- **Claim**: Representing a weight matrix's singular value distribution as 8 DCT coefficients is novel. Prior work uses raw singular values, truncated SVD, or spectral norms (single numbers). The DCT basis provides smooth interpolation for arbitrary matrix sizes.
- **Question**: Has anyone compressed singular value distributions this way before? Check signal processing + deep learning intersections.

---

## 6. Connections to Broader Theory

### 6.1 This relates to the "information geometry" of neural network training
- **Type**: SCALE (1=tenuous, 5=direct connection)
- **Claim**: The spectral shape of weight matrices defines the geometry of the loss landscape. By shaping the spectrum to match the pretrained model's, we are effectively initializing the model at a point in weight space with favorable information-geometric properties (sharper but better-located basin).
- **Question**: Does the natural gradient / information geometry literature (Amari, Martens) connect spectral properties of weights to loss landscape geometry?

### 6.2 The method can be viewed as a form of "spectral curriculum"
- **Type**: SCALE (1=bad analogy, 5=productive framing)
- **Claim**: Just as curriculum learning presents training data in a structured order, spectral initialization presents the weight space in a structured starting configuration. The spectral shape tells the optimizer "start here" rather than "learn this data first."
- **Question**: Is this analogy productive? Does the curriculum learning literature discuss initialization as a form of curriculum?

### 6.3 Connections to random matrix theory
- **Type**: SCALE (1=irrelevant, 5=foundational)
- **Claim**: The random N(0, 0.02) initialization produces Marchenko-Pastur distributed singular values. The pretrained model's heavy-tailed distribution deviates from this in specific, task-determined ways. Our method injects this deviation at initialization.
- **Question**: What does random matrix theory (Marchenko-Pastur, Tracy-Widom) predict about the effect of shaped vs random spectra on gradient dynamics? Are there RMT results about training dynamics under non-MP initial spectra?

---

## Instructions for Research Agent

For each claim above:
1. Search the literature (arXiv, Semantic Scholar, Google Scholar)
2. Provide a verdict: CONFIRMED, DENIED, PARTIALLY CONFIRMED, or UNCERTAIN
3. If DENIED or PARTIALLY CONFIRMED, explain what's wrong and cite the relevant paper
4. If CONFIRMED, cite the most relevant supporting reference
5. For SCALE claims, provide your rating and explain
6. Flag any claims where our framing is misleading even if technically correct
7. Identify any important prior work we've missed that we should cite
8. Suggest the 3 strongest and 3 weakest claims for building a paper around
