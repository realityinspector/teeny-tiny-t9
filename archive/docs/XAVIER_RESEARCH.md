# Xavier initialization and beyond: spectral theory of neural network weights

**Xavier/Glorot initialization preserves signal variance through deep networks by setting weight variance to 2/(n_in + n_out), but its assumptions—linear activations, layer independence, and i.i.d. weights—break down for every major modern architecture.** The field has evolved from variance-matching heuristics toward spectral control of the full Jacobian singular value distribution, culminating in dynamical isometry theory and architecture-specific schemes like muP, HiPPO, and adaLN-Zero. This report traces that evolution from first principles through open frontiers, distinguishing established results from speculative directions throughout.

---

## 1. Deriving the Xavier variance formula from signal propagation

The foundational paper "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, AISTATS 2010) derives initialization conditions by requiring that both forward activations and backward gradients maintain stable variance across layers.

**Setup and assumptions.** Consider layer *l* with pre-activation z^[l] = W^[l] a^[l−1] + b^[l] and activation a^[l] = f(z^[l]). The derivation rests on four assumptions: (1) the activation function is approximately linear near zero, with f'(0) = 1 and f odd-symmetric (valid for tanh, softsign near initialization); (2) activations and weights have **zero mean**; (3) weights are i.i.d. and independent of activations; (4) biases are initialized to zero.

**Forward pass.** For a single neuron, z_k = Σ_j w_{kj} a_j. By the product variance identity with zero means:

> Var(z^[l]) = n_in · Var(W^[l]) · Var(a^[l−1])

Preserving variance requires **n_in · Var(W) = 1**, yielding Var(W) = 1/n_in—which is precisely LeCun initialization (LeCun, Bottou, Orr & Müller, "Efficient BackProp," 1998).

**Backward pass.** Analogous analysis of gradient propagation ∂C/∂s^[l] through the transpose weights gives the condition **n_out · Var(W) = 1**. Since both conditions cannot hold simultaneously unless n_in = n_out, Glorot and Bengio take the harmonic mean compromise:

> **Var(W) = 2/(n_in + n_out)**

For uniform initialization: W ~ U[−a, a] where a = √(6/(n_in + n_out)), derived from Var(U[−a,a]) = a²/3.

**He initialization corrects for ReLU's half-rectification.** He, Zhang, Ren & Sun ("Delving Deep into Rectifiers," ICCV 2015) observed that ReLU zeroes out negative pre-activations, so E[ReLU(z)²] = ½ Var(z) for zero-mean z. This introduces a factor of ½ per layer:

> Var(a^[l]) = ½ · n_in · Var(W) · Var(a^[l−1])

The correction yields **Var(W) = 2/n_in** (He normal). For Leaky ReLU with slope α, the factor generalizes to 2/(1+α²)/n_in. He et al. demonstrated that a 30-layer ReLU network **fails to converge entirely** with Xavier initialization but trains normally with He initialization.

**Exponential failure with depth.** Define α_l = n_l · Var(W^[l]) · γ, where γ accounts for the activation (γ = 1 for linear, γ = ½ for ReLU). If α is constant across layers, forward signal variance after L layers scales as:

> Var(z^[L]) = α^L · Var(x)

With the pre-Xavier default U[−1/√n, 1/√n], the effective α = 1/3 per layer. After just 5 layers, forward variance drops to (1/3)^5 ≈ 0.004× the input—**99.6% signal loss**. At L = 10, attenuation reaches ~10⁻⁵. With Xavier on ReLU (α = ½ for square layers), 100 layers yields (½)^100 ≈ **10⁻³⁰**—numerically indistinguishable from zero in any floating-point format. Even with "correct" initialization, a 2023 analysis showed that the **kurtosis** of the output distribution grows unboundedly with depth in ReLU networks, causing the empirical variance (on finite data) to collapse despite constant theoretical variance (Springer, "The Vanishing Empirical Variance in Randomly Initialized Deep ReLU Networks," 2023).

---

## 2. Where Xavier's assumptions shatter

Xavier initialization rests on a mean-field picture of independent, linear, elementwise signal processing. Every major architectural innovation of the past decade violates at least one of these assumptions.

**ReLU networks** break the linearity and symmetry assumptions most directly. The halving of variance per layer compounds to exponential decay under Xavier scaling. He initialization fixes this for vanilla feedforward networks but does not address higher-order effects: the distribution of activations becomes increasingly heavy-tailed with depth, and finite-sample statistics diverge from infinite-width predictions.

**Residual connections** introduce additive signal accumulation. For y = F(x) + x, if both branches have variance σ², then Var(y) = 2σ². Across L residual blocks, output variance grows linearly as ~Lσ²—a qualitatively different failure mode (linear explosion rather than exponential). Taki (2017) derived that initialization variance must scale as O(1/(nL)) to compensate. FixUp (Zhang, Dauphin & Ma, ICLR 2019) addresses this by scaling residual branch weights by **L^{−1/(2m−2)}** and zero-initializing the final layer of each branch. De & Smith (NeurIPS 2020) showed batch normalization implicitly downscales residual branches by O(1/√L), explaining its effectiveness in deep ResNets.

**Attention mechanisms** violate Xavier at multiple levels. The dot-product q·k in attention has variance d_k when q and k entries have unit variance, pushing softmax toward saturation without the 1/√d_k scaling (Vaswani et al., 2017). Softmax is non-elementwise—it couples all sequence positions, invalidating per-neuron variance analysis. T-Fixup (Huang et al., ICML 2020) derived that attention weights must be scaled by **(9N)^{−1/4}** for N-layer Transformers, and DS-Init (Zhang et al., EMNLP 2019) proposed depth-proportional variance reduction for deep Transformers.

**Very deep networks (>100 layers)** expose the limits of first-moment analysis. Schoenholz, Gilmer, Ganguli & Sohl-Dickstein ("Deep Information Propagation," ICLR 2017) identified characteristic depth scales ξ_c and ξ_∇ that bound maximum trainable depth. At the edge of chaos (χ₁ = 1), these diverge to infinity; away from criticality, trainable depth is fundamentally finite. For ResNets, Yang & Schoenholz (NeurIPS 2017) showed convergence is **polynomial** rather than exponential—a crucial advantage but still limiting at extreme depths.

**Very wide networks (>512 units)** are where Xavier actually improves, since the Central Limit Theorem justifies the mean-field approximation as width → ∞. However, Bordelon & Pehlevan (NeurIPS 2023) quantified finite-width corrections at **O(1/√n)** for dynamical mean-field order parameters. Nguyen & Pham (2020) proved that i.i.d. initializations cause **strong degeneracy** for networks with ≥4 layers in the mean-field limit, requiring correlated initializations for proper behavior.

### The Marchenko-Pastur law reveals the spectral gap between initialization and learning

For an N × M random matrix W with i.i.d. entries of variance σ², the eigenvalues of (1/N)W^TW converge to the **Marchenko-Pastur distribution** (Marchenko & Pastur, 1967) with density:

> dν(x) = [√((λ₊ − x)(x − λ₋))] / (2πσ²γx) for x ∈ [λ₋, λ₊]

where λ± = σ²(1 ± √γ)², γ = M/N is the aspect ratio, and a point mass at zero appears when γ > 1. Under Xavier scaling (σ² = 2/(N+M)), the **condition number** κ ≈ (1+√γ)²/(1−√γ)² for γ < 1—worst at γ = 1 (square matrices) where λ₋ = 0 and the matrix is maximally ill-conditioned.

**At initialization**, weight matrices match Marchenko-Pastur predictions precisely—verified by Thamm, Staats & Rosenow (2022), who confirmed that singular values, eigenvector entries (Porter-Thomas distribution), and level spacings (Wigner surmise) all follow random matrix universality.

**After training**, the picture changes dramatically. Martin & Mahoney, in a series of papers spanning 2018–2021, demonstrated that trained weight matrices develop **heavy-tailed** eigenvalue distributions following power laws ρ(λ) ~ λ^{−1−μ/2}. Their foundational 73-page analysis (JMLR 2021) identified **5+1 phases of training**: random-like → bleeding-out → bulk+spikes → bulk decay → heavy-tailed → rank collapse. The weighted alpha metric (Σ_l α_l · log(λ_max,l)) correlates strongly with test accuracy across VGG, ResNet, DenseNet, and GPT families—**without access to training or test data** (Nature Communications, 2021). Layers with smaller α (heavier tails) are better-trained. A 2024 RMT analysis of BERT, Pythia, and Llama-8B found that **query matrices** deviate most from Marchenko-Pastur, indicating the strongest feature learning, while attention-output matrices remain closer to random (arXiv 2410.17770).

---

## 3. Dynamical isometry and the spectral perspective on initialization

The spectral perspective reframes initialization from "match the variance" to "control the entire singular value distribution of the input-output Jacobian." **Dynamical isometry**—the condition that all singular values of the end-to-end Jacobian concentrate near 1—is strictly stronger than variance preservation and enables qualitatively faster training.

**Saxe, McClelland & Ganguli ("Exact solutions to the nonlinear dynamics of learning in deep linear networks," ICLR 2014)** established the theoretical foundation. Despite producing linear input-output maps, deep linear networks exhibit nonlinear learning dynamics with long plateaus and rapid transitions. The key result: learning dynamics decouple along singular vector modes of the input-output correlation matrix. **Orthogonal initialization achieves exact dynamical isometry** in linear networks, yielding depth-independent learning times. Gaussian initialization produces a log-normal singular value distribution whose spread grows with depth, causing learning time to increase with network depth.

**Pennington, Schoenholz & Ganguli (NeurIPS 2017)** extended this to nonlinear networks using the **S-transform from free probability theory** to analytically compute the full Jacobian spectrum. Their critical finding: **ReLU networks are fundamentally incapable of dynamical isometry**—even with orthogonal initialization at criticality, the singular value distribution accumulates mass at zero with depth. In contrast, tanh networks *can* achieve dynamical isometry with orthogonal initialization when the fixed-point pre-activation variance q* is small. Networks closer to isometry learned **orders of magnitude faster**, with optimal learning time scaling as O(√L) rather than exponentially.

**The edge of chaos provides the complementary mean-field view.** Poole et al. (NeurIPS 2016) discovered an order-to-chaos phase transition governed by χ₁ = σ_w² E[φ'(h)²]. In the ordered phase (χ₁ < 1), signals contract and inputs become indistinguishable; in the chaotic phase (χ₁ > 1), perturbations amplify exponentially. Criticality (χ₁ = 1) is *necessary* for trainability but *not sufficient*—it only ensures the **mean squared** singular value is O(1). Dynamical isometry additionally requires **concentration** of all singular values near 1, which demands both criticality and orthogonal (not Gaussian) weight statistics.

### From theory to practice: spectral initialization methods

Several methods translate spectral insights into practical algorithms:

- **Delta-orthogonal initialization** (Xiao, Bahri, Sohl-Dickstein, Schoenholz & Pennington, ICML 2018) extends dynamical isometry to CNNs. Standard convolution operators have Marchenko-Pastur-distributed singular values regardless of weight initialization (by universality), preventing isometry. Delta-orthogonal uses spatially non-uniform variance concentrated at the kernel center (like a Kronecker delta) with orthogonal channel mixing, achieving isometry across spatial frequencies. Result: trained **vanilla 10,000-layer CNNs** without skip connections or normalization.

- **LSUV** (Mishkin & Matas, ICLR 2016) takes a data-driven approach: (1) initialize with orthonormal matrices, (2) forward-pass a mini-batch, (3) rescale each layer's weights by the output standard deviation. This iterative procedure converges in 1–5 steps per layer and works across activation functions and architectures. It effectively performs empirical variance normalization, compensating for the nonlinearity's actual effect on signal statistics rather than relying on theoretical assumptions.

- **Fixup** (Zhang, Dauphin & Ma, ICLR 2019) targets ResNets specifically: zero-initialize the last layer of each residual branch (so the network starts as approximately identity) and scale remaining weights by L^{−1/(2m−2)}. This makes the initial Jacobian close to the identity matrix, achieving approximate dynamical isometry. Successfully trained **10,000-layer ResNets without any normalization**.

- **MetaInit** (Dauphin & Schoenholz, NeurIPS 2019) learns initialization by optimizing the **gradient quotient**—a measure of how much curvature affects gradient steps. By minimizing this via gradient descent over layer-wise weight norms, MetaInit implicitly controls Jacobian spectral properties. It requires Hessian-vector products (computationally expensive) but enables training plain networks without normalization or skip connections.

---

## 4. How modern architectures actually initialize (2023–2025)

The practical initialization landscape has fragmented: each major architecture family has developed its own scheme, often discovered empirically and rationalized post-hoc.

**Large language models** converge on a simple recipe. Both GPT-2/3 (Radford et al., 2019; Brown et al., 2020) and LLaMA (Touvron et al., 2023) initialize all weight matrices from a **truncated normal with σ = 0.02**. The critical addition is **residual scaling**: output projections in attention and MLP blocks are scaled by **1/√(2N)** where N is the total layer count—the factor of 2 accounts for two residual contributions per layer. RMSNorm/LayerNorm gains are set to 1, biases to 0. Pre-layer normalization is now universal. DeepNorm (Wang et al., 2022) extends this to very deep models with a constant scaling factor α.

**State space models** depend critically on **HiPPO initialization** (Gu, Dao, Ermon, Ré & Rudra, NeurIPS 2020). The state matrix A is initialized as the HiPPO-LegS matrix, which tracks Legendre polynomial coefficients to optimally approximate input history. This single initialization choice improved sequential MNIST from **60% to 98%**. S4 (Gu et al., NeurIPS 2021) uses a diagonal-plus-low-rank parameterization of HiPPO. Mamba (Gu & Dao, 2023) uses S4D-Real initialization (diagonal elements set to −(n+1)) with log-uniform timescale initialization in [0.001, 0.1]. HiPPO is a genuinely task-class-specific spectral initialization: it encodes optimal memory structure rather than generic signal preservation.

**Diffusion models** have adopted **identity initialization via zero gating**. The Diffusion Transformer (DiT; Peebles & Xie, 2023) uses adaLN-Zero: modulation parameters that gate residual connections are initialized to zero, making each block start as the identity function. A 2024 study ("Unveiling the Secret of AdaLN-Zero") confirmed that zero initialization is the **single most influential factor** in DiT's performance—more important than the conditioning mechanism itself. The U-Net diffusion models (DDPM, ADM) similarly zero-initialize final convolutions in each residual block.

### muP reframes initialization as a width-dependent parameterization problem

**Maximal Update Parameterization** (Yang & Hu, "Tensor Programs V," 2022) represents a paradigm shift. Rather than deriving a single variance formula, muP defines width-dependent scaling rules that ensure **optimal hyperparameters transfer exactly across model scales**. The key desiderata—O(1) network output and maximal parameter updates without divergence—uniquely determine muP.

Under muP, hidden weight initialization variance includes an additional **1/m_d** factor (where m_d is the width multiplier relative to a base model), output logits are scaled by 1/m_d, and attention uses **1/d_head** scaling instead of the standard 1/√d_head. Learning rates for hidden layers scale as η/m_d under Adam. This enables training a small proxy model, tuning hyperparameters, and transferring them directly to the full-scale model—Cerebras-GPT validated this from 40M to 2.7B parameters.

Recent extensions include **u-μP** (Blake et al., 2024), which combines muP with unit scaling for out-of-the-box FP8 training; **Sparse muP** (Dey et al., 2024), extending to static sparsity with 11.9% loss improvement at 99.2% sparsity; and **DiT muP** (Zheng et al., 2025), applying the Tensor Programs formalism to diffusion transformers and Fourier neural operators. The Cerebras Practitioner's Guide (Dey, Anthony & Hestness, 2024) provides the definitive implementation reference, recommending coordinate checks across widths and base width ≥ 256.

### Learned and data-dependent initialization remains niche but growing

Meta-learning approaches treat initialization as the outer-loop optimization target. **MAML** (Finn et al., 2017) meta-learns initial parameters for few-shot adaptation; Tancik et al. (2020) applied this to coordinate-based neural representations, achieving convergence in **2 gradient steps** versus 60+ for standard initialization. Hypernetwork-based approaches (Zhao et al., 2020) use GNNs to map architecture DAGs to initial weight tensors, enabling architecture-agnostic initialization.

For fine-tuning, SVD-aligned initialization has gained traction: **EVA** (Paischer et al., 2024) uses minibatch incremental SVD to align LoRA adapter initialization with the dominant singular subspace of pre-trained weights. **POET** (2025) reparameterizes LLM training via orthogonal equivalence transformations, maintaining high spectral diversity (SVD entropy) throughout training.

Direct NAS for initialization distributions remains largely unexplored. The field has instead focused on learning initial *weights* (meta-learning) or learning structural *priors* (spectral shapes for neural fields), rather than searching over initialization distribution families.

---

## 5. Open frontiers: non-uniform spectra, task structure, and universality

**Can non-uniform singular value distributions outperform Xavier?** The theoretical case is strong but the empirical program is nascent. Xavier and He produce approximately flat singular value spectra, yet trained networks consistently develop heavy-tailed distributions with power-law exponents correlating with generalization quality (Martin & Mahoney, 2021). Kunin et al. (2024) proved analytically that **unbalanced initializations accelerate feature learning** in deep linear networks—effectively demonstrating that a non-uniform spectrum across layers promotes faster convergence. Singular Value Bounding (Jia et al., 2017) showed that constraining all singular values to [1/(1+ε), 1+ε] improves deep network training—a specific spectral shape distinct from both Xavier's random spectrum and uniform isometry.

**Task structure should inform initialization spectrum.** Emerging evidence supports this: Rahaman et al. (NeurIPS 2019) demonstrated that networks exhibit spectral bias, learning low-frequency functions first. An initialization shaped to match the target's frequency content could bypass this bottleneck. Bordelon et al. (ICLR 2025) showed that the eigenvalue decay rate β of the data covariance interacts with initialization parameterization to determine neural scaling law exponents—feature learning (muP) yields better exponents precisely when task structure (β < 1) makes the problem harder. Yunis et al. (2024) observed that grokking coincides with simultaneous discovery of low-rank solutions across all weight matrices, suggesting that the transition to generalization is itself a spectral phenomenon connected to task geometry.

**A universal spectral shape almost certainly does not exist in full generality, but architecture-class-specific optimal spectra are plausible.** The interaction between activation functions, architectural topology (attention vs. convolution vs. recurrence), and data distribution is too rich for a single shape to dominate. HiPPO initialization for SSMs—optimized for sequential memorization—generalizes across SSM variants but not to Transformers. muP provides "universal" scaling rules but prescribes only initialization *scale*, not spectral *shape*. The most promising direction is **principled spectral initialization conditioned on architecture class**: deriving the optimal singular value distribution for Transformers, a different one for SSMs, and yet another for convnets, potentially using the free probability tools from Pennington et al. (2017) combined with task-specific covariance information. A 2025 paper on initialization with nonlinear spectral bias characteristics (arXiv 2511.02244) represents early steps in this direction.

## Conclusion

The trajectory from Xavier to modern initialization reveals a progressive refinement: from scalar variance matching (LeCun 1998, Glorot & Bengio 2010, He et al. 2015), through full spectral control of the Jacobian (Saxe et al. 2014, Pennington et al. 2017), to architecture-specific parameterizations (muP 2022, HiPPO 2020, adaLN-Zero 2023). Three insights stand out as underappreciated. First, the Marchenko-Pastur-to-heavy-tail spectral transition during training is not mere epiphenomenon—it encodes the structure of learned representations and predicts generalization without test data. Second, muP's real contribution is not a better initialization formula but a **reparameterization** that makes initialization-learning rate coupling width-invariant, enabling systematic scaling. Third, the impossibility of dynamical isometry for ReLU networks (Pennington et al. 2017) is a foundational negative result that explains why architectural innovations (residual connections, normalization) became necessary—and why activation function choice, often treated as a minor hyperparameter, has deep theoretical consequences for trainability. The most actionable open frontier is developing spectral initialization theory that accounts for both architecture class and task covariance structure, moving beyond the architecture-agnostic assumptions that Xavier pioneered but that modern practice has thoroughly outgrown.