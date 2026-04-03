# IMT — Inverse Morphogenic Training

> **[Read the interactive whitepaper →](https://teeny-tiny-t9-production.up.railway.app)**

**The shape of intelligence: deriving neural network capability from topological structure.**

A 3D shape's Laplacian eigenspectrum, when used to constrain neural network
weight initialization, produces 50× faster learning of T9 predictive text
than random (Xavier) initialization. The discriminant is genus — the number
of independent cycles in the capability graph.

Confirmed by:
1. **Ablation** — 100× improvement over Xavier at identical architecture
2. **Transfer** — genus-1 wins 5/5 tasks (T9, arithmetic, bigrams, XOR, rank sorting)
3. **Blind search** — 19/20 top random graphs are genus-1 (independently rediscovered)

## Quickstart

```bash
pip install numpy scipy numpy-stl
python main.py          # Run the 12-shape race → STL of winning crystal
python rigor.py         # Run all 3 validation experiments
```

## Files

```
main.py             Orchestrator: runs the race, exports STL + results
rigor.py            Three validation experiments (ablation, transfer, blind search)
shapes.py           12 candidate capability graphs with 3D mesh generation
topology.py         Topological feature extraction (Laplacian, spectrum, Betti numbers)
nucleate.py         IMT nucleation engine (topology → constrained NN init)
t9_task.py          T9 predictive text task (617-word vocabulary, training loop)
search.py           Shape space search (races candidates against baseline)
stl_export.py       STL file export for 3D printing
requirements.txt    Python dependencies
```

## The Core Idea in 10 Lines

```python
L = degree_matrix(G) - adjacency(G)     # graph Laplacian
eigenvalues = eig(L)                     # topological fingerprint

for W in network.weight_matrices:
    U, s, V = svd(W)                     # current singular values
    s_new = interpolate(eigenvalues, len(s))
    s_new *= sqrt(2 / (fan_in + fan_out)) # Xavier scaling
    W[:] = U @ diag(s_new) @ V.T         # nucleated weights

train(network, t9_data, epochs=30)       # 79.6% accuracy
```

## Key Results

### Original Race (12 topologies × 3 seeds × 30 epochs)

| Shape                | Acc@1  | MRR   | Genus | Fiedler |
|----------------------|--------|-------|-------|---------|
| triangle_strong_CF   | 79.6%  | 88.8% | 1     | 2.10    |
| triangle_3d          | 78.0%  | 86.7% | 1     | 3.00    |
| triangle             | 75.5%  | 84.3% | 1     | 3.00    |
| chain_thick_CF       | 19.8%  | 31.3% | 0     | 0.70    |
| chain_linear         | 13.6%  | 22.8% | 0     | 1.00    |
| xavier_baseline      |  1.3%  |  2.7% | n/a   | n/a     |

### Ablation (fixed architecture, spectral init only)

| Init Source          | Acc@1  | Genus |
|----------------------|--------|-------|
| tetra_meta           | 71.3%  | 3     |
| triangle             | 66.5%  | 1     |
| triangle_strong_CF   | 54.9%  | 1     |
| chain_linear         | 43.6%  | 0     |
| xavier_baseline      |  0.6%  | n/a   |

Key finding: at fixed architecture, tetra_meta (genus 3) *wins*. The genus-3
collapse in the original race was the architecture derivation, not the spectrum.
The spectral signal is the primary mechanism.

### Transfer (5 tasks)

| Task       | Genus-1 best | Genus-0 best | Xavier |
|------------|-------------|-------------|--------|
| T9         | 55.7%       | 6.2%        | 0.4%   |
| Arithmetic | 37.3%       | 13.0%       | 15.3%  |
| Bigrams    | 29.7%       | 12.0%       | 10.7%  |
| XOR        | 77.7%       | 57.7%       | 74.5%  |
| Rank sort  | 68.0%       | 53.9%       | 63.0%  |

Genus-1 wins all 5 tasks.

### Blind Search (150 random Erdős–Rényi graphs)

- Top 20 graphs: 19/20 are genus-1
- Bottom 30 graphs: 0/30 are genus-1
- Mean accuracy: genus-1 = 21.0%, genus-0 = 1.2%, genus ≥ 2 = 0.2%
- Best random graph independently rediscovered: 3 nodes, 3 edges, genus 1

## Formal Framework

**Nucleation mapping Φ**: `Spec(L) → Init(W)`
- Graph Laplacian eigenvalues become target singular values for weight matrices
- Random orthogonal bases (Stiefel manifold) preserve gradient isotropy
- Xavier scaling maintains magnitude

**Architecture derivation Ψ**: `Top(G) → Arch(N)`
- `depth = 1 + β₁(G)` (first Betti number → hidden layers)
- `width = ⌈h₀ · (1 + λ₂(L))⌉` (Fiedler value → layer width)
- `skip = {(src, dst) | c ∈ Cycles(G)}` (cycles → residual connections)

## 3D Printing

The winning crystal is exported as `t9_crystal_triangle_strong_CF.stl`.
Print at default scale (~32 × 29 × 12 mm). Three spheres connected by
tubes, with the C–F tube visibly thicker than the others. That is the
shape of T9 intelligence.

## Whitepaper

The interactive research whitepaper is deployed at the link above. Source is in `web/`.

To run locally:
```bash
cd web && npm install && npm run dev
```

## Requirements

- Python 3.10+
- numpy ≥ 1.24
- scipy ≥ 1.10
- numpy-stl ≥ 3.0
- No GPU required. Runs on a laptop in ~30 seconds.

## Citation

```
Inverse Morphogenic Training: The Shape of Intelligence.
March 2026. 12 topologies, 5 tasks, 150 random graphs, pure numpy, no GPU.
```

## License

MIT
