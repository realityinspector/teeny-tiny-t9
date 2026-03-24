#!/usr/bin/env python3
"""
IMT Rigor Suite — Three experiments that make or break the thesis.

Experiment 1: ABLATION — Hold architecture fixed, vary only spectral init.
              Isolates whether the eigenspectrum itself helps, independent of
              depth/width/skip connections derived from topology.

Experiment 2: TRANSFER — Run winning shapes on 4 different tasks.
              Tests whether topology generalizes or is T9-specific.

Experiment 3: BLIND SEARCH — Random + Bayesian search over continuous graph
              parameters. Tests whether optimization independently discovers
              genus-1 + asymmetric weights.

Usage: python rigor.py
"""

import sys, os, time, json
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import make_t9_candidates, CapabilityGraph
from topology import TopologicalFeatures
from nucleate import nucleate_from_topology, NucleatedNetwork
from t9_task import T9Task


# ═══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════

def spectral_init_fixed_arch(topo_features, layer_sizes, seed=42):
    """
    Nucleate with FIXED architecture but topology-shaped singular values.
    This is the ablation version: architecture is constant, only SVD changes.
    """
    rng = np.random.RandomState(seed)
    tf = topo_features
    target_spectrum = np.abs(tf.graph_eigenvalues.copy())
    if target_spectrum.max() > 0:
        target_spectrum = target_spectrum / target_spectrum.max()

    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        W = rng.randn(fan_in, fan_out)
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        n_sv = len(s)

        if len(target_spectrum) > 0:
            target_interp = np.interp(
                np.linspace(0, 1, n_sv),
                np.linspace(0, 1, len(target_spectrum)),
                target_spectrum
            )
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            shaped_sv = target_interp * scale * n_sv
            shaped_sv = np.maximum(shaped_sv, scale * 0.1)
        else:
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            shaped_sv = np.ones(n_sv) * scale

        W_shaped = U[:, :n_sv] @ np.diag(shaped_sv) @ Vt[:n_sv, :]
        weights.append(W_shaped)
        biases.append(np.zeros(fan_out))

    return NucleatedNetwork(layer_sizes, weights, biases, skip_connections=[])


def xavier_init_fixed_arch(layer_sizes, seed=42):
    """Standard Xavier for the same fixed architecture."""
    rng = np.random.RandomState(seed)
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        W = rng.randn(fan_in, fan_out) * scale
        weights.append(W)
        biases.append(np.zeros(fan_out))
    return NucleatedNetwork(layer_sizes, weights, biases, skip_connections=[])


def train_and_eval(task, network, n_epochs=30, lr=0.05):
    """Train and return final metrics + learning curve."""
    curve = []
    for ep in range(n_epochs):
        loss = task.train_epoch_fast(network, lr=lr, batch_size=32)
        curve.append(loss)
    metrics = task.evaluate(network)
    return metrics, curve


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: ARCHITECTURE-CONTROLLED ABLATION
# ═══════════════════════════════════════════════════════════════════

def run_ablation(n_seeds=5, n_epochs=30):
    """
    Hold architecture FIXED: 2 hidden layers, 64 units, no skip connections.
    Only vary the SVD singular value distribution from each topology.

    If spectral-from-triangle beats spectral-from-chain at identical arch,
    the topology signal is real.
    If not, we just rediscovered "deeper networks with residuals learn better."
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: ARCHITECTURE-CONTROLLED ABLATION")
    print("  Fixed arch: [input, 64, 64, output] — no skip connections")
    print("  Only varying: SVD singular value distribution from topology")
    print("=" * 70)

    task = T9Task()
    fixed_arch = [task.input_size, 64, 64, task.output_size]

    candidates = make_t9_candidates()
    # Select representative subset: best genus-1, best genus-0, worst, + controls
    selected_names = [
        "triangle_strong_CF",  # best genus-1
        "triangle",            # genus-1, symmetric
        "chain_thick_CF",      # best genus-0
        "chain_linear",        # genus-0, symmetric
        "disconnected_X",      # degenerate
        "tetra_meta",          # genus-3
    ]
    selected = [g for g in candidates if g.name in selected_names]

    results = {}
    for graph in selected:
        topo = TopologicalFeatures(graph)
        scores = []
        for seed in range(n_seeds):
            np.random.seed(seed * 100)
            net = spectral_init_fixed_arch(topo, fixed_arch, seed=seed)
            metrics, curve = train_and_eval(task, net, n_epochs, lr=0.05)
            scores.append(metrics['acc1'])
        results[graph.name] = {
            "mean_acc": np.mean(scores),
            "std_acc": np.std(scores),
            "genus": graph.cycle_rank,
            "fiedler": topo.fiedler,
            "scores": scores,
        }

    # Xavier baseline with same arch
    xavier_scores = []
    for seed in range(n_seeds):
        np.random.seed(seed * 200)
        net = xavier_init_fixed_arch(fixed_arch, seed=seed)
        metrics, curve = train_and_eval(task, net, n_epochs, lr=0.05)
        xavier_scores.append(metrics['acc1'])
    results["xavier_baseline"] = {
        "mean_acc": np.mean(xavier_scores),
        "std_acc": np.std(xavier_scores),
        "genus": None,
        "fiedler": None,
        "scores": xavier_scores,
    }

    # Uniform singular values (all equal, same total energy as Xavier)
    uniform_scores = []
    for seed in range(n_seeds):
        np.random.seed(seed * 300)
        net = xavier_init_fixed_arch(fixed_arch, seed=seed)
        # Flatten SVD to uniform
        for i, W in enumerate(net.weights):
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            s_uniform = np.ones_like(s) * np.mean(s)
            net.weights[i] = U @ np.diag(s_uniform) @ Vt
        metrics, curve = train_and_eval(task, net, n_epochs, lr=0.05)
        uniform_scores.append(metrics['acc1'])
    results["uniform_svd"] = {
        "mean_acc": np.mean(uniform_scores),
        "std_acc": np.std(uniform_scores),
        "genus": None,
        "fiedler": None,
        "scores": uniform_scores,
    }

    print(f"\n  {'Init Source':<25} {'Acc@1':>8} {'±':>3} {'Genus':>6} {'Fiedler':>8}")
    print(f"  {'─' * 55}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_acc']):
        g = str(r['genus']) if r['genus'] is not None else "n/a"
        f = f"{r['fiedler']:.2f}" if r['fiedler'] is not None else "n/a"
        print(f"  {name:<25} {r['mean_acc']:>7.3f}  ±{r['std_acc']:>5.3f} {g:>6} {f:>8}")

    # Statistical test: is triangle_strong_CF > xavier at same arch?
    if "triangle_strong_CF" in results:
        try:
            from scipy.stats import mannwhitneyu
            t_scores = results["triangle_strong_CF"]["scores"]
            x_scores = results["xavier_baseline"]["scores"]
            stat, pval = mannwhitneyu(t_scores, x_scores, alternative='greater')
            print(f"\n  Mann-Whitney U test (triangle_strong_CF > xavier):")
            print(f"  U={stat:.1f}, p={pval:.4f} {'*** SIGNIFICANT' if pval < 0.05 else '(not significant)'}")
        except ImportError:
            print("\n  (scipy.stats not available, skipping significance test)")
        except Exception as e:
            print(f"\n  (Statistical test failed: {e})")

    return results


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: TASK TRANSFER
# ═══════════════════════════════════════════════════════════════════

class ArithmeticTask:
    """Simple addition: given two encoded digits, predict their sum mod 10."""
    def __init__(self):
        self.input_size = 20  # two one-hot digits (10 each)
        self.output_size = 10
        X, y = [], []
        for a in range(10):
            for b in range(10):
                x = np.zeros(20)
                x[a] = 1.0
                x[10 + b] = 1.0
                X.append(x)
                y.append((a + b) % 10)
        self.X = np.array(X)
        self.y = np.array(y)
        self.sample_weights = np.ones(len(y)) / len(y)
        self.vocab_size = 10

    def train_epoch_fast(self, network, lr=0.01, batch_size=32):
        return _generic_train_epoch(self, network, lr, batch_size)

    def evaluate(self, network):
        return _generic_evaluate(self, network)


class BigramTask:
    """Character bigram prediction: given char, predict most likely next char."""
    def __init__(self):
        # Simple bigram statistics from common English patterns
        self.input_size = 26  # one-hot letter
        self.output_size = 26
        pairs = [
            ('t', 'h'), ('h', 'e'), ('i', 'n'), ('e', 'r'), ('a', 'n'),
            ('r', 'e'), ('o', 'n'), ('e', 's'), ('s', 't'), ('e', 'n'),
            ('a', 't'), ('t', 'o'), ('n', 't'), ('o', 'r'), ('i', 's'),
            ('e', 'd'), ('a', 'r'), ('t', 'i'), ('n', 'g'), ('a', 'l'),
            ('i', 't'), ('o', 'f'), ('a', 's'), ('h', 'a'), ('n', 'd'),
            ('l', 'e'), ('s', 'e'), ('o', 'u'), ('c', 'o'), ('d', 'e'),
            ('t', 'a'), ('l', 'l'), ('u', 'r'), ('c', 'h'), ('w', 'a'),
            ('w', 'h'), ('m', 'a'), ('p', 'r'), ('f', 'o'), ('b', 'e'),
        ]
        X, y, w = [], [], []
        for c1, c2 in pairs:
            x = np.zeros(26)
            x[ord(c1) - ord('a')] = 1.0
            X.append(x)
            y.append(ord(c2) - ord('a'))
            w.append(1.0)
        # Add some random noise pairs
        rng = np.random.RandomState(42)
        for _ in range(60):
            a, b = rng.randint(0, 26, 2)
            x = np.zeros(26)
            x[a] = 1.0
            X.append(x)
            y.append(b)
            w.append(0.3)

        self.X = np.array(X)
        self.y = np.array(y)
        self.sample_weights = np.array(w)
        self.sample_weights /= self.sample_weights.sum()
        self.vocab_size = 26

    def train_epoch_fast(self, network, lr=0.01, batch_size=32):
        return _generic_train_epoch(self, network, lr, batch_size)

    def evaluate(self, network):
        return _generic_evaluate(self, network)


class XORClassificationTask:
    """Multi-dimensional XOR — requires compositional reasoning."""
    def __init__(self, n_bits=4, n_samples=200):
        rng = np.random.RandomState(42)
        self.input_size = n_bits
        self.output_size = 2  # binary classification
        self.X = rng.randint(0, 2, (n_samples, n_bits)).astype(float)
        # Label = XOR of all bits
        self.y = (self.X.sum(axis=1) % 2).astype(int)
        self.sample_weights = np.ones(n_samples) / n_samples
        self.vocab_size = 2

    def train_epoch_fast(self, network, lr=0.01, batch_size=32):
        return _generic_train_epoch(self, network, lr, batch_size)

    def evaluate(self, network):
        return _generic_evaluate(self, network)


class SequenceSortTask:
    """Given 4 numbers, predict the rank of the first number."""
    def __init__(self, n_samples=300):
        rng = np.random.RandomState(42)
        self.input_size = 4
        self.output_size = 4  # rank 0-3
        X, y = [], []
        for _ in range(n_samples):
            vals = rng.rand(4)
            rank = np.argsort(np.argsort(vals))[0]  # rank of first element
            X.append(vals)
            y.append(rank)
        self.X = np.array(X)
        self.y = np.array(y)
        self.sample_weights = np.ones(n_samples) / n_samples
        self.vocab_size = 4

    def train_epoch_fast(self, network, lr=0.01, batch_size=32):
        return _generic_train_epoch(self, network, lr, batch_size)

    def evaluate(self, network):
        return _generic_evaluate(self, network)


def _generic_train_epoch(task, network, lr, batch_size):
    """Generic training epoch with analytical backprop."""
    n = len(task.X)
    indices = np.random.permutation(n)
    total_loss, n_batches = 0, 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        X_b, y_b = task.X[batch_idx], task.y[batch_idx]
        bs = len(batch_idx)

        h = X_b
        activations, pre_acts = [h], []
        for i in range(network.n_layers):
            z = h @ network.weights[i] + network.biases[i]

            for (src, dst) in network.skip_connections:
                if dst == i and src < len(activations):
                    skip_val = activations[src]
                    min_d = min(skip_val.shape[-1], z.shape[-1])
                    z[..., :min_d] += skip_val[..., :min_d] * 0.1

            pre_acts.append(z)
            if i < network.n_layers - 1:
                h = np.maximum(0, z)
            else:
                z_s = z - np.max(z, axis=-1, keepdims=True)
                exp_z = np.exp(z_s)
                h = exp_z / (exp_z.sum(axis=-1, keepdims=True) + 1e-10)
            activations.append(h)

        probs = np.clip(activations[-1], 1e-10, 1.0)
        loss = -np.mean(np.log(probs[np.arange(bs), y_b]))
        total_loss += loss

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(bs), y_b] = 1.0
        delta = (probs - one_hot) / bs

        for i in range(network.n_layers - 1, -1, -1):
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)
            # Gradient clipping
            gn = np.linalg.norm(dW)
            if gn > 5.0:
                dW = dW * 5.0 / gn
            network.weights[i] -= lr * dW
            network.biases[i] -= lr * db
            if i > 0:
                delta = delta @ network.weights[i].T
                delta *= (pre_acts[i - 1] > 0).astype(float)
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _generic_evaluate(task, network):
    """Generic evaluation: acc@1, acc@3, mrr."""
    probs = network.forward(task.X)
    preds = np.argmax(probs, axis=1)
    acc1 = np.mean(preds == task.y)

    top3 = np.argsort(probs, axis=1)[:, -3:]
    acc3 = np.mean([task.y[i] in top3[i] for i in range(len(task.y))])

    mrr = 0
    for i in range(len(task.y)):
        ranks = np.argsort(-probs[i])
        rank_of_correct = np.where(ranks == task.y[i])[0][0] + 1
        mrr += 1.0 / rank_of_correct
    mrr /= len(task.y)

    return {'acc1': acc1, 'acc3': acc3, 'mrr': mrr}


def run_transfer(n_seeds=5, n_epochs=30):
    """
    Run the same topologies on 5 different tasks.
    If genus-1 dominates across ALL tasks, the finding generalizes.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: TASK TRANSFER")
    print("  Same topologies, 5 different tasks")
    print("  Using FULL nucleation (topology → architecture + spectral init)")
    print("=" * 70)

    tasks = {
        "T9 (617w)": T9Task(),
        "Arithmetic (mod10)": ArithmeticTask(),
        "Bigrams": BigramTask(),
        "XOR (4-bit)": XORClassificationTask(),
        "Rank sorting": SequenceSortTask(),
    }

    candidates = make_t9_candidates()
    selected_names = [
        "triangle_strong_CF", "triangle", "chain_linear",
        "chain_thick_CF", "disconnected_X", "tetra_meta",
    ]
    selected = [g for g in candidates if g.name in selected_names]

    all_results = {}
    for task_name, task in tasks.items():
        print(f"\n  Task: {task_name} (input={task.input_size}, output={task.output_size}, n={len(task.X)})")
        task_results = {}

        for graph in selected:
            topo = TopologicalFeatures(graph)
            scores = []
            for seed in range(n_seeds):
                np.random.seed(seed * 1000 + hash(task_name) % 10000)
                try:
                    net = nucleate_from_topology(topo, task.input_size, task.output_size, base_hidden=32)
                    metrics, _ = train_and_eval(task, net, n_epochs, lr=0.05)
                    scores.append(metrics['acc1'])
                except Exception:
                    scores.append(0.0)
            task_results[graph.name] = {
                "mean": np.mean(scores), "std": np.std(scores),
                "genus": graph.cycle_rank,
            }

        # Xavier baseline
        xavier_scores = []
        for seed in range(n_seeds):
            np.random.seed(seed * 2000)
            arch = [task.input_size, 32, 32, task.output_size]
            net = xavier_init_fixed_arch(arch, seed=seed)
            metrics, _ = train_and_eval(task, net, n_epochs, lr=0.05)
            xavier_scores.append(metrics['acc1'])
        task_results["xavier_baseline"] = {
            "mean": np.mean(xavier_scores), "std": np.std(xavier_scores),
            "genus": None,
        }

        all_results[task_name] = task_results

    # Print transfer matrix
    print(f"\n\n  {'':>25}", end="")
    for task_name in tasks:
        short = task_name[:12]
        print(f" {short:>13}", end="")
    print()
    print(f"  {'─' * (25 + 14 * len(tasks))}")

    shape_names = selected_names + ["xavier_baseline"]
    for sn in shape_names:
        genus_str = ""
        for tn, tr in all_results.items():
            if sn in tr and tr[sn]['genus'] is not None:
                genus_str = f" g={tr[sn]['genus']}"
                break
        print(f"  {sn:<25}", end="")
        for task_name in tasks:
            r = all_results[task_name].get(sn)
            if r:
                print(f" {r['mean']:>12.3f}", end="")
            else:
                print(f" {'---':>12}", end="")
        print(genus_str)

    # Compute genus-1 vs genus-0 advantage across tasks
    print(f"\n  Genus-1 vs Genus-0 advantage per task:")
    genus1_wins = 0
    for task_name in tasks:
        tr = all_results[task_name]
        g1 = [v['mean'] for k, v in tr.items() if v.get('genus') == 1]
        g0 = [v['mean'] for k, v in tr.items() if v.get('genus') == 0]
        if g1 and g0:
            g1_best = max(g1)
            g0_best = max(g0)
            advantage = g1_best - g0_best
            print(f"    {task_name:<20} genus1_best={g1_best:.3f}  genus0_best={g0_best:.3f}  Δ={advantage:+.3f} {'✓' if advantage > 0 else '✗'}")
            if advantage > 0:
                genus1_wins += 1

    print(f"\n  Genus-1 wins: {genus1_wins}/{len(tasks)} tasks")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: BLIND SEARCH
# ═══════════════════════════════════════════════════════════════════

def random_graph(n_nodes, edge_prob, weight_range, rng):
    """Generate a random capability graph with Erdos-Renyi + random weights."""
    h = np.sqrt(3) / 2
    # Random 3D positions
    positions = rng.randn(n_nodes, 3).tolist()

    # Labels
    labels = [f"N{i}" for i in range(n_nodes)]
    nodes = list(zip(labels, positions))

    edges = []
    edge_weights = {}
    idx = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.rand() < edge_prob:
                edges.append((i, j))
                edge_weights[idx] = weight_range[0] + rng.rand() * (weight_range[1] - weight_range[0])
                idx += 1

    return CapabilityGraph(
        f"random_{n_nodes}n_{len(edges)}e",
        nodes, edges, edge_weights
    )


def run_blind_search(n_random=200, n_seeds=3, n_epochs=20):
    """
    Generate hundreds of random graphs, evaluate each on T9.
    Then analyze: what topological properties predict success?
    Does the optimizer independently discover genus-1 + asymmetric weights?
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: BLIND SEARCH")
    print(f"  {n_random} random capability graphs evaluated on T9")
    print("  Question: do the best random graphs independently show genus=1?")
    print("=" * 70)

    task = T9Task()
    rng = np.random.RandomState(42)

    search_results = []
    for trial in range(n_random):
        n_nodes = rng.randint(2, 6)
        edge_prob = rng.uniform(0.2, 1.0)
        weight_range = (rng.uniform(0.1, 1.0), rng.uniform(1.0, 3.0))

        graph = random_graph(n_nodes, edge_prob, weight_range, rng)
        topo = TopologicalFeatures(graph)

        scores = []
        for seed in range(n_seeds):
            np.random.seed(seed * 500 + trial)
            try:
                net = nucleate_from_topology(topo, task.input_size, task.output_size, base_hidden=32)
                metrics, _ = train_and_eval(task, net, n_epochs, lr=0.05)
                scores.append(metrics['acc1'])
            except Exception:
                scores.append(0.0)

        mean_acc = np.mean(scores)
        search_results.append({
            "trial": trial,
            "n_nodes": n_nodes,
            "n_edges": len(graph.edges),
            "genus": graph.cycle_rank,
            "fiedler": topo.fiedler,
            "spectral_gap": topo.spectral_gap,
            "mean_degree": topo.mean_degree,
            "weight_var": topo.weight_variance,
            "acc": mean_acc,
        })

        if trial % 50 == 0:
            print(f"  ... {trial}/{n_random} evaluated")

    # Analyze
    df = search_results
    print(f"\n  Evaluated {len(df)} random graphs")

    # Top 20
    df_sorted = sorted(df, key=lambda x: -x['acc'])
    print(f"\n  TOP 20 random graphs:")
    print(f"  {'Rank':<5} {'Acc':>6} {'Nodes':>6} {'Edges':>6} {'Genus':>6} {'Fiedler':>8} {'WeightVar':>10}")
    print(f"  {'─' * 55}")
    for i, r in enumerate(df_sorted[:20]):
        print(f"  {i+1:<5} {r['acc']:>5.3f} {r['n_nodes']:>6} {r['n_edges']:>6} "
              f"{r['genus']:>6} {r['fiedler']:>8.3f} {r['weight_var']:>10.3f}")

    # Genus distribution in top vs bottom
    top_n = min(30, len(df) // 5)
    top = df_sorted[:top_n]
    bottom = df_sorted[-top_n:]

    genus_top = [r['genus'] for r in top]
    genus_bottom = [r['genus'] for r in bottom]

    print(f"\n  Genus distribution (top {top_n} vs bottom {top_n}):")
    for g in range(max(max(genus_top, default=0), max(genus_bottom, default=0)) + 1):
        tc = genus_top.count(g)
        bc = genus_bottom.count(g)
        print(f"    genus={g}: top={tc}/{top_n} ({tc/top_n*100:.0f}%)  bottom={bc}/{top_n} ({bc/top_n*100:.0f}%)")

    # Mean accuracy by genus
    genus_groups = {}
    for r in df:
        g = r['genus']
        if g not in genus_groups:
            genus_groups[g] = []
        genus_groups[g].append(r['acc'])

    print(f"\n  Mean accuracy by genus:")
    for g in sorted(genus_groups.keys()):
        accs = genus_groups[g]
        print(f"    genus={g}: mean={np.mean(accs):.3f} ± {np.std(accs):.3f}  (n={len(accs)})")

    # Correlation analysis
    accs = np.array([r['acc'] for r in df])
    print(f"\n  Correlation with accuracy:")
    for feature in ['genus', 'fiedler', 'spectral_gap', 'n_nodes', 'n_edges', 'mean_degree', 'weight_var']:
        vals = np.array([r[feature] for r in df])
        if np.std(vals) > 0:
            corr = np.corrcoef(vals, accs)[0, 1]
            print(f"    {feature:<15} r={corr:+.3f}")

    # Best random vs hand-designed
    best_random = df_sorted[0]
    print(f"\n  Best random graph: acc={best_random['acc']:.3f}")
    print(f"    nodes={best_random['n_nodes']} edges={best_random['n_edges']} "
          f"genus={best_random['genus']} fiedler={best_random['fiedler']:.3f}")

    return search_results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  IMT RIGOR SUITE".center(68) + "║")
    print("║" + "  Three experiments to validate or kill the thesis".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    t0 = time.time()

    # Experiment 1: Does spectrum help independent of architecture?
    ablation = run_ablation(n_seeds=8, n_epochs=30)

    # Experiment 2: Does topology generalize across tasks?
    transfer = run_transfer(n_seeds=5, n_epochs=30)

    # Experiment 3: Does blind search rediscover genus-1?
    blind = run_blind_search(n_random=300, n_seeds=3, n_epochs=20)

    elapsed = time.time() - t0

    # ═══ VERDICT ═══
    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Check 1: Ablation — does spectral init help at fixed arch?
    if ablation:
        tri_acc = ablation.get("triangle_strong_CF", {}).get("mean_acc", 0)
        xav_acc = ablation.get("xavier_baseline", {}).get("mean_acc", 0)
        uni_acc = ablation.get("uniform_svd", {}).get("mean_acc", 0)
        spectral_helps = tri_acc > xav_acc * 1.05
        print(f"\n  1. ABLATION: Spectral init at fixed arch")
        print(f"     triangle_strong_CF: {tri_acc:.3f}")
        print(f"     xavier_baseline:    {xav_acc:.3f}")
        print(f"     uniform_svd:        {uni_acc:.3f}")
        print(f"     → Spectral signal independent of arch: {'YES ✓' if spectral_helps else 'NO ✗'}")

    # Check 2: Transfer — genus-1 wins across tasks?
    print(f"\n  2. TRANSFER: Genus-1 advantage across tasks")
    genus1_tasks = 0
    for task_name, tr in transfer.items():
        g1 = [v['mean'] for k, v in tr.items() if v.get('genus') == 1]
        g0 = [v['mean'] for k, v in tr.items() if v.get('genus') == 0]
        xav = tr.get("xavier_baseline", {}).get("mean", 0)
        if g1 and g0:
            if max(g1) > max(g0):
                genus1_tasks += 1
    print(f"     Genus-1 beats genus-0 in {genus1_tasks}/{len(transfer)} tasks")
    print(f"     → Topology generalizes: {'YES ✓' if genus1_tasks >= 3 else 'PARTIAL' if genus1_tasks >= 2 else 'NO ✗'}")

    # Check 3: Blind search — does optimization find genus-1?
    print(f"\n  3. BLIND SEARCH: Does random search rediscover genus-1?")
    sorted_blind = sorted(blind, key=lambda x: -x['acc'])
    top20_genus = [r['genus'] for r in sorted_blind[:20]]
    genus1_in_top = top20_genus.count(1)
    genus_groups = {}
    for r in blind:
        g = r['genus']
        if g not in genus_groups:
            genus_groups[g] = []
        genus_groups[g].append(r['acc'])
    g1_mean = np.mean(genus_groups.get(1, [0]))
    g0_mean = np.mean(genus_groups.get(0, [0]))
    print(f"     Genus-1 in top 20: {genus1_in_top}/20")
    print(f"     Mean acc genus-1: {g1_mean:.3f} vs genus-0: {g0_mean:.3f}")
    print(f"     → Blind search confirms genus-1: {'YES ✓' if g1_mean > g0_mean * 1.1 else 'NO ✗'}")

    thesis_survives = spectral_helps and genus1_tasks >= 3 and g1_mean > g0_mean * 1.1
    print(f"\n  ══════════════════════════════════════════")
    print(f"  THESIS STATUS: {'SURVIVES ✓ ✓ ✓' if thesis_survives else 'NEEDS QUALIFICATION'}")
    print(f"  ══════════════════════════════════════════")
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save results
    output = {
        "ablation": {k: {kk: vv for kk, vv in v.items() if kk != 'scores'} for k, v in ablation.items()},
        "transfer": {k: {kk: {kkk: vvv for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in transfer.items()},
        "blind_search_summary": {
            "n_evaluated": len(blind),
            "best_acc": sorted_blind[0]['acc'],
            "best_genus": sorted_blind[0]['genus'],
            "genus1_in_top20": genus1_in_top,
            "mean_acc_by_genus": {str(g): float(np.mean(a)) for g, a in genus_groups.items()},
        },
        "thesis_survives": thesis_survives,
    }

    with open("rigor_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to rigor_results.json")


if __name__ == "__main__":
    main()
