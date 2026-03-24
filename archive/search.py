"""
search.py - IMT Shape Space Search
Tests candidate capability graphs for their ability to nucleate T9.
Compares nucleated networks against random baseline.
"""
import numpy as np
import time
from shapes import make_t9_candidates
from topology import TopologicalFeatures
from nucleate import nucleate_from_topology, nucleate_random_baseline
from t9_task import T9Task


class SearchResult:
    def __init__(self, name, topo, metrics, learning_curve, n_params):
        self.name = name
        self.topo = topo
        self.metrics = metrics          # final eval metrics
        self.learning_curve = learning_curve  # loss per epoch
        self.n_params = n_params
        # Composite score: weighted combination of metrics
        self.score = (
            metrics['acc1'] * 0.4 +
            metrics['acc3'] * 0.3 +
            metrics['mrr'] * 0.2 +
            metrics.get('ambiguity', 0) * 0.1
        )
        # Learning efficiency: how fast did it learn?
        if len(learning_curve) > 1:
            # Rate of loss decrease in first half of training
            mid = len(learning_curve) // 2
            early_drop = learning_curve[0] - learning_curve[mid]
            self.learning_speed = early_drop / max(learning_curve[0], 1e-6)
        else:
            self.learning_speed = 0


def run_search(n_epochs=30, n_seeds=3, verbose=True):
    """
    Main search: test all candidate shapes, compare to random baseline.
    Runs multiple random seeds for statistical robustness.
    """
    task = T9Task()
    candidates = make_t9_candidates()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  INVERSE MORPHOGENIC TRAINING — T9 NUCLEATION SEARCH")
        print(f"{'='*60}")
        print(f"  Vocabulary: {task.vocab_size} words")
        print(f"  Input dim:  {task.input_size}")
        print(f"  Training samples: {len(task.X)}")
        print(f"  Epochs: {n_epochs} | Seeds: {n_seeds}")
        print(f"  Candidate shapes: {len(candidates)}")
        print(f"{'='*60}\n")

    all_results = []

    # --- Test each candidate shape ---
    for ci, graph in enumerate(candidates):
        if verbose:
            print(f"\n[{ci+1}/{len(candidates)}] Shape: {graph.name}")

        topo = TopologicalFeatures(graph)
        if verbose:
            print(topo.summary())

        seed_scores = []
        best_result = None

        for seed in range(n_seeds):
            np.random.seed(seed * 1000 + ci)

            # Nucleate
            net = nucleate_from_topology(topo, task.input_size,
                                         task.output_size, base_hidden=32)

            # Train
            curve = []
            for epoch in range(n_epochs):
                loss = task.train_epoch_fast(net, lr=0.05, batch_size=32)
                curve.append(loss)

            # Evaluate
            metrics = task.evaluate(net)
            metrics['ambiguity'] = task.evaluate_ambiguity(net)

            result = SearchResult(graph.name, topo, metrics, curve,
                                  net.n_params())
            seed_scores.append(result.score)

            if best_result is None or result.score > best_result.score:
                best_result = result

        avg_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)
        best_result.avg_score = avg_score
        best_result.std_score = std_score
        all_results.append(best_result)

        if verbose:
            m = best_result.metrics
            print(f"  acc@1={m['acc1']:.3f}  acc@3={m['acc3']:.3f}  "
                  f"mrr={m['mrr']:.3f}  ambig={m['ambiguity']:.3f}")
            print(f"  score={avg_score:.4f} ± {std_score:.4f}  "
                  f"params={best_result.n_params}")

    # --- Random baseline ---
    if verbose:
        print(f"\n[baseline] Random Xavier init")

    baseline_scores = []
    best_baseline = None

    for seed in range(n_seeds):
        np.random.seed(seed * 10000)
        net = nucleate_random_baseline(task.input_size, task.output_size,
                                        hidden_size=32, n_hidden=2)
        curve = []
        for epoch in range(n_epochs):
            loss = task.train_epoch_fast(net, lr=0.05, batch_size=32)
            curve.append(loss)

        metrics = task.evaluate(net)
        metrics['ambiguity'] = task.evaluate_ambiguity(net)
        result = SearchResult("random_baseline", None, metrics, curve,
                              net.n_params())
        baseline_scores.append(result.score)
        if best_baseline is None or result.score > best_baseline.score:
            best_baseline = result

    best_baseline.avg_score = np.mean(baseline_scores)
    best_baseline.std_score = np.std(baseline_scores)

    if verbose:
        m = best_baseline.metrics
        print(f"  acc@1={m['acc1']:.3f}  acc@3={m['acc3']:.3f}  "
              f"mrr={m['mrr']:.3f}  ambig={m['ambiguity']:.3f}")
        print(f"  score={best_baseline.avg_score:.4f} ± "
              f"{best_baseline.std_score:.4f}  params={best_baseline.n_params}")

    return all_results, best_baseline
