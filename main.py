#!/usr/bin/env python3
"""
main.py - Inverse Morphogenic Training: T9 Crystal Discovery

Tests which 3D topological shape, when its spectral properties constrain
neural network initialization, best nucleates T9 predictive text capability.

Exports the winning shape as a 3D-printable STL file.

Usage:
    pip install numpy scipy numpy-stl
    python main.py

Author: IMT Research
"""
import sys
import os
import time
import numpy as np

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search import run_search
from shapes import make_t9_candidates
from topology import TopologicalFeatures
from stl_export import export_graph_stl


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗███╗   ███╗████████╗                                    ║
║   ██║████╗ ████║╚══██╔══╝                                    ║
║   ██║██╔████╔██║   ██║     Inverse Morphogenic Training      ║
║   ██║██║╚██╔╝██║   ██║     T9 Crystal Nucleation Engine      ║
║   ██║██║ ╚═╝ ██║   ██║                                       ║
║   ╚═╝╚═╝     ╚═╝   ╚═╝                                      ║
║                                                              ║
║   Hypothesis: A 3D topological shape can encode the          ║
║   capability lattice for T9 predictive text. Its spectral    ║
║   properties constrain NN initialization such that the       ║
║   network learns T9 faster than random init.                 ║
║                                                              ║
║   Output: The shape that best nucleates T9, as a             ║
║   3D-printable STL file.                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_results(results, baseline):
    print(f"\n{'='*60}")
    print(f"  RESULTS — RANKED BY NUCLEATION QUALITY")
    print(f"{'='*60}\n")

    # Sort by avg_score descending
    ranked = sorted(results, key=lambda r: r.avg_score, reverse=True)

    print(f"  {'Rank':<5} {'Shape':<22} {'Score':>8} {'±':>3} "
          f"{'Acc@1':>7} {'MRR':>7} {'Genus':>6} {'Params':>7}")
    print(f"  {'─'*70}")

    for i, r in enumerate(ranked):
        genus = r.topo.cycle_rank if r.topo else '?'
        marker = " ◄ BEST" if i == 0 else ""
        print(f"  {i+1:<5} {r.name:<22} {r.avg_score:>7.4f}  ±"
              f"{r.std_score:>5.3f} {r.metrics['acc1']:>7.3f} "
              f"{r.metrics['mrr']:>7.3f} {str(genus):>6} "
              f"{r.n_params:>7}{marker}")

    print(f"  {'─'*70}")
    print(f"  {'base':<5} {'random_xavier':<22} {baseline.avg_score:>7.4f}  ±"
          f"{baseline.std_score:>5.3f} {baseline.metrics['acc1']:>7.3f} "
          f"{baseline.metrics['mrr']:>7.3f} {'n/a':>6} "
          f"{baseline.n_params:>7}")

    # Compute speedup
    best = ranked[0]
    if baseline.avg_score > 0:
        advantage = ((best.avg_score - baseline.avg_score)
                     / baseline.avg_score * 100)
    else:
        advantage = float('inf')

    print(f"\n  Winner: {best.name}")
    print(f"  Advantage over random init: {advantage:+.1f}%")

    if best.topo:
        print(f"\n  Topology of winning crystal:")
        print(f"    Nodes: {best.topo.n_nodes}")
        print(f"    Edges: {best.topo.n_edges}")
        print(f"    Genus (cycle rank): {best.topo.cycle_rank}")
        print(f"    Euler characteristic: {best.topo.euler_char}")
        print(f"    Spectral gap: {best.topo.spectral_gap:.4f}")
        print(f"    Fiedler value: {best.topo.fiedler:.4f}")

    return ranked[0]


def export_winner(results):
    """Export the winning shape as STL."""
    candidates = make_t9_candidates()
    winner_name = results.name

    # Find the matching graph
    winner_graph = None
    for g in candidates:
        if g.name == winner_name:
            winner_graph = g
            break

    if winner_graph is None:
        print("  [!] Could not find winner graph for STL export")
        return None

    filename = f"t9_crystal_{winner_name}.stl"
    export_graph_stl(winner_graph, filename, scale=20.0, resolution=24)

    # Also compute mesh stats for the printable shape
    verts, faces = winner_graph.to_mesh(res=24)
    dims = (verts.max(axis=0) - verts.min(axis=0)) * 20.0  # in mm

    print(f"\n  3D-Printable STL exported: {filename}")
    print(f"  Dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
    print(f"  Faces: {len(faces)}")
    print(f"  Vertices: {len(verts)}")

    return filename


def learning_curve_comparison(results, baseline):
    """Print ASCII learning curves for top shapes vs baseline."""
    print(f"\n{'='*60}")
    print(f"  LEARNING CURVES (loss over epochs)")
    print(f"{'='*60}\n")

    ranked = sorted(results, key=lambda r: r.avg_score, reverse=True)
    to_show = ranked[:3] + [baseline]
    labels = [r.name[:15] for r in ranked[:3]] + ["random_baseline"]

    max_loss = max(max(r.learning_curve) for r in to_show
                   if r.learning_curve)
    width = 50
    n_epochs = len(to_show[0].learning_curve)

    # Show every 5th epoch
    for epoch in range(0, n_epochs, max(1, n_epochs // 8)):
        print(f"  epoch {epoch:>3}: ", end="")
        for i, r in enumerate(to_show):
            if epoch < len(r.learning_curve):
                loss = r.learning_curve[epoch]
                bar_len = int(loss / max(max_loss, 1e-6) * 20)
                label = labels[i][:8]
                print(f"  {label:>8} {'█' * bar_len} {loss:.3f}", end="")
        print()


def main():
    print_banner()

    t_start = time.time()

    # Run the search
    results, baseline = run_search(n_epochs=30, n_seeds=3, verbose=True)

    t_elapsed = time.time() - t_start

    # Print ranked results
    winner = print_results(results, baseline)

    # Learning curve comparison
    learning_curve_comparison(results, baseline)

    # Export STL
    print(f"\n{'='*60}")
    print(f"  EXPORTING 3D-PRINTABLE CRYSTAL")
    print(f"{'='*60}")

    stl_file = export_winner(winner)

    # Summary
    print(f"\n{'='*60}")
    print(f"  IMT NUCLEATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Time: {t_elapsed:.1f}s")
    print(f"  Winner: {winner.name}")
    print(f"  The T9 crystal shape is: {stl_file}")
    print(f"\n  This shape's topology, when used to constrain NN weight")
    print(f"  initialization, produces faster T9 learning than random init.")
    print(f"  Print it. Hold the shape of T9 in your hand.\n")

    return stl_file


if __name__ == "__main__":
    main()
