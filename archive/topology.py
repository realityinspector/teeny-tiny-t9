"""
topology.py - Topological Feature Extraction for IMT
Extracts spectral and topological invariants from capability graphs and their meshes.
These invariants become the constraints for NN nucleation.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def graph_laplacian(adj_matrix):
    """Compute normalized graph Laplacian from adjacency matrix."""
    D = np.diag(adj_matrix.sum(axis=1))
    L = D - adj_matrix
    # Normalize
    d_inv_sqrt = np.zeros_like(D)
    diag = np.diag(D)
    mask = diag > 0
    d_inv_sqrt[mask, mask] = 1.0 / np.sqrt(diag[mask])
    # Symmetric normalized Laplacian
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    return L, L_norm


def mesh_laplacian_spectrum(vertices, faces, n_eigenvalues=20):
    """
    Compute the Laplacian spectrum of a triangle mesh.
    Returns eigenvalues that encode the shape's geometry and topology.
    """
    n = len(vertices)
    if n < 4:
        return np.zeros(min(n, n_eigenvalues))

    # Build adjacency from faces
    rows, cols, vals = [], [], []
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            rows.extend([a, b])
            cols.extend([b, a])
            # Cotangent weight approximation: use edge length inverse
            d = np.linalg.norm(vertices[a] - vertices[b])
            w = 1.0 / max(d, 1e-6)
            vals.extend([w, w])

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    D = sparse.diags(np.array(A.sum(axis=1)).flatten())
    L = D - A

    k = min(n_eigenvalues, n - 2)
    if k < 1:
        return np.zeros(1)

    try:
        eigenvalues = eigsh(L, k=k, sigma=0, which='LM', return_eigenvectors=False)
        eigenvalues = np.sort(np.abs(eigenvalues))
    except Exception:
        # Fallback: dense eigensolve for small meshes
        try:
            eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(L.toarray())))[:k]
        except Exception:
            eigenvalues = np.zeros(k)

    return eigenvalues


class TopologicalFeatures:
    """Extracted topological features of a capability graph / shape."""

    def __init__(self, graph):
        self.graph = graph
        self._extract()

    def _extract(self):
        A = self.graph.adjacency_matrix
        n = self.graph.n

        # Graph-level features
        self.n_nodes = n
        self.n_edges = len(self.graph.edges)
        self.cycle_rank = self.graph.cycle_rank  # = genus for thickened graph

        # Betti numbers (for closed orientable surface)
        # Thickened graph: genus = cycle_rank
        self.betti_0 = 1  # assume connected (simplification)
        self.betti_1 = 2 * self.cycle_rank  # for orientable surface of genus g
        self.betti_2 = 1  # closed surface

        # Euler characteristic
        self.euler_char = 2 - 2 * self.cycle_rank

        # Graph Laplacian spectrum
        if n > 1:
            L, L_norm = graph_laplacian(A)
            self.graph_eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
            # Spectral gap: second smallest eigenvalue of L
            raw_eigs = np.sort(np.linalg.eigvalsh(L))
            self.spectral_gap = raw_eigs[1] if len(raw_eigs) > 1 else 0.0
            # Algebraic connectivity (Fiedler value)
            self.fiedler = raw_eigs[1] if len(raw_eigs) > 1 else 0.0
        else:
            self.graph_eigenvalues = np.array([0.0])
            self.spectral_gap = 0.0
            self.fiedler = 0.0

        # Degree distribution
        if n > 0 and len(A) > 0:
            degrees = A.sum(axis=1)
            self.mean_degree = np.mean(degrees)
            self.max_degree = np.max(degrees)
            self.degree_variance = np.var(degrees)
        else:
            self.mean_degree = 0
            self.max_degree = 0
            self.degree_variance = 0

        # Edge weight statistics
        weights = [self.graph.edge_weights.get(i, 1.0)
                    for i in range(self.n_edges)]
        if weights:
            self.mean_weight = np.mean(weights)
            self.weight_variance = np.var(weights)
        else:
            self.mean_weight = 0
            self.weight_variance = 0

        # Mesh spectrum (compute lazily)
        self._mesh_spectrum = None

    def compute_mesh_spectrum(self, n_eigs=20):
        """Compute spectrum of the thickened 3D mesh."""
        if self._mesh_spectrum is None:
            verts, faces = self.graph.to_mesh(res=12)  # lower res for speed
            if len(verts) > 3:
                self._mesh_spectrum = mesh_laplacian_spectrum(
                    verts, faces, n_eigs)
            else:
                self._mesh_spectrum = np.zeros(n_eigs)
        return self._mesh_spectrum

    def feature_vector(self):
        """Compact feature vector for this topology."""
        return np.array([
            self.n_nodes,
            self.n_edges,
            self.cycle_rank,
            self.spectral_gap,
            self.fiedler,
            self.mean_degree,
            self.max_degree,
            self.degree_variance,
            self.euler_char,
            self.mean_weight,
            self.weight_variance,
        ])

    def summary(self):
        return (
            f"  nodes={self.n_nodes} edges={self.n_edges} "
            f"genus={self.cycle_rank} euler={self.euler_char}\n"
            f"  spectral_gap={self.spectral_gap:.4f} "
            f"fiedler={self.fiedler:.4f} "
            f"mean_degree={self.mean_degree:.2f}"
        )
