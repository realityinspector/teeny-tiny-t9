"""
nucleate.py - IMT Nucleation Engine
Maps topological features of a capability graph to neural network weight constraints.
The topology determines: architecture (depth, width, skip connections) and
weight initialization (singular value spectrum from mesh Laplacian).
"""
import numpy as np
from topology import TopologicalFeatures


class NucleatedNetwork:
    """A tiny neural network nucleated from topological constraints."""

    def __init__(self, layer_sizes, weights, biases, skip_connections=None):
        self.layer_sizes = layer_sizes
        self.weights = weights  # list of weight matrices
        self.biases = biases
        self.skip_connections = skip_connections or []
        self.n_layers = len(weights)

    def forward(self, x):
        """Forward pass with ReLU hidden layers, softmax output."""
        activations = [x]
        h = x
        for i in range(self.n_layers):
            z = h @ self.weights[i] + self.biases[i]
            z = np.clip(z, -50, 50)  # prevent overflow
            # Add skip connections
            for (src, dst) in self.skip_connections:
                if dst == i and src < len(activations):
                    # Project if sizes differ
                    skip_val = activations[src]
                    if skip_val.shape[-1] != z.shape[-1]:
                        # Truncate or pad
                        min_d = min(skip_val.shape[-1], z.shape[-1])
                        z[..., :min_d] += skip_val[..., :min_d] * 0.1
                    else:
                        z += skip_val * 0.1

            if i < self.n_layers - 1:
                h = np.maximum(0, z)  # ReLU
            else:
                # Softmax
                z_shifted = z - np.max(z, axis=-1, keepdims=True)
                exp_z = np.exp(z_shifted)
                h = exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-10)

            activations.append(h)
        return h

    def get_params(self):
        """Flatten all parameters into a single vector."""
        params = []
        for w in self.weights:
            params.append(w.ravel())
        for b in self.biases:
            params.append(b.ravel())
        return np.concatenate(params)

    def set_params(self, params):
        """Restore parameters from flat vector."""
        idx = 0
        for i, w in enumerate(self.weights):
            size = w.size
            self.weights[i] = params[idx:idx + size].reshape(w.shape)
            idx += size
        for i, b in enumerate(self.biases):
            size = b.size
            self.biases[i] = params[idx:idx + size].reshape(b.shape)
            idx += size

    def n_params(self):
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


def nucleate_from_topology(topo_features, input_size, output_size,
                           base_hidden=32):
    """
    Core IMT nucleation: topology -> constrained NN initialization.

    Mapping rules:
    - cycle_rank (genus) -> number of hidden layers + skip connections
    - spectral_gap -> hidden layer width scaling
    - fiedler value -> connectivity density of skip connections
    - graph eigenvalue distribution -> singular value distribution of weights
    - edge weights -> relative layer widths
    """
    tf = topo_features

    # --- Architecture from topology ---

    # Depth: 1 base layer + 1 per genus (cycles need depth for feedback)
    n_hidden = max(1, 1 + tf.cycle_rank)

    # Width: scaled by spectral gap (higher gap = tighter coupling = wider layers)
    width_scale = 1.0 + tf.spectral_gap
    hidden_size = max(16, int(base_hidden * width_scale))

    # Layer sizes
    layer_sizes = [input_size]
    for _ in range(n_hidden):
        layer_sizes.append(hidden_size)
    layer_sizes.append(output_size)

    # Skip connections from cycles
    skip_connections = []
    if tf.cycle_rank >= 1:
        # Each cycle adds a skip connection bridging one layer
        for c in range(min(tf.cycle_rank, n_hidden - 1)):
            skip_connections.append((c, c + 2))

    # --- Weight initialization from spectrum ---

    weights = []
    biases = []

    # Target singular value distribution from graph eigenvalues
    target_spectrum = tf.graph_eigenvalues.copy()
    target_spectrum = np.abs(target_spectrum)
    if target_spectrum.max() > 0:
        target_spectrum = target_spectrum / target_spectrum.max()

    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]

        # Generate random matrix
        W = np.random.randn(fan_in, fan_out)

        # SVD
        U, s, Vt = np.linalg.svd(W, full_matrices=False)

        # Replace singular values with topology-constrained spectrum
        n_sv = len(s)
        if len(target_spectrum) > 0:
            # Interpolate target spectrum to match n_sv
            target_interp = np.interp(
                np.linspace(0, 1, n_sv),
                np.linspace(0, 1, len(target_spectrum)),
                target_spectrum
            )
            # Scale: use Xavier-like scaling but shaped by topology
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            shaped_sv = target_interp * scale * n_sv

            # Ensure minimum singular value (avoid dead layers)
            shaped_sv = np.maximum(shaped_sv, scale * 0.1)
        else:
            # No topology info: fall back to Xavier
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            shaped_sv = np.ones(n_sv) * scale

        # Reconstruct weight matrix with shaped spectrum
        W_shaped = U[:, :n_sv] @ np.diag(shaped_sv) @ Vt[:n_sv, :]
        weights.append(W_shaped)

        # Bias: small, shaped by degree distribution
        b = np.zeros(fan_out)
        if tf.mean_degree > 0:
            b += np.random.randn(fan_out) * 0.01 * tf.degree_variance
        biases.append(b)

    return NucleatedNetwork(layer_sizes, weights, biases, skip_connections)


def nucleate_random_baseline(input_size, output_size, hidden_size=32,
                              n_hidden=2):
    """Standard Xavier init baseline for comparison."""
    layer_sizes = [input_size] + [hidden_size] * n_hidden + [output_size]
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        W = np.random.randn(fan_in, fan_out) * scale
        weights.append(W)
        biases.append(np.zeros(fan_out))

    return NucleatedNetwork(layer_sizes, weights, biases)
