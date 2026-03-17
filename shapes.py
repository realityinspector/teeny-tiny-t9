"""
shapes.py - IMT Shape Generation
Generates 3D meshes from capability graphs by thickening graph structures.
Nodes -> spheres, edges -> tubes. Surface topology encodes capability structure.
"""
import numpy as np


def generate_sphere_mesh(center, radius, resolution=16):
    u = np.linspace(0, 2 * np.pi, resolution + 1)[:-1]
    v = np.linspace(0.15, np.pi - 0.15, resolution // 2)

    verts = []
    verts.append(center + np.array([0, 0, radius]))

    for vi in v:
        for ui in u:
            x = radius * np.sin(vi) * np.cos(ui)
            y = radius * np.sin(vi) * np.sin(ui)
            z = radius * np.cos(vi)
            verts.append(center + np.array([x, y, z]))

    verts.append(center + np.array([0, 0, -radius]))
    pole_bot = len(verts) - 1
    verts = np.array(verts)

    faces = []
    n_u = resolution
    n_v = resolution // 2

    for i in range(n_u):
        faces.append([0, 1 + i, 1 + (i + 1) % n_u])

    for j in range(n_v - 1):
        for i in range(n_u):
            v0 = 1 + j * n_u + i
            v1 = 1 + j * n_u + (i + 1) % n_u
            v2 = 1 + (j + 1) * n_u + i
            v3 = 1 + (j + 1) * n_u + (i + 1) % n_u
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    last = 1 + (n_v - 1) * n_u
    for i in range(n_u):
        faces.append([pole_bot, last + (i + 1) % n_u, last + i])

    return verts, np.array(faces)


def generate_tube_mesh(start, end, radius, resolution=12):
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    d = direction / length
    if abs(d[2]) < 0.9:
        perp1 = np.cross(d, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(d, np.array([1, 0, 0]))
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(d, perp1)

    angles = np.linspace(0, 2 * np.pi, resolution + 1)[:-1]
    n_rings = 8

    verts = []
    for t in np.linspace(0, 1, n_rings):
        c = start + t * direction
        for a in angles:
            verts.append(c + radius * (np.cos(a) * perp1 + np.sin(a) * perp2))
    verts = np.array(verts)

    faces = []
    for j in range(n_rings - 1):
        for i in range(resolution):
            i2 = (i + 1) % resolution
            v0, v1 = j * resolution + i, j * resolution + i2
            v2, v3 = (j + 1) * resolution + i, (j + 1) * resolution + i2
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Caps
    ci = len(verts)
    verts = np.vstack([verts, [start]])
    for i in range(resolution):
        faces.append([ci, (i + 1) % resolution, i])

    ci = len(verts)
    verts = np.vstack([verts, [end]])
    lr = (n_rings - 1) * resolution
    for i in range(resolution):
        faces.append([ci, lr + i, lr + (i + 1) % resolution])

    return verts, np.array(faces)


def merge_meshes(meshes):
    all_v, all_f, offset = [], [], 0
    for v, f in meshes:
        if len(v) == 0:
            continue
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.vstack(all_v), np.vstack(all_f)


class CapabilityGraph:
    """Capability graph that thickens into a 3D printable shape."""

    def __init__(self, name, nodes, edges, edge_weights=None, node_sizes=None):
        self.name = name
        self.nodes = nodes  # list of (label, [x,y,z])
        self.edges = edges  # list of (idx, idx)
        self.edge_weights = edge_weights or {}
        self.node_sizes = node_sizes or {}

    @property
    def n(self):
        return len(self.nodes)

    @property
    def cycle_rank(self):
        # For connected graph: E - V + 1
        return max(0, len(self.edges) - len(self.nodes) + 1)

    @property
    def adjacency_matrix(self):
        A = np.zeros((self.n, self.n))
        for i, (a, b) in enumerate(self.edges):
            w = self.edge_weights.get(i, 1.0)
            A[a, b] = w
            A[b, a] = w
        return A

    def to_mesh(self, node_radius=0.3, tube_radius=0.12, res=16):
        meshes = []
        for i, (_, pos) in enumerate(self.nodes):
            r = self.node_sizes.get(i, node_radius)
            meshes.append(generate_sphere_mesh(np.array(pos, dtype=float), r, res))
        for i, (a, b) in enumerate(self.edges):
            w = self.edge_weights.get(i, 1.0)
            s = np.array(self.nodes[a][1], dtype=float)
            e = np.array(self.nodes[b][1], dtype=float)
            meshes.append(generate_tube_mesh(s, e, tube_radius * w, res))
        return merge_meshes(meshes)


def make_t9_candidates():
    """
    Generate candidate capability lattices for T9.
    C = Constraint satisfaction, F = Frequency ranking, X = Bigram context
    """
    h = np.sqrt(3) / 2
    candidates = []

    # 1. Chain: C -> F -> X (acyclic, genus 0)
    candidates.append(CapabilityGraph(
        "chain_linear",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [2, 0, 0])],
        [(0, 1), (1, 2)]))

    # 2. Triangle: C-F-X fully connected (1 cycle, genus 1)
    candidates.append(CapabilityGraph(
        "triangle",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [0.5, h, 0])],
        [(0, 1), (1, 2), (0, 2)]))

    # 3. Star: F hub
    candidates.append(CapabilityGraph(
        "star_F_hub",
        [("C", [-1, 0, 0]), ("F", [0, 0, 0]), ("X", [1, 0, 0])],
        [(0, 1), (1, 2)]))

    # 4. Star: C hub
    candidates.append(CapabilityGraph(
        "star_C_hub",
        [("C", [0, 0, 0]), ("F", [1, 0.5, 0]), ("X", [1, -0.5, 0])],
        [(0, 1), (0, 2)]))

    # 5. Triangle with C-F bond emphasized
    candidates.append(CapabilityGraph(
        "triangle_strong_CF",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [0.5, h, 0])],
        [(0, 1), (1, 2), (0, 2)],
        edge_weights={0: 2.0, 1: 0.7, 2: 0.7}))

    # 6. Chain with thick C-F
    candidates.append(CapabilityGraph(
        "chain_thick_CF",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [2, 0, 0])],
        [(0, 1), (1, 2)],
        edge_weights={0: 2.0, 1: 0.5}))

    # 7. 3D triangle (lifted into z)
    candidates.append(CapabilityGraph(
        "triangle_3d",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [0.5, 0.5, 0.8])],
        [(0, 1), (1, 2), (0, 2)]))

    # 8. Tetrahedron with meta-composition node
    candidates.append(CapabilityGraph(
        "tetra_meta",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]),
         ("X", [0.5, h, 0]), ("M", [0.5, h/3, 0.8])],
        [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]))

    # 9. F self-loop proxy (F-F' micro-cycle)
    candidates.append(CapabilityGraph(
        "chain_F_loop",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("F'", [1.3, 0.3, 0]),
         ("X", [2, 0, 0])],
        [(0, 1), (1, 2), (2, 1), (1, 3)],
        node_sizes={2: 0.15}))

    # 10. Disconnected X
    candidates.append(CapabilityGraph(
        "disconnected_X",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [2, 1, 0])],
        [(0, 1)]))

    # 11. Triangle big C
    candidates.append(CapabilityGraph(
        "triangle_big_C",
        [("C", [0, 0, 0]), ("F", [1, 0, 0]), ("X", [0.5, h, 0])],
        [(0, 1), (1, 2), (0, 2)],
        node_sizes={0: 0.5, 1: 0.2, 2: 0.2}))

    # 12. Singleton
    candidates.append(CapabilityGraph(
        "singleton_C",
        [("C", [0, 0, 0])],
        []))

    return candidates
