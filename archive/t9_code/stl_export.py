"""
stl_export.py - Export capability graph meshes as 3D-printable STL files.
"""
import numpy as np
from stl import mesh as stl_mesh


def export_stl(vertices, faces, filename, scale=20.0):
    """
    Export mesh as binary STL file.
    scale: multiply coordinates to get mm-scale for 3D printing.
    """
    n_faces = len(faces)
    stl_data = stl_mesh.Mesh(np.zeros(n_faces, dtype=stl_mesh.Mesh.dtype))

    scaled_verts = vertices * scale

    for i, face in enumerate(faces):
        for j in range(3):
            stl_data.vectors[i][j] = scaled_verts[int(face[j])]

    stl_data.save(filename)
    return filename


def export_graph_stl(graph, filename, scale=20.0, resolution=16):
    """Export a CapabilityGraph as a 3D-printable STL."""
    verts, faces = graph.to_mesh(
        node_radius=0.3,
        tube_radius=0.12,
        res=resolution
    )
    return export_stl(verts, faces, filename, scale)
