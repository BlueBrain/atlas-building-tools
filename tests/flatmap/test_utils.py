"""test flatmap.utils"""
import numpy as np
from voxcell import VoxelData

import atlas_building_tools.flatmap.utils as tested


def test_reconstruct_surface_mesh():
    volume = np.zeros((10, 10, 3), dtype=bool)
    volume[:, :, 1] = True
    direction_vectors = np.full((10, 10, 3, 3), np.nan, dtype=float)
    direction_vectors[:, :, 1] = [0.0, 0.0, 1.0]
    normals = VoxelData(
        direction_vectors, offset=[5.0, 10.0, 15.0], voxel_dimensions=[1.0, 2.0, 3.0]
    )
    reconstructed_mesh = tested.reconstruct_surface_mesh(volume, normals)
    points = normals.indices_to_positions(np.array(np.nonzero(volume)).T)
    for point in points:
        deltas = np.linalg.norm(point - reconstructed_mesh.vertices, axis=1)
        assert np.min(deltas) <= 0.4
