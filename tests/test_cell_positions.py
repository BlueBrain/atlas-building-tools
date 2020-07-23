'''test cell_positions'''
import numpy as np
import numpy.testing as npt

from voxcell import VoxelData

import atlas_building_tools.cell_positions as tested


def test_generate_cell_positions():
    # Sample following a normal distribution
    sigma = 15.0
    range_ = 100.0
    sampling_size = 200
    space = np.linspace(-range_, range_, sampling_size)
    cell_count = 200000
    density = (
        np.array(
            [
                [
                    np.exp(-(space ** 2) / (2.0 * sigma ** 2))
                    / (sigma * np.sqrt(2.0 * np.pi))
                ]
            ]
        )
        * cell_count
    )
    annotation_indices = np.array(np.nonzero(density)).T
    density_data = VoxelData(
        density, offset=(1.0, 2.0, -3.0), voxel_dimensions=(1.0, -2.0, 3.0)
    )
    np.random.seed(0)
    cell_positions = tested.generate_cell_positions(annotation_indices, density_data)
    assert len(cell_positions) == int(np.round(np.sum(density)))
    indices = density_data.positions_to_indices(cell_positions)[:, 2]
    npt.assert_allclose(np.mean(indices), (sampling_size - 1) / 2, rtol=0.01)
    npt.assert_allclose(np.std(indices), sigma, rtol=0.01)

    # Sample following a uniform distribution
    sigma = np.sqrt(1.0 / 12.0)
    cell_count = 200000
    delta = 1.0 / 300.0
    density = np.full((1, 1, 300), cell_count * delta)
    annotation_indices = np.array(np.nonzero(density)).T
    density_data = VoxelData(
        density, offset=(-3.0, 2.0, 3.5), voxel_dimensions=(10.0, 0.5, -3.0)
    )
    cell_positions = tested.generate_cell_positions(annotation_indices, density_data)
    assert len(cell_positions) == int(np.round(np.sum(density)))
    indices = density_data.positions_to_indices(cell_positions)[:, 2]
    npt.assert_allclose(np.mean(indices), 150.0, rtol=0.01)
    npt.assert_allclose(np.std(indices) / 300.0, sigma, rtol=0.01)

    # A density distribution supported by the boundary of a 3D box
    cell_count = 1000
    density = (
        np.ones((20, 20, 20), dtype=np.float32) * cell_count / (6 * 20 ** 2 - 12 * 20)
    )
    density[1:19, 1:19, 1:19] = 0.0
    annotation_indices = np.array(np.nonzero(np.ones(density.shape))).T
    density_data = VoxelData(
        density, offset=(100.0, 20.0, 35.0), voxel_dimensions=(10.0, 10.5, -30.0)
    )
    cell_positions = tested.generate_cell_positions(annotation_indices, density_data)
    assert len(cell_positions) == int(np.round(np.sum(density)))
    indices = density_data.positions_to_indices(cell_positions)
    boundary_mask = (
        np.isin(indices[:, 0], [0, 19])
        | np.isin(indices[:, 1], [0, 19])
        | np.isin(indices[:, 2], [0, 19])
    )
    # All generated positions are located on the box boundary
    assert len(cell_positions) == np.count_nonzero(boundary_mask)
