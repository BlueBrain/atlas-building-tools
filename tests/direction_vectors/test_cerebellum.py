import numpy as np
import numpy.testing as npt
from voxcell import VoxelData

from atlas_building_tools.direction_vectors import cerebellum as tested


def test_compute_cerebellum_direction_vectors():
    cerebellum_raw = np.zeros((8, 8, 8))
    for x_index, region_id in enumerate([10707, 10692, 10706, 10691, 10705, 10690, 744, 728]):
        cerebellum_raw[x_index, :, :] = region_id
    cerebellum_raw = np.pad(
        cerebellum_raw, 2, "constant", constant_values=0
    )  # Add 2-voxel void margin around the positive annotations

    cerebellum = VoxelData(cerebellum_raw, (25.0, 25.0, 25.0), offset=(1.0, 2.0, 3.0))
    direction_vectors = tested.compute_direction_vectors(cerebellum)
    # Outside the Flocculus and the Lingula, the direction vectors are not defined
    for region_id in [0, 728, 744]:
        region_mask = cerebellum_raw == region_id
        np.all(np.isnan(direction_vectors[region_mask]))

    # Check that the well-defined direction vectors have all norm 1.0
    norm = np.linalg.norm(direction_vectors, axis=3)
    npt.assert_array_almost_equal(
        norm[~np.isnan(norm)], np.full((12, 12, 12), 1.0)[~np.isnan(norm)]
    )
