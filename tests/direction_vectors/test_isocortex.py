from pathlib import Path
import warnings
import numpy as np
import numpy.testing as npt


from voxcell import RegionMap, VoxelData
from tests.direction_vectors.algorithms.test_layer_based_direction_vectors import (
    check_direction_vectors,
)
import atlas_building_tools.direction_vectors.isocortex as tested

TEST_PATH = Path(Path(__file__).parent.parent)
HIERARCHY_PATH = str(Path(TEST_PATH, '1.json'))


def test_get_isocortical_regions():
    hierarchy_json = HIERARCHY_PATH
    raw = np.arange(1, 35).reshape((1, 2, 17))
    expected = ['SSp-m', 'SSp-tr', 'VISp']
    # Path to hierarchy.json
    regions = tested.get_isocortical_regions(raw, hierarchy_json)
    npt.assert_array_equal(regions, expected)

    # True RegionMap object
    region_map = RegionMap.load_json(HIERARCHY_PATH)
    regions = tested.get_isocortical_regions(raw, region_map)
    npt.assert_array_equal(regions, expected)


def test_compute_direction_vectors():
    # Two high-level regions, namely ACAd and ACAv
    # with layers 1, 2/3, 5 and 6
    raw = np.zeros((16, 16, 16), dtype=int)
    # ACAd6
    raw[3:8, 3:12, 3] = 927  # ACAd6b, since ACAd6a is ignored
    raw[3:8, 3:12, 12] = 927
    # ACAd5
    raw[3:8, 3:12, 4] = 1015
    raw[3:8, 3:12, 11] = 1015
    # ACAd2/3
    raw[3:8, 3:12, 5:7] = 211
    raw[3:8, 3:12, 9:11] = 211
    # ACAd1
    raw[3:8, 3:12, 7:9] = 935

    # ACAv6
    raw[8:12, 3:12, 3] = 819  # ACAv6b since ACAv6a is ignored
    raw[8:12, 3:12, 12] = 819
    # ACAv5
    raw[8:12, 3:12, 4] = 772
    raw[8:12, 3:12, 11] = 772
    # ACAv2/3
    raw[8:12, 3:12, 5:7] = 296
    raw[8:12, 3:12, 9:11] = 296
    # ACAv1
    raw[8:12, 3:12, 7:9] = 588

    voxel_data = VoxelData(raw, (1.0, 1.0, 1.0))
    direction_vectors = tested.compute_direction_vectors(HIERARCHY_PATH, voxel_data)
    check_direction_vectors(
        direction_vectors, raw > 0, {'opposite': 'target', 'strict': False}
    )


def test_compute_direction_vectors_with_missing_bottom():
    # Two high-level regions, namely ACAd and ACAv
    # with layers 1, 2/3, 5
    # Layer 6 is missing and troubles are expected!
    raw = np.zeros((16, 16, 16), dtype=int)

    # ACAd5
    raw[3:8, 3:12, 4] = 1015
    raw[3:8, 3:12, 11] = 1015
    # ACAd2/3
    raw[3:8, 3:12, 5:7] = 211
    raw[3:8, 3:12, 9:11] = 211
    # ACAd1
    raw[3:8, 3:12, 7:9] = 935

    # ACAv5
    raw[8:12, 3:12, 4] = 772
    raw[8:12, 3:12, 11] = 772
    # ACAv2/3
    raw[8:12, 3:12, 5:7] = 296
    raw[8:12, 3:12, 9:11] = 296
    # ACAv1
    raw[8:12, 3:12, 7:9] = 588

    voxel_data = VoxelData(raw, (1.0, 1.0, 1.0))
    with warnings.catch_warnings(record=True) as w:
        tested.compute_direction_vectors(HIERARCHY_PATH, voxel_data)
        assert "NaN" in str(w[-1].message)
