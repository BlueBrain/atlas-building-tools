'''
Unit tests for overall cell density computation
'''
from pathlib import Path
import numpy as np
import numpy.testing as npt
from mock import patch

from voxcell import RegionMap, VoxelData  # type: ignore
from atlas_building_tools.densities.cell_counts import cell_counts
from atlas_building_tools.densities.utils import (
    get_group_ids,
    get_region_masks,
)
import atlas_building_tools.densities.cell_density as tested

TESTS_PATH = Path(__file__).parent.parent


def test_compute_cell_density():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    annotation_raw = np.arange(8000).reshape(20, 20, 20)
    nissl = np.random.random_sample(annotation_raw.shape)
    cell_density = tested.compute_cell_density(region_map, annotation_raw, nissl)
    # Each group has a prescribed cell count
    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation_raw)
    for group, mask in region_masks.items():
        npt.assert_array_almost_equal(np.sum(cell_density[mask]), cell_counts()[group])

    # The voxels in the Cerebellum group which belong to the Purkinje layer
    # should all have the same cell density.
    purkinje_layer_mask = np.isin(annotation_raw, list(group_ids['Purkinje layer']))
    purkinje_layer_mask = np.logical_and(
        region_masks['Cerebellum group'], purkinje_layer_mask
    )
    densities = np.unique(cell_density[purkinje_layer_mask])
    assert len(densities) == 1


def test_cell_density_with_soma_correction():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    annotation_raw = np.arange(8000).reshape(20, 20, 20)
    nissl = np.random.random_sample(annotation_raw.shape)
    cell_density = tested.compute_cell_density(
        region_map,
        annotation_raw,
        nissl,
        {1: '0.1', 2: '0.2', 1000: '0.12', 6000: '0.7', 222: '0.9'},
    )
    # Each group has a prescribed cell count
    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation_raw)
    for group, mask in region_masks.items():
        npt.assert_array_almost_equal(np.sum(cell_density[mask]), cell_counts()[group])

    # The voxels in the Cerebellum group which belong to the Purkinje layer
    # should all have the same cell density.
    purkinje_layer_mask = np.isin(annotation_raw, list(group_ids['Purkinje layer']))
    purkinje_layer_mask = np.logical_and(
        region_masks['Cerebellum group'], purkinje_layer_mask
    )
    densities = np.unique(cell_density[purkinje_layer_mask])
    assert len(densities) == 1


def test_cell_density_options():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    annotation_raw = np.arange(8000).reshape(20, 20, 20)
    nissl = np.random.random_sample(annotation_raw.shape)
    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation_raw)
    with patch(
        'atlas_building_tools.densities.cell_density.compensate_cell_overlap',
        return_value=nissl,
    ):
        actual = tested.compute_cell_density(region_map, annotation_raw, nissl.copy())
        expected = tested.fix_purkinje_layer_density(
            region_map, annotation_raw, nissl, cell_counts()
        )
        for group, mask in region_masks.items():
            expected[mask] = nissl[mask] * (cell_counts()[group] / np.sum(nissl[mask]))
        npt.assert_array_equal(expected, actual)

    with patch(
        'atlas_building_tools.densities.cell_density.compensate_cell_overlap',
        return_value=nissl,
    ):
        with patch(
            'atlas_building_tools.densities.cell_density.apply_soma_area_correction'
        ) as apply_correction_mock:
            tested.compute_cell_density(region_map, annotation_raw, nissl, {666: '0.1'})
            assert apply_correction_mock.called
