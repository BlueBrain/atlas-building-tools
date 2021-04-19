"""
Unit tests for densities utils
"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from mock import patch
from voxcell import RegionMap

import atlas_building_tools.densities.utils as tested
from atlas_building_tools.exceptions import AtlasBuildingToolsError

TESTS_PATH = Path(__file__).parent.parent


def test_normalize():
    annotation_raw = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 255, 1211, 0],
                [0, 347, 100, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.uint32,
    )
    marker = np.array(
        [
            [
                [0.0, 0.5, 0.5, 0.2],
                [0.1, 0.0, 1.0, 0.1],
                [0.1, 3.0, 5.0, 0.1],
                [0.2, 0.1, 0.1, 0.0],
            ]
        ],
        dtype=float,
    )
    original_marker = marker.copy()
    expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4 / 4.4, 0.0],
                [0.0, 2.4 / 4.4, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=float,
    )
    # Return a modified copy of the input
    actual = tested.normalize_intensity(marker, annotation_raw)
    npt.assert_array_almost_equal(actual, expected)
    npt.assert_array_equal(marker, original_marker)
    # In place modification
    actual = tested.normalize_intensity(marker, annotation_raw, copy=False)
    npt.assert_array_almost_equal(marker, expected)


def test_compensate_cell_overlap():
    annotation_raw = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 15, 1, 0],
                [0, 999, 1001, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.uint32,
    )
    marker = np.array(
        [
            [
                [0.0, -0.5, 0.5, 0.2],
                [0.1, 10.0, 1.0, 0.1],
                [0.1, 3.0, 5.0, 0.1],
                [0.2, 0.1, -0.1, 0.0],
            ]
        ],
        dtype=float,
    )
    original_marker = marker.copy()

    # Return a modified copy of the input
    actual = tested.compensate_cell_overlap(marker, annotation_raw)
    npt.assert_array_equal(marker, original_marker)
    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)
    assert np.all(actual[annotation_raw == 0] == 0.0)
    # In place modification
    tested.normalize_intensity(marker, annotation_raw, copy=False)
    assert np.all(marker >= 0.0)
    assert np.all(marker <= 1.0)
    assert np.all(marker[annotation_raw == 0] == 0.0)


def test_get_group_ids():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    group_ids = tested.get_group_ids(region_map)
    for ids in group_ids.values():
        assert len(ids) > 0
    assert len(group_ids["Molecular layer"] & group_ids["Purkinje layer"]) == 0
    assert len(group_ids["Cerebellum group"] & group_ids["Isocortex group"]) == 0
    assert len(group_ids["Cerebellum group"] & group_ids["Molecular layer"]) > 0
    assert len(group_ids["Cerebellum group"] & group_ids["Purkinje layer"]) > 0
    assert len(group_ids["Isocortex group"] & group_ids["Molecular layer"]) > 0
    assert len(group_ids["Isocortex group"] & group_ids["Purkinje layer"]) == 0
    assert group_ids["Cerebellar cortex"].issubset(group_ids["Cerebellum group"])


def test_get_region_masks():
    region_map = RegionMap.load_json(Path(TESTS_PATH, "1.json"))
    group_ids = tested.get_group_ids(region_map)
    annotation_raw = np.arange(27000).reshape(30, 30, 30)
    region_masks = tested.get_region_masks(group_ids, annotation_raw)
    brain_mask = np.logical_or(
        np.logical_or(region_masks["Cerebellum group"], region_masks["Isocortex group"]),
        region_masks["Rest"],
    )
    npt.assert_array_equal(annotation_raw != 0, brain_mask)
    np.all(~np.logical_and(region_masks["Cerebellum group"], region_masks["Isocortex group"]))


def test_optimize_distance_to_line_2D():
    line_direction_vector = np.array([1, 2], dtype=float)
    upper_bounds = np.array([3, 1], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 3.0, threshold=1e-7, copy=True
    )
    npt.assert_array_almost_equal(optimum, np.array([2.0, 1.0]))

    line_direction_vector = np.array([2, 3], dtype=float)
    upper_bounds = np.array([1, 3], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 2.0, threshold=1e-7, copy=False
    )
    npt.assert_array_almost_equal(optimum, np.array([0.8, 1.2]))

    line_direction_vector = np.array([1, 2], dtype=float)
    upper_bounds = np.array([3, 1], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 5.0, threshold=1e-7
    )
    npt.assert_array_almost_equal(optimum, np.array([3.0, 1.0]))


def test_optimize_distance_to_line_3D():
    line_direction_vector = np.array([0.5, 2.0, 1.0], dtype=float)
    upper_bounds = np.array([1.0, 1.0, 1.0], dtype=float)
    optimum = tested.optimize_distance_to_line(
        line_direction_vector, upper_bounds, 2.0, threshold=1e-7, copy=True
    )
    npt.assert_array_almost_equal(optimum, np.array([1.0 / 3.0, 1.0, 2.0 / 3.0]))


def test_constrain_cell_counts_per_voxel():
    upper_bound = np.array([[[1.0, 1.0, 2.0, 0.5, 0.5]]])
    cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
    zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
    max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
    cell_counts = tested.constrain_cell_counts_per_voxel(
        3.0,
        cell_counts,
        upper_bound,
        max_cell_counts_mask,
        zero_cell_counts_mask,
        epsilon=1e-7,
        copy=True,
    )
    expected = np.array([[[0.0, 0.0, 2.0, 0.5, 0.5]]])
    npt.assert_almost_equal(cell_counts, expected, decimal=6)

    upper_bound = np.array([[[2.0, 0.9, 0.75, 0.5, 1.5, 0.1]]])
    cell_counts = np.array([[[0.1, 0.8, 0.25, 0.25, 0.85, 0.1]]])
    zero_cell_counts_mask = np.array([[[True, False, False, False, False, True]]])
    max_cell_counts_mask = np.array([[[False, True, False, False, True, False]]])
    cell_counts = tested.constrain_cell_counts_per_voxel(
        3.4,
        cell_counts,
        upper_bound,
        max_cell_counts_mask,
        zero_cell_counts_mask,
        epsilon=1e-7,
        copy=False,
    )
    expected = np.array([[[0.0, 0.9, 0.5, 0.5, 1.5, 0.0]]])
    npt.assert_almost_equal(cell_counts, expected, decimal=6)

    # Same constraints, but with a different line
    cell_counts = np.array([[[0.1, 0.8, 0.6, 0.2, 0.85, 0.1]]])
    expected = np.array([[[0.0, 0.9, 0.75, 0.25, 1.5, 0.0]]])
    cell_counts = tested.constrain_cell_counts_per_voxel(
        3.4,
        cell_counts,
        upper_bound,
        max_cell_counts_mask,
        zero_cell_counts_mask,
        epsilon=1e-7,
        copy=True,
    )
    npt.assert_almost_equal(cell_counts, expected, decimal=6)


def test_constrain_cell_counts_per_voxel_exceptions():
    # Should raise because the contribution of voxels
    # with maximum cell counts exceeds the target sum
    with pytest.raises(AtlasBuildingToolsError):
        upper_bound = np.array([[[1.0, 1.0, 4.0, 0.5, 0.5]]])
        cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
        zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
        max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
        tested.constrain_cell_counts_per_voxel(
            3.0,
            cell_counts,
            upper_bound,
            max_cell_counts_mask,
            zero_cell_counts_mask,
            epsilon=1e-7,
            copy=True,
        )

    # Should raise because the maximum contribution of voxels
    # with non-zero cell counts is less than the target sum.
    with pytest.raises(AtlasBuildingToolsError):
        upper_bound = np.array([[[1.0, 1.0, 1.0, 0.5, 0.5]]])
        cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
        zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
        max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
        tested.constrain_cell_counts_per_voxel(
            3.0,
            cell_counts,
            upper_bound,
            max_cell_counts_mask,
            zero_cell_counts_mask,
            epsilon=1e-7,
            copy=True,
        )

    # Should raise because the target sum is not reached
    with pytest.raises(AtlasBuildingToolsError):
        upper_bound = np.array([[[1.0, 1.0, 1.0, 0.5, 0.5]]])
        cell_counts = np.array([[[0.1, 0.1, 0.6, 0.25, 0.15]]])
        zero_cell_counts_mask = np.array([[[True, True, False, False, False]]])
        max_cell_counts_mask = np.array([[[False, False, True, False, False]]])
        with patch(
            "atlas_building_tools.densities.utils.optimize_distance_to_line",
            return_value=cell_counts,
        ):
            tested.constrain_cell_counts_per_voxel(
                3.0,
                cell_counts,
                upper_bound,
                max_cell_counts_mask,
                zero_cell_counts_mask,
                copy=True,
            )
