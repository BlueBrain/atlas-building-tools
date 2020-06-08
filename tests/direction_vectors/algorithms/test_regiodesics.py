import pytest
import numpy as np
import numpy.testing as npt
from mock import patch

from atlas_building_tools.direction_vectors.algorithms import regiodesics as tested
from atlas_building_tools.direction_vectors.algorithms.regiodesics import (
    RegiodesicsLabels,
)
from atlas_building_tools.exceptions import AtlasBuildingToolsError


@patch(
    'atlas_building_tools.direction_vectors.algorithms.regiodesics.find_executable',
    return_value='/home/software/Regiodesics/layer_segmenter',
)
def test_find_regiodesics_exec_or_raise_found(_):
    assert (
        tested.find_regiodesics_exec_or_raise('layer_segmenter')
        == '/home/software/Regiodesics/layer_segmenter'
    )


@patch(
    'atlas_building_tools.direction_vectors.algorithms.regiodesics.find_executable',
    return_value='',
)
def test_find_regiodesics_exec_or_raise_raises(_):
    with pytest.raises(FileExistsError):
        tested.find_regiodesics_exec_or_raise('geodesics')


def test_compute_boundary():
    v_1 = np.zeros((5, 5, 5), dtype=np.bool)
    v_1[1:4, 1:4, 1:4] = True
    v_2 = ~v_1
    boundary = tested.compute_boundary(v_1, v_2)
    expected = np.copy(v_1)
    expected[2, 2, 2] = False
    npt.assert_array_equal(boundary, expected)

    v_1 = np.zeros((5, 5, 5), dtype=np.bool)
    v_1[0:2, :, 1:4] = True
    v_2 = np.zeros_like(v_1)
    v_2[2:, :, 1:4] = True
    boundary = tested.compute_boundary(v_1, v_2)
    expected = np.zeros_like(v_1)
    expected[1, :, 1:4] = True
    npt.assert_array_equal(boundary, expected)


def test_mark_with_regiodesics_labels():
    full_volume = np.zeros((9, 9, 9), dtype=np.int)
    full_volume[:, :, :3] = 1  # bottom
    full_volume[:, :, 3:6] = 2  # in between
    full_volume[:, :, 6:] = 3  # top
    marked = tested.mark_with_regiodesics_labels(
        full_volume == 1, full_volume == 2, full_volume == 3
    )
    expected = np.zeros((9, 9, 9), dtype=np.int)
    expected[:, :, 4] = RegiodesicsLabels.INTERIOR
    expected[:, :, 3] = RegiodesicsLabels.BOTTOM
    expected[0, :, 4] = RegiodesicsLabels.SHELL
    expected[8, :, 4] = RegiodesicsLabels.SHELL
    expected[:, 0, 4] = RegiodesicsLabels.SHELL
    expected[:, 8, 4] = RegiodesicsLabels.SHELL
    expected[:, :, 5] = RegiodesicsLabels.TOP
    npt.assert_array_equal(marked, expected)


def test_compute_direction_vectors():
    raw = np.zeros((8, 8, 8), dtype=np.int)
    raw[:, :, :2] = 1  # bottom
    raw[:, :, 2:6] = 2  # interior
    raw[:, :, 6:8] = 3  # top
    direction_vectors = tested.compute_direction_vectors(raw == 1, raw == 2, raw == 3)
    expected = np.zeros(raw.shape + (3,))
    expected[:, :, :] = np.array([0.0, 0.0, 1.0])
    nan_mask = np.isnan(direction_vectors)
    npt.assert_array_almost_equal(direction_vectors[~nan_mask], expected[~nan_mask])


def test_compute_direction_vectors_exception():
    raw = np.zeros((8, 8, 8), dtype=np.int)
    raw[:, :, 2:6] = 2  # interior
    raw[:, :, 6:8] = 3  # top
    with pytest.raises(AtlasBuildingToolsError):
        tested.compute_direction_vectors(raw == 1, raw == 2, raw == 3)
