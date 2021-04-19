import numpy as np
import numpy.testing as npt
import pytest

from atlas_building_tools.direction_vectors.algorithms.blur_gradient import (
    RegionShading,
    compute_direction_vectors,
    compute_initial_field,
    create_thick_boundary_mask,
    shading_from_boundary,
)


def test_create_thick_boundary_mask():
    input_mask = np.zeros((6, 6, 6))
    input_mask[1:3, 1:5, 1:5] = 1
    input_mask[3:5, 1:5, 1:5] = 2

    with pytest.raises(AssertionError):
        create_thick_boundary_mask(input_mask, 1, 2, -1)

    assert np.all(~create_thick_boundary_mask(input_mask, 2, 3, 1))
    npt.assert_array_equal(create_thick_boundary_mask(input_mask, 2, 2, 1), (input_mask == 2))

    expected_res = np.full(input_mask.shape, False)
    expected_res[2:4, 1:5, 1:5] = True
    npt.assert_array_equal(create_thick_boundary_mask(input_mask, 1, 2, 1), expected_res)
    npt.assert_array_equal(create_thick_boundary_mask(input_mask, 1, 2, 2), (input_mask > 0))
    assert np.all(~create_thick_boundary_mask(input_mask, 1, 2, 70))


def test_shading_from_boundary():
    input_mask = np.zeros((6, 6, 6))
    input_mask[1:3, 1:5, 1:5] = 1
    input_mask[3:5, 1:5, 1:5] = 2
    shading = RegionShading([0, 1], 1, 0, -1, invert=True)
    assert np.all(shading_from_boundary(input_mask, shading) == 0)  # negative max distance
    shading = RegionShading([0, 1], 1, 0, 0, invert=True)
    assert np.all(shading_from_boundary(input_mask, shading) == 0)
    shading = RegionShading([0, 3], 3, 0, 1, invert=True)
    assert np.all(shading_from_boundary(input_mask, shading) == 0)  # roi not present
    shading = RegionShading([0, 1, 2], 1, 0, 1, invert=True)
    assert np.all(shading_from_boundary(input_mask, shading) == 0)  # all regions ignored

    expected_res = np.zeros(input_mask.shape, dtype=int)
    expected_res[3, 1:5, 1:5] = 1
    shading = RegionShading([0, 1], 1, 0, 1, invert=True)
    npt.assert_array_equal(shading_from_boundary(input_mask, shading), expected_res)
    expected_res[4, 1:5, 1:5] = 2
    shading = RegionShading([0, 1], 1, 0, 2, invert=True)
    npt.assert_array_equal(shading_from_boundary(input_mask, shading), expected_res)
    shading = RegionShading([0, 1], 1, 0, 10, invert=True)
    npt.assert_array_equal(shading_from_boundary(input_mask, shading), expected_res)
    expected_res[1:3, 0, 1:5] = 1
    expected_res[1:3, 5, 1:5] = 1
    expected_res[1:3, 1:5, 0] = 1
    expected_res[1:3, 1:5, 5] = 1
    expected_res[0, 1:5, 1:5] = 1
    expected_res[5, 1:5, 1:5] = 3
    shading = RegionShading([1], 1, 0, 10, invert=True)
    npt.assert_array_equal(shading_from_boundary(input_mask, shading), expected_res)


def test_compute_initial_field_single_weights():
    # Two regions
    raw = np.zeros((2, 2, 2))
    raw[0, :, :] = 111
    raw[1, :, :] = 222
    region_weights = {111: 1, 222: 1}
    initial_field = compute_initial_field(raw, region_weights)
    npt.assert_array_equal(initial_field, np.ones((2, 2, 2)))

    # Four regions
    raw = np.zeros((4, 2, 2))
    raw[0, :, :] = 111
    raw[1, :, :] = 222
    raw[2, :, :] = 333
    raw[3, :, :] = 444
    region_weights = {111: 1, 222: -1, 333: 2, 444: 2}
    initial_field = compute_initial_field(raw, region_weights)
    expected_field = np.zeros_like(raw)
    expected_field[0, :, :] = 1
    expected_field[1, :, :] = -1
    expected_field[2, :, :] = 2
    expected_field[3, :, :] = 2
    npt.assert_array_equal(initial_field, expected_field)


def test_compute_initial_field_with_shadings():
    raw = np.zeros((6, 2, 2))
    raw[0, :, :] = 111
    raw[1, :, :] = 222  # to shade
    raw[2, :, :] = 333  # single weight only
    raw[3, :, :] = 444  # single weight only
    raw[4, :, :] = 555  # to shade
    raw[5, :, :] = 666  # to shade
    region_weights = {333: -1, 444: 1}
    shadings = [
        # offset = 1, limit_distance = 3, which is greater than the actual space on the left
        RegionShading([111, 222], 333, 1, 3),
        # offset = 1, limit_distance = 1, which is less than the actual space on the right
        RegionShading([555, 666], 444, 2, 1),
    ]
    expected_field = np.zeros_like(raw)
    expected_field[0, :, :] = 3
    expected_field[1, :, :] = 2
    expected_field[2, :, :] = -1
    expected_field[3, :, :] = 1
    expected_field[4, :, :] = 3
    expected_field[5, :, :] = 0
    initial_field = compute_initial_field(raw, region_weights, shadings)
    npt.assert_array_equal(initial_field, expected_field)
    # Same field as before, but the regions to shade
    # are specified by means of their complement
    shadings = [
        # Inverted ids selection
        RegionShading([333, 444, 555, 666], 333, 1, 3, invert=True),
        # Inverted ids selection
        RegionShading([111, 222, 333, 444], 444, 2, 1, invert=True),
    ]
    initial_field = compute_initial_field(raw, region_weights, shadings)
    npt.assert_array_equal(initial_field, expected_field)
    # Same annotation as before, but we specify additional single weights
    region_weights = {
        # The following two weights won't have any effect, since the shadings
        # are positive in these regions
        111: 6,
        222: 6,
        333: -1,
        444: 1,
        555: 6,
        666: 6,
    }
    shadings = [
        # Inverted ids selection
        RegionShading([333, 444, 555, 666], 333, 1, 3, invert=True),
        # Inverted ids selection
        RegionShading([111, 222, 333, 444], 444, 2, 1, invert=True),
    ]
    expected_field = np.zeros_like(raw)
    expected_field[0, :, :] = 3
    expected_field[1, :, :] = 2
    expected_field[2, :, :] = -1
    expected_field[3, :, :] = 1
    expected_field[4, :, :] = 3
    expected_field[5, :, :] = 6
    initial_field = compute_initial_field(raw, region_weights, shadings)
    npt.assert_array_equal(initial_field, expected_field)


def test_compute_direction_vectors():
    raw = np.zeros((7 * 3, 2, 2))
    raw[0:3, :, :] = 111  # non-roi, fixed weight
    raw[3:6, :, :] = 222  # non-roi, fixed weight
    raw[6:9, :, :] = 333  # roi, single weight only
    raw[9:12, :, :] = 444  # roi, single weight only
    raw[12:15, :, :] = 555  # roi, single weight only
    raw[15:18, :, :] = 666  # non-roi, to shade
    raw[18:21, :, :] = 777  # non-roi, to shade, plus overlayed fixed weight
    region_weights = {111: -5, 222: -5, 333: -1, 444: 0, 555: 1, 777: 5}
    shadings = [
        RegionShading([666, 777], 555, 1, 3),
    ]
    initial_field = compute_initial_field(raw, region_weights, shadings)
    direction_vectors = compute_direction_vectors(raw, initial_field, [333, 444, 555])
    # Direction vectors are [np.nan] * 3 for every voxel outside the
    # the regions of interest.
    region_of_interest_mask = np.isin(raw, [333, 444, 555])
    assert np.all(np.isnan(direction_vectors[~region_of_interest_mask]))
    # Inside the regions of interest, the non-nan direction vectors
    # should be unit vectors.
    norm = np.linalg.norm(direction_vectors, axis=3)
    npt.assert_array_equal(norm[~np.isnan(norm)], np.full(raw.shape, 1.0)[~np.isnan(norm)])
    # Direction vectors should flow along the positive x-axis
    assert np.all(direction_vectors[region_of_interest_mask, 0] > 0.0)
