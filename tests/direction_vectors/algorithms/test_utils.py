import pytest as pyt
import numpy as np
import numpy.testing as npt
import atlas_building_tools.direction_vectors.algorithms.utils as tested


def test_zero_to_nan():
    # 3D unit vectors
    field = np.ones((2, 2, 2, 3), dtype=np.float32)
    field[0, 0, 0] = 0
    field[1, 1, 1] = 0
    expected = np.ones((2, 2, 2, 3), dtype=float)
    expected[0, 0, 0] = np.nan
    expected[1, 1, 1] = np.nan
    tested.zero_to_nan(field)
    npt.assert_array_equal(field, expected)

    # Quaternions
    field = np.ones((2, 2, 2, 4), dtype=np.float64)
    field[0, 1, 0] = 0
    field[1, 0, 1] = 0
    expected = np.ones((2, 2, 2, 4), dtype=float)
    expected[0, 1, 0] = np.nan
    expected[1, 0, 1] = np.nan
    tested.zero_to_nan(field)
    npt.assert_array_equal(field, expected)

    # Wrong input type
    field = np.ones((2, 2, 2, 4), dtype=int)
    with pyt.raises(ValueError):
        tested.zero_to_nan(field)


def test_normalized():
    # A 3D vector field of integer type
    vector_field = np.ones((2, 2, 2, 3), dtype=np.int32)
    normalized = tested.normalized(vector_field)
    npt.assert_array_almost_equal(normalized, np.full((2, 2, 2, 3), 1.0 / np.sqrt(3.0)))
    # A 4D vector field with some zero vectors
    vector_field = np.zeros((2, 2, 2, 4), dtype=np.float32)
    vector_field[0, 0, 0] = np.array([1, 1.0, 0.0, 0.0])
    vector_field[1, 1, 1] = np.array([0, 1, 1, 1])
    normalized = tested.normalized(vector_field)
    expected = np.full(normalized.shape, np.nan)
    expected[0, 0, 0] = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    expected[1, 1, 1] = np.array([0.0, 1.0, 1.0, 1.0]) / np.sqrt(3.0)
    npt.assert_array_almost_equal(normalized, expected)


def test_compute_blur_gradient():
    scalar_field = np.zeros((5, 5, 5))
    scalar_field[0, :, :] = -1
    scalar_field[1, :, :] = 1

    # The standard deviation  of the Gaussian blur is large enough, so that
    # the gradient is non-zero everywhere.
    gradient = tested.compute_blur_gradient(scalar_field)  # the default stddev is 3.0
    assert np.all(~np.isnan(gradient))
    assert np.all(gradient[..., 0] > 0.0)  # vectors flow along the positive x-axis
    npt.assert_array_almost_equal(
        np.linalg.norm(gradient, axis=3), np.full((5, 5, 5), 1.0)
    )

    # The standard deviation of the Gaussian blur is too small:
    # some gradient vectors are zero, but the non-zero ones
    # are normalized as expected.
    gradient = tested.compute_blur_gradient(scalar_field, 0.1)
    nan_mask = np.isnan(gradient)
    assert np.any(nan_mask)
    norm = np.linalg.norm(gradient, axis=3)
    npt.assert_array_almost_equal(
        norm[~np.isnan(norm)], np.full((5, 5, 5), 1.0)[~np.isnan(norm)]
    )

    # Wrong input type
    scalar_field = np.ones((2, 2, 2), dtype=int)
    with pyt.raises(ValueError):
        tested.compute_blur_gradient(scalar_field, 0.1)


class Test_quaternion_to_vector:
    def test_single_vector(self):
        vectors = tested.quaternion_to_vector(np.array([1.0, 0.0, 0.0, 0.0]))
        npt.assert_almost_equal(vectors, [0.0, 1.0, 0.0])
        vectors = tested.quaternion_to_vector(np.array([1.0, 0.0, 0.0, -1.0]))
        npt.assert_almost_equal(vectors, [1.0, 0.0, 0.0])
        vectors = tested.quaternion_to_vector(np.array([1.0, 1.0, 0.0, 0.0]))
        npt.assert_almost_equal(vectors, [0, 0, 1])
        vectors = tested.quaternion_to_vector(np.array([1.0, 1.0, 0.0, 0.0]) * 2.0)
        npt.assert_almost_equal(vectors, [0, 0, 1])

    def test_identity_quaternion_gives_y(self):
        vectors = tested.quaternion_to_vector(np.array([[[1.0, 0.0, 0.0, 0.0]]]))
        npt.assert_almost_equal(vectors, [[[0.0, 1.0, 0.0]]])

    def test_invert_quaternion_gives_negative_y(self):
        vectors = tested.quaternion_to_vector(np.array([[[0.0, 1.0, 0.0, 0.0]]]))
        npt.assert_almost_equal(vectors, [[[0, -1, 0]]])


class Test_vector_to_quaternion:
    @pyt.mark.xfail
    def test_long_vector_gives_same_quat(self):
        npt.assert_almost_equal(
            tested.vector_to_quaternion(np.array([[[1.0, 0.0, 0.0]]])),
            tested.vector_to_quaternion(np.array([[[5, 0, 0]]])),
        )


def test_quaternion_converters_consistency():
    vectors = tested.normalized(
        np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [0.0, 2.0, 1.0]])
    )
    quaternions = tested.vector_to_quaternion(vectors)
    npt.assert_almost_equal(
        tested.quaternion_to_vector(quaternions), vectors, decimal=3
    )
    vectors = tested.normalized(
        np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    )
    quaternions = tested.vector_to_quaternion(vectors)
    npt.assert_almost_equal(
        tested.quaternion_to_vector(quaternions), vectors, decimal=3
    )
