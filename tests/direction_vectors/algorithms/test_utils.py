import numpy as np
import numpy.testing as npt
from atlas_building_tools.direction_vectors.algorithms.utils import (
    compute_blur_gradient,
)


def test_compute_blur_gradient():
    scalar_field = np.zeros((5, 5, 5))
    scalar_field[0, :, :] = -1
    scalar_field[1, :, :] = 1

    # The standard deviation  of the Gaussian blur is large enough, so that
    # the gradient is non-zero everywhere.
    gradient = compute_blur_gradient(scalar_field)  # the default stddev is 3.0
    assert np.all(~np.isnan(gradient))
    assert np.all(gradient[..., 0] > 0.0)  # vectors flow along the positive x-axis
    npt.assert_array_almost_equal(
        np.linalg.norm(gradient, axis=3), np.full((5, 5, 5), 1.0)
    )

    # The standard deviation of the Gaussian blur is too small:
    # some gradient vectors are zero, but the non-zero ones
    # are normalized as expected.
    gradient = compute_blur_gradient(scalar_field, 0.1)
    nan_mask = np.isnan(gradient)
    assert np.any(nan_mask)
    norm = np.linalg.norm(gradient, axis=3)
    npt.assert_array_almost_equal(
        norm[~np.isnan(norm)], np.full((5, 5, 5), 1.0)[~np.isnan(norm)]
    )
