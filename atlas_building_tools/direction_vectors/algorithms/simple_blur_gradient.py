'''Generate direction vectors using the normalized gradient of a Gaussian blur.

The source and target regions of fibers are assigned respectively
a low and a high weight value. A Gaussian blur is applied to generate a smooth scalar field
on the volume of interest. The algorithm returns the normalized gradient of this field.

This algorithm can be used for laminar regions such as the isocortex.
'''

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore
from nptyping import NDArray  # type: ignore

from atlas_building_tools.direction_vectors.algorithms.utils import (
    compute_blur_gradient,
)


def compute_direction_vectors(
    source: NDArray[bool],
    inside: NDArray[bool],
    target: NDArray[bool],
    sigma: float = 10.0,
) -> NDArray[np.float32]:
    '''
    Compute direction vectors in the `inside` volume.

    The `source` and `target` regions are assigned the weight -1.0 and +1.0 respectively.
    Voxels outside these two regions zeroed.
    This initial landscape is blurred by a Gaussian filter with standard deviation
    `sigma` and the normalized gradient of the latter field is returned.

    All input masks are assumed to have the same shape.

    Args:
        source: 3D binary mask of the source region, i.e.,
            the region where the fibers originate from.
        inside: 3D binary mask of the region where direction vectors
            are computed.
        target: 3D binary mask of the fibers target region.
        sigma: standard deviation of the Gaussian kernel used
            to blur the initial scalar field.

    Returns:
        Array holding a vector field of unit vectors
        defined on the `inside` 3D volume. The shape of this array
        is (W, L, D, 3) if the shape of `inside` is (W, L, D).
        Outside the `inside` volume, the returned direction vectors
        have np.nan coordinates.

    '''
    npt.assert_array_equal(source.shape, inside.shape)
    npt.assert_array_equal(inside.shape, target.shape)

    scalar_field = np.zeros_like(inside, np.float32)
    scalar_field[source] = -1.0
    scalar_field[target] = +1.0

    direction_vectors = compute_blur_gradient(scalar_field, gaussian_stddev=sigma)
    direction_vectors[~inside, :] = np.nan
    return direction_vectors
