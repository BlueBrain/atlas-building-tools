"""Low-level tools for the computation of direction vectors"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def compute_blur_gradient(scalar_field, gaussian_stddev=3.0):
    """
    Blurs a scalar field and returns its normalized gradient.

    A Gaussian filter (blur) with standard deviation `gaussian_stdev`
    is applied to the input field. The function returns the normalized
    gradient of the filtered field.

    Arguments:
        scalar_field(numpy.ndarray): scalar field defined over
            a 3D volume.
        gaussian_stddev: standard deviation of the Gaussian kernel used by the
            Gaussian filter.
    Returns:
        3D numpy.ndarray of floats. A 3D unit vector field over the underlying 3D volume
        of the input scalar field. This vector field will contain np.nan vectors if the
        normalization process encountered some zero vectors.
    """

    blurred = gaussian_filter(np.float32(scalar_field), sigma=gaussian_stddev)
    gradient = np.array(np.gradient(blurred))
    norm = np.linalg.norm(gradient, axis=0)
    with np.errstate(
        divide='ignore', invalid='ignore'
    ):  # Handles division by zero silently
        gradient /= norm[np.newaxis, ...]

    return np.moveaxis(gradient, 0, -1)
