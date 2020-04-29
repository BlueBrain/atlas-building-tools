'''Low-level tools for the computation of direction vectors'''
from typing import Tuple, Union
from nptyping import NDArray  # type: ignore
import numpy as np  # type: ignore
from scipy.ndimage.filters import gaussian_filter  # type: ignore
import quaternion  # type: ignore

import voxcell  # type: ignore

# pylint: disable=invalid-name
FloatArray = Union[
    NDArray[float], NDArray[np.float16], NDArray[np.float32], NDArray[np.float64]
]
NumericArray = Union[NDArray[bool], NDArray[int], NDArray[float]]


def zero_to_nan(field: FloatArray) -> None:
    '''
    Turns, in place, the zero vectors of a vector field into NaN vectors.

    Zero vectors are replaced, in place, by vectors with np.nan coordinates.

    Note: This function is used to invalidate zero vectors or zero quaternions as a zero vector
    cannot be used to define a direction or an orientation.
    In addition, it allows the multiplication of an invalid quaternion, i.e., a quaternion with
     NaN coorinates with a vector (the output is a NaN vector) without raising exception.

    Args:
        field: N-dimensional vector field, i.e., numerical array of shape (..., N).
    Raises:
        ValueError if the input field is not of floating point type.
    '''
    if not np.issubdtype(field.dtype, np.floating):
        raise ValueError(
            f'The input field must be of floating point type. Got {field.dtype}.'
        )
    norms = np.linalg.norm(field, axis=-1)
    # pylint: disable=unsupported-assignment-operation
    field[norms == 0] = np.nan


def normalize(vector_field: NumericArray):
    '''
    Normalize in place a vector field wrt to the Euclidean norm.

    Zero vectors are turned into vectors with np.nan coordinates
    silently.

    Args:
        vector_field: vector field of floating point type and of shape (..., N)
         where N is the number of vector components.
    '''
    norm = np.linalg.norm(vector_field, axis=-1)
    norm = np.where(norm > 0, norm, 1.0)
    vector_field /= norm[..., np.newaxis]
    zero_to_nan(vector_field)


def normalized(vector_field: NumericArray):
    '''
    Normalize a vector field wrt to the Euclidean norm.

    Zero vectors are turned into vectors with np.nan coordinates
    silently.

    Args:
        vector_field: vector field of floating point type and of shape (..., N)
         where N is the number of vector components.
    Return:
        normalized_:
            vector field of unit vectors of the same shape and the same type as `vector_field`.
    '''
    normalized_ = voxcell.math_utils.normalize(vector_field)
    zero_to_nan(normalized_)
    return normalized_


def compute_blur_gradient(scalar_field: FloatArray, gaussian_stddev=3.0) -> FloatArray:
    '''
    Blurs a scalar field and returns its normalized gradient.

    A Gaussian filter (blur) with standard deviation `gaussian_stdev`
    is applied to the input field. The function returns the normalized
    gradient of the filtered field.

    Arguments:
        scalar_field: floating point scalar field defined over
            a 3D volume.
        gaussian_stddev: standard deviation of the Gaussian kernel used by the
            Gaussian filter.
    Returns:
        numpy.ndarray of float type. A 3D unit vector field over the underlying 3D volume
        of the input scalar field. This vector contains np.nan vectors if the normalization
        process encounters some zero vectors.
    Raises:
        ValueError if the input field is not of floating point type.
    '''
    if not np.issubdtype(scalar_field.dtype, np.floating):
        raise ValueError(
            f'The input field must be of floating point type. Got {scalar_field.dtype}.'
        )
    blurred = gaussian_filter(scalar_field, sigma=gaussian_stddev)
    gradient = np.array(np.gradient(blurred))
    gradient = np.moveaxis(gradient, 0, -1)
    normalize(gradient)
    return gradient


def quaternion_to_vector(quaternion_field: FloatArray) -> FloatArray:
    '''
    Rotate the reference vector (0.0, 1.0, 0.0) by the quaternions of `quaternion_field`.

    Arguments:
        quaternion_field: float array of shape (W, L, D, 4),
            the 4 quaternion coordinates are given by the last axis;
            in other words, it is a quaternionic vector field on a 3D volume.

    Returns:
        A 3D vector field on a 3D volume under the form of a numpy.ndarray of shape
        (W, L, D, 3).
    '''
    quaternion_field = quaternion_field.copy()
    zero_to_nan(quaternion_field)
    return quaternion.rotate_vectors(
        quaternion.from_float_array(quaternion_field), (0.0, 1.0, 0.0)
    )


def _quaternion_from_vectors(s: NumericArray, t: NumericArray) -> NumericArray:
    '''
    Returns the quaternion (s cross t) + (s dot t + |s||t|).

    This quaternion q maps s to t, i.e., qsq^{-1} = t.

    Args:
        s: numeric array of shape (3,) or (N, 3).
        t: numeric array of shape (N, 3) if s has two dimensions and its first dimension is N.
    Returns:
        Numeric array of shape (N, 4) where N is the first dimension of t.
        This data is interpreted as a 1D array of quaternions with size N. A quaternion is a 4D
        vector [w, x, y, z] where [x, y, z] is the imaginary part.
    '''
    w = np.matmul(s, np.array(t).T) + np.linalg.norm(s, axis=-1) * np.linalg.norm(
        t, axis=-1
    )
    return np.hstack([w[:, np.newaxis], np.cross(s, t)])


def vector_to_quaternion(vector_field: FloatArray) -> FloatArray:
    '''
    Find quaternions which rotate [0.0, 1.0, 0.0] to each vector in `vector_field`.

    A returned quaternion is of the form [w, x, y, z] where [x, y, z] is imaginary part and w the
     real part.
    The specific choice of returned quaternion is documented in _quaternion_from_vectors.

    Arguments:
        vector_field: field of floating point 3D unit vectors, i.e., a float array of shape
            (..., 3).

    Returns:
        numpy.ndarray of shape (..., 4) and of the same type as the input
        vector field.

    '''
    if not np.issubdtype(vector_field.dtype, np.floating):
        raise ValueError(
            f'The input field must be of floating point type. Got {vector_field.dtype}.'
        )
    quaternions = np.full(
        vector_field.shape[:-1] + (4,), np.nan, dtype=vector_field.dtype
    )
    non_nan_mask = ~np.isnan(np.linalg.norm(vector_field, axis=-1))
    quaternions[non_nan_mask] = _quaternion_from_vectors(
        [0.0, 1.0, 0.0], vector_field[non_nan_mask]
    )
    return quaternions


def split_into_halves(
    volume: NumericArray, halfway_offset: int = 0,
) -> Tuple[NumericArray, NumericArray]:
    '''
    Split input 3D volume into two halves along the z-axis.

    Args:
        volume: 3D numeric array.
            halfway_offset: Optional offset used for the
            splitting along the selected axis.
    Returns:
        tuple(left_volume, right_volume), the two halves of the
        input volume. Each has the same shape as `volume`.
        Voxels are zeroed for the `axis` values above, respectively
        below, the half of the `axis` dimension.
    '''
    z_halfway = volume.shape[2] // 2 + halfway_offset
    left_volume = volume.copy()
    left_volume[..., z_halfway:] = 0
    right_volume = volume.copy()
    right_volume[..., :z_halfway] = 0
    return left_volume, right_volume
