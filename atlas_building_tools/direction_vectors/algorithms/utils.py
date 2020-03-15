'''Low-level tools for the computation of direction vectors'''
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import quaternion

import voxcell

NP_FLOAT_TYPES = {
    float,
    np.dtype('float'),
    np.dtype('float16'),
    np.dtype('float32'),
    np.dtype('float64'),
}


def zero_to_nan(field):
    '''
    Turns, in place, the zero vectors of a vector field into NaN vectors.

    Zero vectors are replaced, in place, by vectors with np.nan coordinates.

    Note: This function is used to invalidate zero vectors or zero quaternions as a zero vector
    cannot be used to define a direction or an orientation.
    In addition, it allows the multiplication of an invalid quaternion, i.e., a quaternion with
     NaN coorinates with a vector (the output is a NaN vector) without raising exception.

    Args:
        field(numpy.ndarray): N-dimensional vector field, i.e., numerical array of shape (..., N).
    Raises:
        ValueError if the input field is not of floating point type.
    '''
    if field.dtype not in NP_FLOAT_TYPES:
        raise ValueError(
            f'The input field must be of floating point type. Got {field.dtype}.'
        )
    norms = np.linalg.norm(field, axis=-1)
    # pylint: disable=unsupported-assignment-operation
    field[norms == 0] = np.nan


def normalize(vector_field):
    '''
    Normalize in place a vector field wrt to the Euclidean norm.

    Zero vectors are turned into vectors with np.nan coordinates
    silently.

    Args:
        vector_field(numpy.ndarray): vector field of floating point type and of shape (..., N)
         where N is the number of vector components.
    '''
    norm = np.linalg.norm(vector_field, axis=-1)
    norm = np.where(norm > 0, norm, 1.0)
    vector_field /= norm[..., np.newaxis]
    zero_to_nan(vector_field)


def normalized(vector_field):
    '''
    Normalize a vector field wrt to the Euclidean norm.

    Zero vectors are turned into vectors with np.nan coordinates
    silently.

    Args:
        vector_field(numpy.ndarray): vector field of floating point type and of shape (..., N)
         where N is the number of vector components.
    Return:
        normalized_(numpy.ndarray):
            vector field of unit vectors of the same shape and the same type as `vector_field`.
    '''
    normalized_ = voxcell.math_utils.normalize(vector_field)
    zero_to_nan(normalized_)
    return normalized_


def compute_blur_gradient(scalar_field, gaussian_stddev=3.0):
    '''
    Blurs a scalar field and returns its normalized gradient.

    A Gaussian filter (blur) with standard deviation `gaussian_stdev`
    is applied to the input field. The function returns the normalized
    gradient of the filtered field.

    Arguments:
        scalar_field(numpy.ndarray): floating point scalar field defined over
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
    if scalar_field.dtype not in NP_FLOAT_TYPES:
        raise ValueError(
            f'The input field must be of floating point type. Got {scalar_field.dtype}.'
        )
    blurred = gaussian_filter(scalar_field, sigma=gaussian_stddev)
    gradient = np.array(np.gradient(blurred))
    gradient = np.moveaxis(gradient, 0, -1)
    normalize(gradient)
    return gradient


def quaternion_to_vector(quaternion_field):
    '''
    Rotate the reference vector (0.0, 1.0, 0.0) by the quaternions of `quaternion_field`.

    Arguments:
        quaternion_field(numpy.ndarray): float array of shape (W, L, D, 4),
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


def vector_to_quaternion(vector_field):
    '''
    Find quaternions which rotate [0.0, 1.0, 0.0] to each vector in
    `vector_field`.

    A returned quaternion is of the form:
    [cos(angle / 2), x * sin(angle / 2), y * sin(angle / 2), z * (angle / 2)]
    This represents a rotation of `angle` around vector `axis = [x, y, z]`
    where `axis` is the normalized cross product of [0.0, 1.0, 0.0] and the input vector.
    If angle = 0, return the identity quaternion [1.0, 0.0, 0.0, 0.0].
    If angle = pi, choose x-axis as axis, i.e., return [0.0, 1.0, 0.0, 0.0].

    Arguments:
        vector_field: np.array of floating point 3D unit vectors, shape (..., 3)
            (e.g. (3,), (N, 3), (N, M, 3) etc.)

    Returns:
        numpy.ndarray of shape (..., 4) and of the same type as the input
        vector field.

    '''
    quaternions = np.zeros(vector_field.shape[:-1] + (4,), dtype=vector_field.dtype)
    # vector_field's vectors and [0.0, 1.0, 0.0] are unit vectors,
    # so [0.0, 1.0, 0.0] (dot) vector_field = cos(angle)
    cos_angle = vector_field[..., 1]  # scalar product with [0.0, 1.0, 0.0]

    # cos(angle / 2.0) = sqrt((1 + cos(angle)) / 2.0)
    # we will need 2.0 * cos(angle / 2.0) later, 2.0 cos(angle/2.0) = sqrt(2.0 + 2.0 * cos(angle))
    # we will need (2.0  * cos(angle / 2.0))^2 to identify cases where angle = pi
    cos_half_angle_x2_squared = 2.0 + 2.0 * cos_angle
    # when angle = pi, cos(angle / 2.0) = 0, take only cases where this is not so
    angle_less_than_pi = cos_half_angle_x2_squared > 0
    # as 0 <= angle <= pi, cos_half_angle_x2 >= 0
    cos_half_angle_x2 = np.sqrt(cos_half_angle_x2_squared[angle_less_than_pi])[
        ..., np.newaxis
    ]
    # [0.0, 1.0, 0.0] cross vector_field = axes * sin(angle)
    # sin(angle / 2.0) = sin(angle) / (2.0 * cos(angle / 2.0))
    # axes * sin(angle / 2.0) = [0.0, 1.0, 0.0] * vector_field / (2.0 * cos(angle / 2.0))
    axes = (
        np.cross([0.0, 1.0, 0.0], vector_field[angle_less_than_pi]) / cos_half_angle_x2
    )
    # quaternion =
    # [cos(angle / 2.0), x * sin(angle / 2.0), y * sin(angle / 2.0), z * sin(angle / 2.0)]
    quaternions[angle_less_than_pi] = np.concatenate(
        ((0.5 * cos_half_angle_x2), axes), axis=-1
    )
    # Choose x-axis:
    # quaternion = [cos(pi/ 2.0), 1 sin(pi/ 2.0), 0.0, 0.0] = [0.0, 1.0, 0.0, 0.0]
    quaternions[~angle_less_than_pi] = [0.0, 1.0, 0.0, 0.0]
    return quaternions
