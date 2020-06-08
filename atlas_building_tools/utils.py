'''Generic Atlas files tools'''

from typing import Tuple, Union


import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

from voxcell import RegionMap  # type: ignore


# pylint: disable=invalid-name
FloatArray = Union[
    NDArray[float], NDArray[np.float16], NDArray[np.float32], NDArray[np.float64]
]
NumericArray = Union[NDArray[bool], NDArray[int], NDArray[float]]


def load_region_map(region_map: Union[str, dict, RegionMap]) -> RegionMap:
    '''
    Load a RegionMap object specified in one of three possible ways.

    Args:
        region_map: path to hierarchy.json, dict made of such a file or a RegionMap object.

    Returns:
        A true RegionMap object.
    '''
    if isinstance(region_map, str):
        region_map = RegionMap.load_json(region_map)
    elif isinstance(region_map, dict):
        region_map = RegionMap.from_dict(region_map)
    elif isinstance(region_map, RegionMap):
        return region_map
    else:
        raise TypeError('Cannot convert {} to RegionMap'.format(type(region_map)))
    return region_map


def get_region_mask(
    acronym: str, annotation: NDArray[int], region_map: Union[str, dict, RegionMap]
) -> RegionMap:
    '''
    Create a mask for the region defined by `acronym`.

    Args:
        acronym: the acronym of the region to mask. If it starts with @
                 the remainder is interprereted as a regexp.
        annotation: 3D array of region ids containing the region to mask.
        region_map: path to hierarchy.json, dict made of such a file or a RegionMap object.
    Returns:
       3D boolean array of the same shape as annotation.

    '''
    region_map = load_region_map(region_map)
    ids = list(region_map.find(acronym, 'acronym', with_descendants=True))
    return np.isin(annotation, ids)


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
        Voxels are zeroed for the z-values above, respectively
        below, the half of the z-dimension.
    '''
    z_halfway = volume.shape[2] // 2 + halfway_offset
    left_volume = volume.copy()
    left_volume[..., z_halfway:] = 0
    right_volume = volume.copy()
    right_volume[..., :z_halfway] = 0
    return left_volume, right_volume


def is_obtuse_angle(
    vector_field_1: NumericArray, vector_field_2: NumericArray
) -> NDArray[bool]:
    '''
    Returns a mask indicating which vector pairs form an obtuse angle.

    Arguments:
        vector_field_1: 3D vector field, i.e., numeric array of shape
            (M, N, ..., 3).
        vector_field_2: 3D vector field, i.e., numeric array of shape
            (M, N, ..., 3).
    Returns:
       Binary mask of shape (M, N, ...) indicating which pairs of vectors
        form an obtuse angle.
    '''
    return np.sum(vector_field_1 * vector_field_2, axis=-1) < 0
