'''Generic Atlas files tools'''

from typing import Union

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

from voxcell import RegionMap  # type: ignore


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
