'''
Function computing the direction vectors of the mouse isocortex
'''
import re
import logging
from typing import List, TYPE_CHECKING, Union

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

from voxcell.math_utils import minimum_aabb  # pylint: disable=ungrouped-imports

from atlas_building_tools.direction_vectors.algorithms.layer_based_direction_vectors import (
    compute_direction_vectors as layer_based_direction_vectors,
)
from atlas_building_tools.direction_vectors.algorithms.regiodesics import (
    find_regiodesics_exec_or_raise,
)
from atlas_building_tools.utils import load_region_map, get_region_mask

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import VoxelData, RegionMap  # type: ignore

L = logging.getLogger(__name__)
# The endings of names and acronyms in the 6 layers of the mouse isocortex are:
#  * 1, 2, 3, 4, 5
#  * 2/3
#  * 6a, 6b
# Reference: AIBS 1.json as of 2020/04.
# We search for these endings with the following regular expression:
LAYER_ENDINGS = '^([a-zA-Z]*-?[a-zA-Z]+)(?:[1-5]|2/3|6[ab])$'

# pylint: disable=fixme
# TODO: get layered subregions, excluding unrepresented
# including non-leaf represented from the hierarchy


def get_isocortical_regions(
    brain_regions: NDArray[int], region_map: Union[str, 'RegionMap']
) -> List[str]:
    '''
    Get the acronyms of all isocortical regions present in `brain_regions`.

    Args:
        brain_regions: 3D array of region identifiers containing the isocortex ids.
        region_map: path to the hierarchy.json file or a RegionMap to navigate the
            brain regions hierarchy.

    Returns:
        A list containing the isocortical region acronyms.

    Note: The output list may vary from one annotation file to the other.
    For the Mouse ccfv2 annotation with a resolution of 25um, 40 acronyms
    are returned. For the Mouse ccfv2 annotation of the same resolution,
    43 acronyms are returned.
    '''
    isocortex_mask = get_region_mask('Isocortex', brain_regions, region_map)
    ids = np.unique(brain_regions[isocortex_mask])
    region_map = load_region_map(region_map)
    acronyms = set()
    for id_ in ids:
        acronym = region_map.get(id_, 'acronym')
        search = re.search(LAYER_ENDINGS, acronym)
        if search is not None:
            acronym = search.group(1)
            acronyms |= {acronym}

    return sorted(list(acronyms))


def compute_direction_vectors(
    region_map: Union[str, dict, 'RegionMap'], brain_regions: 'VoxelData'
) -> NDArray[np.float32]:
    '''
    Compute the mouse isocortex direction vectors.

    Arguments:
        region_map: path to the json file containing atlas region hierarchy,
            or a dict, or a RegionMap object.
        brain_regions: VoxelData object containing the isocortex or a superset.

    Returns:
        Vector field of 3D unit vectors over the isocortex volume with the same shape
        as the input one. Voxels outside the Isocortex have np.nan coordinates.
    '''
    direction_vectors = np.full(brain_regions.shape + (3,), np.nan)
    region_map = load_region_map(region_map)
    # Get the highest-level regions of the isocortex: ACAd, ACAv, AId, AIp, AIv, ...
    # In the AIBS mouse ccfv3 annotation, there 43 isocortical regions.
    regions = get_isocortical_regions(brain_regions.raw, region_map)

    for region in regions:
        L.info('Computing direction vectors for region %s', region)
        region_mask = get_region_mask(region, brain_regions.raw, region_map)
        # pylint: disable=not-an-iterable
        aabb_slice = tuple(
            [
                slice(bottom, top + 1)
                for (bottom, top) in np.array(minimum_aabb(region_mask)).T
            ]
        )
        voxel_data = brain_regions.with_data(brain_regions.raw[aabb_slice])
        region_direction_vectors = layer_based_direction_vectors(
            region_map,
            voxel_data,
            {
                'source': [('acronym', '@.*6[b]$')],
                'inside': [('acronym', region)],
                'target': [('acronym', '@.*1$')],
            },
            algorithm='regiodesics',
            hemisphere_options={'set_opposite_hemisphere_as': 'target'},
            regiodesics_path=find_regiodesics_exec_or_raise('direction_vectors'),
        )
        direction_vectors[aabb_slice] = region_direction_vectors
        del region_direction_vectors

    # Warns about generated NaN vectors within Isocortex
    nans = np.mean(
        np.isnan(
            direction_vectors[
                get_region_mask('Isocortex', brain_regions.raw, region_map)
            ]
        )
    )
    if nans > 0:
        L.warning('NaN direction_vectors in {:.5%} of isocortical voxels'.format(nans))

    return direction_vectors
