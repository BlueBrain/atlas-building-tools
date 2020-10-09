'''
Algorithm splitting the layer 2/3 of the mouse isocortex into
layer 2 and layer 3.

This algorithm relies on the computation
of voxel-to-(layer boundary mesh) distances wrt to direction vectors.
'''

import copy
from itertools import count, islice
from typing import Any, Dict, Iterator, Set, TYPE_CHECKING
from nptyping import NDArray  # type: ignore

import numpy as np  # type: ignore

# pylint: disable=ungrouped-imports
from voxcell import RegionMap  # type: ignore

from atlas_building_tools.exceptions import AtlasBuildingToolsError

if TYPE_CHECKING:
    from voxcell import VoxelData  # type: ignore

# The following constants documented
# Section in Section 5.1.1.4 of the release report
# of the Neocortex Tissue Reconstruction.
DEFAULT_L2_THICKNESS = 95.10
DEFAULT_L3_THICKNESS = 225.3199
DEFAULT_RATIO = DEFAULT_L2_THICKNESS / (DEFAULT_L2_THICKNESS + DEFAULT_L3_THICKNESS)

HierarchyDict = Dict[str, Any]


def get_isocortex_hierarchy(allen_hierachy: HierarchyDict):
    '''
    Extract the hierarchy dict of the iscortex from AIBS hierarchy dict.
    Args:
        allen_hierarchy: AIBS hierarchy dict instantiated from
            http://api.brain-map.org/api/v2/structure_graph_download/1.json.
    '''
    error_msg = 'Wrong input. The AIBS 1.json file is expected.'
    if 'msg' not in allen_hierachy:
        raise AtlasBuildingToolsError(error_msg)

    hierarchy = allen_hierachy['msg'][0]
    try:
        while hierarchy['acronym'] != 'Isocortex':
            hierarchy = hierarchy['children'][0]
    except KeyError as error:
        raise AtlasBuildingToolsError(error_msg) from error

    return hierarchy


def create_id_generator(region_map: 'RegionMap') -> Iterator[int]:
    '''Create an identifiers generator.

    The generator produces an identifier which is different from all
    the previous ones and from the identifiers in use in `self.region_map`.

    Args:
        region_map: map to navigate the brain region hierarchy.

    Return:
        iterator providing the next id.
    '''
    last = max(region_map.find('root', attr='acronym', with_descendants=True))
    return count(start=last + 1)


def edit_hierarchy(hierarchy: HierarchyDict, layer_3_new_ids: Dict[int, int]) -> None:
    '''
    Edit in place layer 2/3 into 2 and 3 within the hierarchy dict.

    Acronyms and names ending with 2/3 are changed.
    A new identifier is used when a layer 2/3 identifier is used to annotate voxels both in
    layer 2 and layer 3.

    Note: The isocortex should not have any identifier whose acronym refers to
    layer 3. Isocortex identifiers correponding to layer 2/3 should all be leaf
    region identifiers. These observations are based on
    http://api.brain-map.org/api/v2/structure_graph_download/1.json.

    Args:
        hierarchy: brain regions hierarchy dict.
        layer_3_new_ids: dict whose
            - keys are the region identifiers on layer 2/3 for which new ids
                corresponding to layer 3 should be created,
            - values are the new identifiers to be used.
    '''
    children = islice(hierarchy['children'], 0, len(hierarchy['children']))
    # The children list can be extended during iteration.
    for child in children:
        if child['acronym'].endswith('2/3'):
            assert child['name'].endswith('2/3')
            # For layer 2, ids are preserved.
            child['acronym'] = child['acronym'][:-2]
            child['name'] = child['name'][:-2]
            if child['id'] in layer_3_new_ids.keys():
                new_child = copy.deepcopy(child)
                new_child['id'] = layer_3_new_ids[child['id']]
                new_child['acronym'] = child['acronym'][:-1] + '3'
                new_child['name'] = child['name'][:-1] + '3'
                hierarchy['children'].append(new_child)
        edit_hierarchy(child, layer_3_new_ids)


def _edit_layer_23(
    hierarchy: HierarchyDict,
    region_map: RegionMap,
    volume: NDArray[int],
    layer_23_ids: Set[int],
    layer_3_mask: NDArray[bool],
) -> None:
    '''
    Edit layer 2/3 into 2 and 3.

    Edit in place `hierarchy` and `volume` to perform the splitting
    of layer 2/3 into layer 2 and layer 3.

    Args:
        hierarchy: brain regions hierarchy dict.
        region_map: map to navigate the brain regions hierarchy.
        volume: whole brain annotated volume.
            - keys are the region identifiers on layer 2/3
                which should be replaced by new ids corresponding to layer 3,
            - values are the new identifiers to be used.
        layer_23_ids: the set of all layer 2/3 identifiers,
            i.e., the identifiers whose corresponding acronyms and names
            end with '2/3'.
        layer_3_mask: binary mask of the voxels sitting in layer 3.
    '''
    id_generator = create_id_generator(region_map)
    layer_3_new_ids = {}
    for id_ in layer_23_ids:
        change_to_3 = np.logical_and(volume == id_, layer_3_mask)
        if np.any(change_to_3):
            new_id = next(id_generator)
            volume[change_to_3] = new_id
            layer_3_new_ids[id_] = new_id
    isocortex_hierarchy = get_isocortex_hierarchy(hierarchy)
    edit_hierarchy(isocortex_hierarchy, layer_3_new_ids)


def _compute_distance_to_boundary(
    start: NDArray[int],
    volume_mask: NDArray[bool],
    direction_vectors: NDArray[float],
    direction: int,
) -> float:
    '''
    Compute the distance to the volume boundary following the specified direction.

    Args:
        start: 3D index of the voxel for which the computation is queried.
        volume_mask: boolean array of shape (W, H, D) holding the mask of a volume.
        direction_vectors: float array of shape (W, H, D, 3) holding a field of 3D unit vectors
            defined over the input volume.
        direction: Either 1 (forward) or -1 (backward). The value 1 means that
            we follow the direction vectors. The value -1 means that we follow the negated
            direction vectors.

    Returns:
        the distance to the boundary of the volume along direction vectors.
    '''

    def _is_in_volume(voxel: NDArray[int], volume_mask: NDArray[bool]) -> bool:
        '''
        Indicates whether `voxel`lies in the maske volume.

        Args:
            voxel: 3D index of the input voxel.
            volume_mask: boolean mask of shape (W, H, D); mask of a 3D brain region.
        Retuns:
            True, if `voxel` is the `volume_mask`, False otherwise.
        '''
        return volume_mask[tuple(voxel.astype(int))]

    delta = 0.5
    distance = 0.0
    current = start.copy()
    while _is_in_volume(current, volume_mask):
        current = (
            current + direction_vectors[tuple(current.astype(int))] * delta * direction
        )
        distance += delta

    return distance


def _get_deepest_layer_mask(
    volume_mask: NDArray[bool],
    thickness_ratio: float,
    direction_vectors: NDArray[float],
):
    '''
    Get the mask of the voxels with cortical depth at least `thickness_ratio` of the full thickness.

    The distance is taken from the top of the volume to the voxel along fiber tracts.
    Fiber tracts are approximated by polygonal lines built out of the input direction vectors.

    Args:
        volume_mask: boolean array of shape (W, H, D) holding the mask of a volume
            to split according to the specified `thickness_ratio`.
        thickness_ratio: ratio used to determine if a voxel belongs to the returned mask.
        direction_vectors: float array of shape (W, H, D, 3) holding a field of 3D unit vectors
            defined over the input volume.

    Returns:
      boolean array of shape (W, H, D), mask of the voxels whose cortical depth is at least
      `thickness_ratio` of the volume thickness.

    '''
    layer_mask = np.zeros_like(volume_mask)
    voxels = np.array(np.where(volume_mask)).T

    for voxel in voxels:  # pylint: disable=not-an-iterable
        forward_distance = _compute_distance_to_boundary(
            voxel, volume_mask, direction_vectors, 1
        )
        backward_distance = _compute_distance_to_boundary(
            voxel, volume_mask, direction_vectors, -1
        )
        ratio = forward_distance / (forward_distance + backward_distance)
        layer_mask[tuple(voxel)] = bool(ratio >= thickness_ratio)

    return layer_mask


def split(
    hierarchy: HierarchyDict,
    annotation: 'VoxelData',
    direction_vectors: NDArray[float],
    thickness_ratio: float = DEFAULT_RATIO,
) -> None:
    '''
    Splits in place layer 2/3 into layer 2 and layer 3 based on a relative thickness ratio.

    The ratio is used to determined which voxels of layer 2/3 should sit in
    layer 2 or layer 3.

    Voxels initially sitting in layer 2 will continue to use the same identifiers.
    Voxels sitting in layer 2/3 and identified as member of layer 3
    will use new identifiers.

    Note: In the isocortex, the voxels corresponding to a name or an acronym ending with
    'layer 2' are left unchanged. No voxels of the isocortex should correspond to a name
    or an acronym ending with 'layer 3'.

    Args:
        hierarchy: brain regions hierarchy dict.
        region_map: map to navigate the brain regions hierarchy.
        annotation: whole brain annotation.
        direction_vectors: array of shape (W, L, D, 3) if the shape of
            `annotation.raw` is (W, L, D, 3). This array holds a field of unit
            3D vectors reflecting the directions of the isocortex fibers.
        thickness_ratio: ratio of average layer thicknesses, i.e.,
            layer_2 thickness / (layer_2 thickness + layer_3 thickness)
            Defaults to `DEFAULT_RATIO`.
    '''
    assert (
        'msg' in hierarchy
    ), 'Wrong hierarchy input. The AIBS 1.json file is expected.'
    region_map = RegionMap.from_dict(hierarchy['msg'][0])
    isocortex_ids = region_map.find('Isocortex', attr='acronym', with_descendants=True)
    layers_2_and_3_ids = isocortex_ids & region_map.find(
        '@.*2(/3)?$', attr='acronym', with_descendants=True
    )

    layer_3_mask = _get_deepest_layer_mask(
        np.isin(annotation.raw, list(layers_2_and_3_ids)),
        thickness_ratio,
        direction_vectors,
    )
    layer_23_ids = isocortex_ids & region_map.find(
        '@.*2/3$', attr='acronym', with_descendants=True
    )

    _edit_layer_23(
        hierarchy, region_map, annotation.raw, layer_23_ids, layer_3_mask,
    )
