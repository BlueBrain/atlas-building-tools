'''Generic sript for the computation of voxel-to-layer distances wrt
to direction vectors, a.k.a placement hints, in a layered atlas.
'''

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from nptyping import NDArray  # type: ignore

import numpy as np  # type: ignore

from atlas_building_tools.placement_hints.layered_atlas import (
    compute_distances_to_layer_meshes,
)

from atlas_building_tools.distances.distances_to_meshes import (
    report_problems,
    interpolate_nan_voxels,
)

if TYPE_CHECKING:  # pragma: no cover
    from atlas_building_tools.placement_hints.layered_atlas import LayeredAtlas
    from voxcell import VoxelData, RegionMap  # type: ignore


DistanceInfo = Dict[str, Union['LayeredAtlas', NDArray[float], NDArray[bool]]]
DistancesReport = Dict[str, float]


# pylint: disable=too-many-arguments
def compute_placement_hints(
    region_map: 'RegionMap',
    annotation: 'VoxelData',
    region_acronym: str,
    layers_regexp: List[str],
    direction_vectors: NDArray[float],
    max_thicknesses: Optional[List[float]] = None,
    flip_direction_vectors: bool = False,
    has_hemispheres: bool = True,
) -> Tuple[DistanceInfo, DistancesReport, NDArray[bool]]:
    '''
    Compute the placement hints for a laminar region of the mouse brain.

    Args:
        region_map: object to navigate brain regions hierarchy.
        region_acronym: acronym of the region for which the computation is requested.
            Example: 'CA1', 'Isocortex'.
        layer_regexps: list of regular expressions defining the layers of `region_acronym`.
        annotation: annotated 3D volume holding the full mouse brain atlas.
        direction_vectors: unit vector field of shape (W, H, D, 3) if `annotation`'s array
            is of shape (W, H, D).
        max_thicknesses: (optional) thicknesses of `region_acronym` layers.
            Defaults to None, i.e., there will be no validity check with input from literature.
        flip_direction_vectors: If True, the input direction vectors are negated before use.
            This is required if direction vectors flaw from the top layer (shallowest) to the
            bottom layer (deepest). Otherwise, they are left unchanged. Defaults to false.
        has_hemispheres: (optional) If True, split the volume into halves along the z-axis and
            handle each of theses 'hemispheres' separately. Otherwise the whole volume is handled.
            Defaults to True.

    Returns:
        Tuple with the following items.
        distances_info: dict with the following entries.
            layered_atlas: LayeredAtlas instance of the requested acronym.
            obtuse_angles: 3D binary mask indicating which voxels have rays
                intersecting a layer boundary with an obtuse angle. The direction vectors
                of such voxels are considered as problematic.
            distances_to_layer_meshes(numpy.ndarray): 4D float array of shape
                (number of layers + 1, W, H, D) holding the distances from
                voxel centers to the upper boundaries of layers wrt to voxel direction vectors.
        distances_report:
            dict reporting the proportion of voxels subject to each
            distance-related problem, see distances.distance_to_meshes.report_problems
                doc.
        problematic_volume: 3D binary mask of the voxels with at least one
            distance-related problem. See distances.distance_to_meshes.report_problems
                doc.
    '''
    distances_info = compute_distances_to_layer_meshes(
        region_acronym,
        annotation,
        region_map,
        direction_vectors,
        layers_regexp,
        flip_direction_vectors=flip_direction_vectors,
        has_hemispheres=has_hemispheres,
    )
    atlas = distances_info['layered_atlas']
    distances_to_meshes = distances_info['distances_to_layer_meshes']
    distances_report, problematic_volume = report_problems(
        distances_to_meshes,
        distances_info['obtuse_angles'],
        atlas.region,
        max_thicknesses=max_thicknesses,
    )
    interpolate_nan_voxels(
        distances_to_meshes, np.logical_and(atlas.region.raw, ~problematic_volume),
    )
    return distances_info, distances_report, problematic_volume
