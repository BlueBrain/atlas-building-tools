"""
Function computing the direction vectors of the mouse thalamus
"""
import logging
import warnings
from typing import TYPE_CHECKING, Union

import numpy as np
from nptyping import NDArray  # type: ignore
from scipy.ndimage import correlate  # type: ignore
from scipy.ndimage.morphology import generate_binary_structure  # type: ignore

from atlas_building_tools.direction_vectors.algorithms.layer_based_direction_vectors import (
    direction_vectors_for_hemispheres,
)
from atlas_building_tools.utils import get_region_mask, load_region_map

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore

L = logging.getLogger(__name__)
logging.captureWarnings(True)


def _get_common_outer_boundary(mask: NDArray[bool], sub_mask: NDArray[bool]) -> NDArray[bool]:
    """
    Get the mask of the voxels outside `mask` which are both
    in the outer boundary of `mask` and of `sub_mask`.

    The mask `submask`is assumed to represent a voxel subset of
    `mask`.

    Args:
        mask: boolean array of shape (W, H, D) where W, H and D are integer dimensions.
            This array holds the mask of a 3D region.
        sub_mask: boolean array of the same shape (W, H, D) as `mask`.
            mask of the voxels of a 3D sub region of the region defined by `mask`.

    Returns:
        boolean array of shape (W, H, D) representing the common outer boundary of
        `mask` and `sub_mask`.
    """
    filter_ = generate_binary_structure(3, 1).astype(int)

    return np.logical_and(
        np.logical_and(correlate(sub_mask, filter_), correlate(mask, filter_)), ~mask
    )


def compute_direction_vectors(
    region_map: Union[str, dict, "RegionMap"], brain_regions: "VoxelData"
) -> NDArray[np.float32]:
    """
    Compute the mouse thalamus direction vectors.

    Arguments:
        region_map: path to the json file containing atlas region hierarchy,
            or a dict, or a RegionMap object.
        brain_regions: VoxelData object containing the mouse thalamus or a superset.

    Returns:
        Vector field of 3D unit vectors over the thalamus volume with the same shape
        as the input one. Voxels outside the thalamus have np.nan coordinates.
    """
    region_map = load_region_map(region_map)
    thalamus_mask = get_region_mask("TH", brain_regions.raw, region_map)
    reticular_nucleus_mask = get_region_mask("RT", brain_regions.raw, region_map)
    reticular_nucleus_complement_mask = np.logical_and(thalamus_mask, ~reticular_nucleus_mask)
    common_outer_boundary_mask = _get_common_outer_boundary(thalamus_mask, reticular_nucleus_mask)
    landscape = {
        "source": np.zeros_like(thalamus_mask),
        "inside": reticular_nucleus_complement_mask,
        "target": common_outer_boundary_mask,
    }
    ratio = (
        brain_regions.voxel_dimensions[0] / 25
    )  # tuning based on tests with the 25 um resolution
    rt_complement_direction_vectors = direction_vectors_for_hemispheres(
        landscape,
        algorithm="simple_blur_gradient",
        hemisphere_options={"set_opposite_hemisphere_as": "source"},
        # The constants below have been derived empirically by Hugo Dictus
        sigma=ratio * 18.0,
        source_weight=-2.0 * 0.9999999,
        target_weight=2 * 0.1111111,
    )
    landscape = {
        "source": reticular_nucleus_complement_mask,
        "inside": reticular_nucleus_mask,
        "target": common_outer_boundary_mask,
    }
    rt_direction_vectors = direction_vectors_for_hemispheres(
        landscape,
        algorithm="simple_blur_gradient",
        hemisphere_options={"set_opposite_hemisphere_as": "source"},
        # The constants below have been derived empirically by Hugo Dictus
        sigma=ratio * 5.0,
        source_weight=-1,
        target_weight=1,
    )

    direction_vectors = np.full(rt_direction_vectors.shape, np.nan, dtype=np.float32)
    direction_vectors[reticular_nucleus_complement_mask] = rt_complement_direction_vectors[
        reticular_nucleus_complement_mask
    ]
    direction_vectors[reticular_nucleus_mask] = rt_direction_vectors[reticular_nucleus_mask]

    # Warns about generated NaN vectors within thalamus
    nans = np.mean(np.isnan(direction_vectors[thalamus_mask]))
    if nans > 0.0:
        warnings.warn("NaN direction vectors in {:.5%} of thalamic voxels".format(float(nans)))

    return direction_vectors