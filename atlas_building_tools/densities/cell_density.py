"""Functions to compute the overall mouse brain cell density.
"""

from typing import Dict, Optional

import numpy as np
from nptyping import NDArray  # type: ignore
from voxcell import RegionMap  # type: ignore

from atlas_building_tools.densities.cell_counts import cell_counts
from atlas_building_tools.densities.soma_radius import apply_soma_area_correction
from atlas_building_tools.densities.utils import (
    compensate_cell_overlap,
    get_group_ids,
    get_region_masks,
)


def fix_purkinje_layer_intensity(
    region_map: "RegionMap",
    annotation: NDArray[int],
    region_masks: Dict[str, NDArray[bool]],
    cell_intensity: NDArray[float],
) -> None:
    """
    Assign a constant number of cells to the voxels sitting both in Cerebellum and the
    Purkinje layer.

    The array `cell_intensity` is modified in place.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
        annotation: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.
        region_masks: A dictionary whose keys are region group names and whose values are
            the boolean masks of these groups. Each boolean array is of shape (W, H, D) and
            encodes which voxels belong to the corresponding group.
        cell_intensity: float array of shape (W, H, D) with non-negative entries. The overall cell
            intensity to be corrected. The array `cell_intensity` is modified in place, in such a
            way that the Purkinje layer has a constant intensity value.
    """

    group_ids = get_group_ids(region_map)
    purkinje_layer_mask = np.isin(annotation, list(group_ids["Purkinje layer"]))
    # Force Purkinje Layer regions of the Cerebellum group to have a constant intensity
    # equal to the average intensity of the complement.
    # pylint: disable=fixme
    # TODO: The Purkinje cell diameter is 25um. A correction of cell densities is required for the
    #  10um resolution.
    cerebellum_purkinje_layer_mask = np.logical_and(
        region_masks["Cerebellum group"], purkinje_layer_mask
    )
    cerebellum_wo_purkinje_layer_mask = np.logical_and(
        region_masks["Cerebellum group"], ~purkinje_layer_mask
    )
    purkinje_layer_count = np.count_nonzero(cerebellum_purkinje_layer_mask)
    cell_intensity[cerebellum_purkinje_layer_mask] = np.sum(
        cell_intensity[cerebellum_wo_purkinje_layer_mask]
    ) / (cell_counts()["Cerebellum group"] - purkinje_layer_count)


def compute_cell_density(
    region_map: RegionMap,
    annotation: NDArray[int],
    voxel_volume: float,
    nissl: NDArray[float],
    soma_radii: Optional[Dict[int, str]] = None,
) -> NDArray[float]:
    """
    Compute the overall cell density based on Nissl staining and cell counts from literature.

    The input Nissl stain intensity volume of AIBS is assumed to be depend linearly on the cell
    density (number of cells per voxel) when restricted to a mouse brain region.
    It is turned into an actual density field complying with the cell counts of several
    regions.

    The input array `nissl` is modified in-line and returned by the function.

    Note: Nissl staining, according to https://en.wikipedia.org/wiki/Nissl_body, is a
    "method (that) is useful to localize the cell body, as it can be seen in the soma and dendrites
    of neurons, though not in the axon or axon hillock." Here is the assumption on Nissl staining
    volume, as written in the introduction of
    "A Cell Atlas for the Mouse Brain" by C. Ero et al., 2018", page 3:
    "We assumed the stained intensity of Nissl and other genetic markers to be a good indicator
    of soma density specific to the population of interest, without significantly staining axons
    and dendrites."

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
        annotation: an integer array of shape (W, H, D) which encloses the AIBS annotation of
            the whole mouse brain. The integers W, H and D are the integer dimensions of the array.
        voxel_volume: the common volume of a voxel associated to any of the input arrays.
        nissl: float array of shape (W, H, D) with non-negative entries. The input
            Nissl stain intensity.
        soma_radii: (Optional) dictionary whose keys are structure IDs (AIBS region identifiers)
            and whose values are the average soma radii corresponding to these regions.
            Defaults to None. If specified, the input Nissl stain intensity is adjusted by
            taking these radii into account.

    Returns:
        float array of shape (W, H, D) with non-negative entries. The returned array is a
        transformation of `nissl` which is modified in-line. It represents the overall mouse brain
        cell density, expressed in number of cells per mm^3. It is compliant with several
        region-specific cell counts provided by the scientific literature as well as the Purkinje
        layer constraint of a constant number of cells per voxel.
    """

    nissl = compensate_cell_overlap(nissl, annotation, gaussian_filter_stdv=1.0, copy=False)

    if soma_radii:
        apply_soma_area_correction(region_map, annotation, nissl, soma_radii)

    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation)
    fix_purkinje_layer_intensity(region_map, annotation, region_masks, nissl)
    for group, mask in region_masks.items():
        nissl[mask] = nissl[mask] * (cell_counts()[group] / np.sum(nissl[mask]))

    return nissl / voxel_volume
