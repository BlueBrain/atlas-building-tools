'''Functions to compute the overall mouse brain cell density.'''

from typing import Dict, Optional
import numpy as np
from nptyping import NDArray  # type: ignore

from voxcell import RegionMap, VoxelData  # type: ignore
from atlas_building_tools.densities.cell_counts import cell_counts
from atlas_building_tools.densities.utils import (
    compensate_cell_overlap,
    get_group_ids,
    get_region_masks,
)
from atlas_building_tools.densities.soma_radius import apply_soma_area_correction


def fix_purkinje_layer_density(
    region_map: 'RegionMap',
    annotation_raw: NDArray[int],
    cell_density: NDArray[float],
    region_masks: Dict[str, NDArray[bool]],
) -> NDArray[float]:
    '''
    Assign a constant number of cells to the voxels sitting both in Cerebellum and the
    Purkinje layer.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
        annotation_raw: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.
        cell_density: float array of shape (W, H, D) with non-negative entries. The input
            overall cell density to be corrected.
        cell_counts: a dictionary whose keys are region group names and whose values are
            integer cell counts.
        region_masks: A dictionary whose keys are region group names and whose values are
            the boolean masks of these groups. Each boolean array is of shape (W, H, D) and
            encodes which voxels belong to the corresponding group.

    Returns:
        float array of shape (W, H, D) with non-negative entries.
        The array represents the overall cell density, satifying the constraint that
        the Purkinje layer has a constant number of cells per voxel.
    '''

    group_ids = get_group_ids(region_map)
    purkinje_layer_mask = np.isin(annotation_raw, list(group_ids['Purkinje layer']))
    # Force Purkinje Layer regions of the Cerebellum group to have a constant density
    # equal to the average density of the complement.
    # pylint: disable=fixme
    # TODO: The Purkinje cell diameter is 25um. A correction of cell densities is required for the
    #  10um resolution.
    cerebellum_purkinje_layer_mask = np.logical_and(
        region_masks['Cerebellum group'], purkinje_layer_mask
    )
    cerebellum_wo_purkinje_layer_mask = np.logical_and(
        region_masks['Cerebellum group'], ~purkinje_layer_mask
    )
    purkinje_layer_count = np.count_nonzero(cerebellum_purkinje_layer_mask)
    cell_density[cerebellum_purkinje_layer_mask] = np.sum(
        cell_density[cerebellum_wo_purkinje_layer_mask]
    ) / (cell_counts()['Cerebellum group'] - purkinje_layer_count)

    return cell_density


def compute_cell_density(
    region_map: RegionMap,
    annotation: VoxelData,
    nissl: NDArray[float],
    soma_radii: Optional[Dict[int, str]] = None,
) -> NDArray[float]:
    '''
    Compute the overall cell density based on Nissl staining and cell counts from literature.

    The input Nissl stain intensity volume of AIBS is assumed to be depend linearly on the cell
    density (number of cells per voxel) when restrictied to a mouse brain region.
    It is turned into an actual density field complying with the cell counts of several
    regions.

    The input array `nissl` is modified in-line and returned by the function.

    Note: Nissl staining, according to https://en.wikipedia.org/wiki/Nissl_body, is a
    "method (that) is useful to localize the cell body, as it can be seen in the soma and dendrites
     of neurons, though not in the axon or axon hillock."
    Here is the assumption on Nissl staining volume, as written in the introduction of
    "A Cell Atlas for the Mouse Brain" by C. Ero et al., 2018.
    "We assumed the stained intensity of Nissl and other genetic markers to be a good indicator
     of soma density specific to the population of interest, without significantly staining axons
      and dendrites."

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
        annotation_raw: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.
        nissl: float array of shape (W, H, D) with non-negative entries. The input
            Nissl stain intensity.
        soma_radii: Optional dictionary whose keys are structure IDs (AIBS region identifiers)
            and whose values are the average soma radii corresponding to these regions.
            Defaults to None. If specified, the input Nissl stain intensity is adjusted by
            taking these radii into account.

    Returns:
        float array of shape (W, H, D) with non-negative entries. The returned array is a
        transformation of `nissl` which is modified in-line.
        The overall mouse brain cell density, respecting region-specific cell counts provided
        by the scientific literature as well as the Purkinje layer constraint of a constant number
        of cells per voxel.
    '''

    nissl = compensate_cell_overlap(
        nissl, annotation.raw, gaussian_filter_stdv=1.0, copy=False
    )

    if soma_radii:
        apply_soma_area_correction(region_map, annotation.raw, nissl, soma_radii)

    group_ids = get_group_ids(region_map)
    region_masks = get_region_masks(group_ids, annotation.raw)
    cell_density = fix_purkinje_layer_density(
        region_map, annotation.raw, nissl, region_masks
    )
    for group, mask in region_masks.items():
        cell_density[mask] = nissl[mask] * (cell_counts()[group] / np.sum(nissl[mask]))

    return cell_density
