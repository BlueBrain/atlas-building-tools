'''Functions to compute glia cell densities.'''

from typing import Dict, Set, TYPE_CHECKING
import numpy as np
from nptyping import NDArray  # type: ignore

from atlas_building_tools.densities.utils import (
    constrain_density,
    get_group_ids,
    compensate_cell_overlap,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap  # type: ignore


def compute_glia_density(
    glia_cell_count: int,
    group_ids: Dict[str, Set[int]],
    annotation_raw: NDArray[float],
    glia_density: NDArray[float],
    cell_density: NDArray[float],
    copy: bool = True,
) -> NDArray[float]:
    '''
    Compute the overall glia cell density using a prescribed cell count and the overall cell
    density as an upper bound constraint.

    Further constraints are imposed:
        * voxels of fiber tracts are assigned the largest possible cell density
        * voxels lying both in the Cerebellum and in the Purkinje layer are assigned a zero density
         value.

    Args:
        glia_cell_count: glia cell count found in the scientific literature.
        group_ids: dictionary whose keys are group names and whose values are
            sets of AIBS structure ids. These groups are used to identify the voxels
            with maximum density (fiber tracts) and those of zero density (Purkinje layer).
        annotation_raw: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.
        glia_density: float array of shape (W, H, D) with non-negative entries. The input
            glia density obtained by averaging different marker datasets.
        cell_density: float array of shape (W, H, D) with non-negative entries. The overall
            cell density of the mouse brain.
        copy: If True, the input `glia_density` array is copied. Otherwise it is modified in-place
             and returned by the function.

    Returns:
        float array of shape (W, H, D) with non-negative entries.
        The overall glia cell density, respecting the constraints imposed by the `cell_density`
        upper bound, the input cell count, as well as region specific hints
        (fiber tracts and Purkinje layer).
    '''

    fiber_tracts_mask = np.isin(annotation_raw, list(group_ids['Fiber tracts group']))
    fiber_tracts_free_mask = np.isin(annotation_raw, list(group_ids['Purkinje layer']),)

    return constrain_density(
        glia_cell_count,
        glia_density,
        cell_density,
        max_density_mask=fiber_tracts_mask,
        zero_density_mask=fiber_tracts_free_mask,
        copy=copy,
    )


def compute_glia_densities(
    region_map: 'RegionMap',
    annotation_raw: NDArray[int],
    glia_cell_count: int,
    cell_density: NDArray[float],
    glia_densities: Dict[str, NDArray[float]],
    glia_proportions: Dict[str, str],
    copy: bool = True,
) -> Dict[str, NDArray[float]]:
    '''
    Compute the overall glia cell density as well as astrocyte, ologidendrocyte and microglia
    densities.

    Each of the output glia cell densities should satisfy the following properties:
        * It is bounded voxel-wise by the specified overall cell density.
        * It sums up to a cell count matching the total cell count times the prescribed glia cell
         type proportion.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
        annotation_raw: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.
        glia_cell_count: overall glia cell count (taken for instance from the scientific
            literature).
        cell_density: float array of shape (W, H, D) with non-negative entries. The overall
            cell density of the mouse brain.
        glia_densities: dict whose keys are glia cell types (astrocytes, oligodendrocytes,
            microglia) and whose values are the unconstrained glia cell densities corresponding to
            these types. Each density array is a float array of shape (W, H, D) with non-negative
            entries. It holds the input (unconstrained) glia density obtained by averaging different
             marker datasets.
        glia_proportions: a dict whose keys are glia cell types and whose values are strings
            encoding glia cell type proportions (float values).
        copy: If True, the input `glia_density` array is copied. Otherwise it is modified in-place
             and returned by the function as the dict value corresponding to 'glia'.

    Returns:
        float array of shape (W, H, D) with non-negative entries.
        The overall glia cell density, respecting the constraints imposed by the `cell_density`
        upper bound, the input cell count, as well as region-specific hints
        (fiber tracts and Purkinje layer).
    '''
    glia_densities = glia_densities.copy()
    glia_densities['glia'] = compensate_cell_overlap(
        glia_densities['glia'], annotation_raw, gaussian_filter_stdv=1.0, copy=copy
    )
    glia_densities['glia'] = compute_glia_density(
        glia_cell_count,
        get_group_ids(region_map),
        annotation_raw,
        glia_densities['glia'],
        cell_density,
    )
    for glia_type in ['astrocyte', 'oligodendrocyte']:
        glia_densities[glia_type] = compensate_cell_overlap(
            glia_densities[glia_type],
            annotation_raw,
            gaussian_filter_stdv=1.0,
            copy=copy,
        )
        glia_densities[glia_type] = constrain_density(
            glia_cell_count * float(glia_proportions[glia_type]),
            glia_densities[glia_type],
            glia_densities['glia'],
            copy=copy,
        )

    # pylint: disable=fixme
    # FIXME(Luc): The microglia density can be negative.
    glia_densities['microglia'] = (
        glia_densities['glia']
        - glia_densities['astrocyte']
        - glia_densities['oligodendrocyte']
    )

    return glia_densities
