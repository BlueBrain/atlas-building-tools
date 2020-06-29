'''Functions to compute inhibitory neuron density.'''

import warnings
from typing import Dict, Optional, TYPE_CHECKING, Tuple, Union
import numpy as np
from nptyping import NDArray  # type: ignore

from atlas_building_tools.exceptions import AtlasBuildingToolsError
from atlas_building_tools.densities.utils import (
    constrain_density,
    compensate_cell_overlap,
    get_group_ids,
    get_region_masks,
)

if TYPE_CHECKING:
    from voxcell import RegionMap  # type: ignore


def _compute_inhibitory_neuron_density(  # pylint: disable=too-many-arguments
    gad1: NDArray[float],
    nrn1: NDArray[float],
    neuron_density: NDArray[float],
    inhibitory_proportion: Optional[float] = None,
    inhibitory_data: Optional[Dict[str, Union[int, Dict[str, float]]]] = None,
    region_masks: Optional[Dict[str, NDArray[bool]]] = None,
) -> Tuple[NDArray[float], int]:
    '''
    Compute a first approximation of the inhibitory neuron density using a prescribed neuron count.

    The input genetic marker datasets GAD1 and NRN1 are used to shape the spatial density
    distribution of the inhibitory and excitatory neurons respectively. Gene marker stained
    intensities are assumed to depend linearly on neuron density (number of neurons per voxel).

    Note regarding these markers:
        Every Gabaergic neuron expresses GAD1 and every GAD1 reacting cell is a gabaergic neuron.
        This genetic marker is indeed responsible for over 90% of the synthesis of GABA. GAD1 is
        only expressed in neurons.

        From "A Cell Atlas for the Mouse Brain" by C. Eroe et al., 2018, in the section
        Neuron Type Differentiation:
        "GAD67 is mainly expressed in inhibitory neurons, and can thus be used to estimate their
        density, while NRN1 is mainly expressed in excitatory neurons (Figure 4A). We normalized
         the GAD67 marker with a sum of both markers, with an overall ratio of 7.94% between
         inhibitory neurons (Kim et al., 2017). We then used the resulting volumetric inhibitory
         marker density."

    Args:
        gad1: float array of shape (W, H, D) with non-negative entries. The GAD1 (a.k.a GAD67)
            marker dataset.
        nrn1: float array of shape (W, H, D) with non-negative entries. The Nrn1 marker dataset.
        neuron_density: float array of shape (W, H, D) with non-negative entries. The overall
            neuron density.
        inhibitory_proportion: (Optional) proportion of inhibitory neurons among all neuron cells.
            If it is not provided, then `inhibitory_data` and `region_masks` must be specified.
        inhibitory_data: (Optional) a dictionary with two keys:
            'ratios': the corresponding value is a dictionary of type Dict[str, float] assigning
            the proportion of inhibitory neurons to each group named by a key string.
            'neuron_count': the total inhibitory neuron count for the whole mouse brain.
        region_masks: (Optional) dictionary whose keys are region group names and whose values are
            the boolean masks of these groups. Each boolean array is of shape (W, H, D) and
            encodes which voxels belong to the corresponding group.

    Returns:
        tuple (innhibitory_neuron_density, inhibitory_neuron_count)
        `innhibitory_neuron_density` is a float array of shape (W, H, D) with non-negative entries.
        `inhibitory_neuron_count` is a self-explanatory integer cell count.

    Raises:
        AtlasBuildingToolsError if both `inhibitory_proportion` and `inhibitory_data` are None.
        or if `inhibitory_proportion` is None, `inhibitory_data` is not None but `region_masks`
        is None.
    '''

    if inhibitory_proportion is None and inhibitory_data is None:
        raise AtlasBuildingToolsError(
            'Either inhibitory_proportion or inhibitory_data should be provided. Both are None.'
        )

    if inhibitory_proportion is not None and inhibitory_data is not None:
        warnings.warn(
            'Both inhibitory_proportion and inhibitory_data are specified.'
            ' Using inhibitory_proportion only.'
        )

    if (
        inhibitory_proportion is None
        and inhibitory_data is not None
        and region_masks is None
    ):
        raise AtlasBuildingToolsError(
            'region_masks must be provided if inhibitory_proportion isn\'t.'
        )

    inhibitory_neuron_density = gad1 / np.mean(gad1)
    excitatory_neuron_density = nrn1 / np.mean(nrn1)

    marker_sum = np.zeros_like(inhibitory_neuron_density)
    if inhibitory_proportion is not None:
        marker_sum = (
            inhibitory_neuron_density * inhibitory_proportion
            + excitatory_neuron_density * (1.0 - inhibitory_proportion)
        )
        inhibitory_neuron_density *= inhibitory_proportion
        inhibitory_neuron_count = int(np.sum(neuron_density) * inhibitory_proportion)
    else:
        for group, mask in region_masks.items():
            inhibitory_neuron_density[mask] = (
                inhibitory_neuron_density[mask] * inhibitory_data['proportions'][group]
            )
            marker_sum[mask] = inhibitory_neuron_density[mask] + excitatory_neuron_density[
                mask
            ] * (1.0 - inhibitory_data['proportions'][group])
        inhibitory_neuron_count = inhibitory_data['neuron_count']

    inhibitory_neuron_density[marker_sum > 0.0] /= marker_sum[marker_sum > 0.0]
    inhibitory_neuron_density /= np.max(inhibitory_neuron_density)

    return inhibitory_neuron_density, inhibitory_neuron_count


def compute_inhibitory_neuron_density(  # pylint: disable=too-many-arguments
    region_map: 'RegionMap',
    annotation_raw: NDArray[float],
    gad1: NDArray[float],
    nrn1: NDArray[float],
    neuron_density: NDArray[float],
    inhibitory_proportion: Optional[float] = None,
    inhibitory_data: Optional[Dict[str, Union[int, Dict[str, float]]]] = None,
) -> NDArray[float]:
    '''
    Compute the inhibitory neuron density using a prescribed neuron count and the overall neuron
     density as an upper bound constraint.

    Further constraints are imposed:
        * voxels of Purkinje layer are assigned the largest possible cell density
        * voxels sitting both in cerebellar cortex and the molecular layer are also assigned
        the largest possible neuron density.

    Args:
        region_map: object to navigate the brain regions hierarchy.
        annotation_raw: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.
        gad1: float array of shape (W, H, D) with non-negative entries. The GAD marker dataset.
        nrn1: float array of shape (W, H, D) with non-negative entries. The Nrn1 marker dataset.
        neuron_density: float array of shape (W, H, D) with non-negative entries. The input
            overall neuron density.
        inhibitory_proportion: (Optional) proportion of inhibitory neurons among all neurons.
            If it is not provided, then `inhibitory_data` and `region_masks` must be specified.
        inhibitory_data: a dictionary with two keys:
            'ratios': the corresponding value is a dictionary of type Dict[str, float] assigning
            the proportion of ihnibitory neurons in each group named by a key string.
            'neuron_count': the total inhibitory neuron count.

    Returns:
        float array of shape (W, H, D) with non-negative entries.
        The overall inhibitory neuron density, respecting the constraints imposed by the
        `neuron_density` upper bound, the input neuron count, as well as region hints
        (Purkinje layer and molecular layer).

    Raises:
        AtlasBuildingToolsError if both `inhibitory_proportion` and `inhibitory_data` are None.
    '''

    region_masks = None
    group_ids = get_group_ids(region_map)
    if inhibitory_proportion is None:
        if inhibitory_data is None:
            raise AtlasBuildingToolsError(
                'Either inhibitory_proportion or inhibitory_data should be provided'
                '. Both are None.'
            )
        region_masks = get_region_masks(group_ids, annotation_raw)

    inhibitory_neuron_density, inhibitory_neuron_count = _compute_inhibitory_neuron_density(
        compensate_cell_overlap(gad1, annotation_raw, gaussian_filter_stdv=1.0),
        compensate_cell_overlap(nrn1, annotation_raw, gaussian_filter_stdv=1.0),
        neuron_density,
        inhibitory_proportion,
        inhibitory_data,
        region_masks,
    )

    inhibitory_neurons_mask = np.isin(annotation_raw, list(group_ids['Purkinje layer']),)
    inhibitory_neurons_mask = np.logical_or(
        inhibitory_neurons_mask,
        np.isin(
            annotation_raw,
            list(group_ids['Cerebellar cortex'] & group_ids['Molecular layer']),
        ),
    )
    return constrain_density(
        inhibitory_neuron_count,
        inhibitory_neuron_density,
        neuron_density,
        inhibitory_neurons_mask,
        copy=False,
    )
