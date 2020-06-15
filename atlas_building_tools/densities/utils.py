'''Utility functions for cell density computation.'''

from typing import Dict, Set, TYPE_CHECKING
import numpy as np
import scipy.misc
import scipy.ndimage
from nptyping import NDArray  # type: ignore

from atlas_building_tools.utils import copy_array
from atlas_building_tools.exceptions import AtlasBuildingToolsError

if TYPE_CHECKING:  # pragme: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore

# Updated Neuronal Scaling Rules for the Brains of Glires
# (Rodents/Lagomorphs), Herculano-Houzel 2011
CELL_COUNTS = {
    'Cerebellum group': 49170000,  # Cerebellum and arbor vitae
    'Isocortex group': 23378142,  # Isocortex plus the Entorhinal and Piriform areas
    'Rest': 38531858,
}
TOTAL_CELL_COUNT = sum(CELL_COUNTS.values())
GLIA_RATIO = 0.3539791141519625

CEREBELLUM_GROUP_INH_COUNT = (
    (1843742.7 + 1765397.7) * 0.5 + (49159.5 + 54348) * 0.5 + (4179.9 + 6372.6) * 0.5
)  # Cerebellum (+ Arbor vitae, but no inhibitory cells there)
ISOCORTEX_GROUP_INH_COUNT = (
    (543269.7 + 523462.2 + 14680.8 + 14636.1 + 8739.9 + 9374.1) * 0.5
    + (588314.4 + 601225.2 + 70075.5 + 69720.9 + 55536.6 + 59896.2) * 0.5
    + (280005.9 + 281542.2 + 15315 + 15513.6 + 11374.8 + 11428.2) * 0.5
)
REST_INH_COUNT = (
    (
        (2959233.3 + 2823564.0) * 0.5
        + (2268931.5 + 2446630.5) * 0.5
        + (442264.5 + 460858.5) * 0.5
    )
    - CEREBELLUM_GROUP_INH_COUNT
    - ISOCORTEX_GROUP_INH_COUNT
)

INHIBITORY_CELL_COUNT = sum(
    [CEREBELLUM_GROUP_INH_COUNT, ISOCORTEX_GROUP_INH_COUNT, REST_INH_COUNT]
)
UNIFORM_INHIBITORY_RATIO = 0.07944176630434784
INHIBITORY_RATIOS = {
    'Cerebellum group': CEREBELLUM_GROUP_INH_COUNT / 42220000.0,
    'Isocortex group': ISOCORTEX_GROUP_INH_COUNT / 13690000.0,
    'Rest': REST_INH_COUNT / (71760000.0 - 42220000.0 - 13690000.0),
}  # combined with neuron numbers from Herculano-Houzel
INHIBITORY_DATA = {'ratios': INHIBITORY_RATIOS, 'cell_count': INHIBITORY_CELL_COUNT}


def normalize_intensity(
    marker_intensity: NDArray[float],
    annotation_raw: NDArray[int],
    threshold_scale_factor: float = 3.0,
    region_id: int = 0,
    copy: bool = True,
) -> NDArray[float]:
    '''
    Subtract a positive constant from the marker intensity and constraint intensity in [0, 1].

    This function
        * subtracts a so-called threshold obtained as the average positive intensity of the marker
            within `region_id` times `scale_factor`. Typically, `region_id` is zero, the background
            annotation associated to any out-of-brain voxel.
        * zeroes negative intensity values and divides intensity by its maximum.

    This function is used to filter out cells expressing a genetic marker below a
    certain threshold. For instance, as PV+ cells react greatly to GAD because of their size,
    they should be filtered in.
    The function also clears the remaining expression of supposed non-expressing regions, e.g.,
    the outside of the annotated brain.

    Args:
        marker_intensity: 3D float array holding the marker intensity.
        annotation_raw: 3D integer array of region identifiers.
        threshold_scale_factor: Scale factor for the threshold.
        region_id: (Optional) identifier of the region over which an intensity average is computed.
            The latter constant is then subtracted from the `marker_intensity`.
            Defaults to 0, the background identifier, which identfies voxels lying out of the
            whole brain annotated volume.
        copy: If True, a deepcopy of the input is normalized and returned. Otherwise, the input is
             normalized in place. Defaults to True.

    Returns:
        3D array, the normalized marker intensity array.
    '''

    outside_mean = np.mean(
        marker_intensity[
            np.logical_and((annotation_raw == region_id), (marker_intensity > 0.0))
        ]
    )
    output_intensity = copy_array(marker_intensity, copy=copy)
    output_intensity -= outside_mean * threshold_scale_factor
    output_intensity[output_intensity < 0.0] = 0.0
    output_intensity /= np.max(output_intensity)

    return output_intensity


def compensate_cell_overlap(
    marker_intensity: NDArray[float],
    annotation_raw: NDArray[float],
    copy: bool = True,
    gaussian_filter_stdv: float = 0.0,
) -> NDArray[float]:
    '''
    Transform, in place, the marker intensity I into - A * log(1 - I) to compensate cell overlap.

    This function, referred to as 'transfer function', compensates the possible cell overlap, see
    'Estimating the Volumetric Cell Density' section in 'A Cell Atlas for the Mouse Brain',
     (Eröe et al. 2018) and Appendix 1.2 of the Supplementary Material.
     https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full#supplementary-material

    The transfer function
        f:I -> -log(1.0 - I / (1.0 + epsilon))
    is applied to marker intensity where epsilon is choosen close to zero.

    Before applying the transfer function,
        * optionally, applies a Gaussian filter of standard deviation `gaussian_filter_stdv`,
        * intensity values are zeroed outside the annotated volume,
        * the intensity is divided by its maximum value.

    After applying the transfer function,
        * negative values are zeroed,
        * the intensity is divided by its maximum value.

    Arguments:
        marker: intensity scalar field. 3D float array holding the marker intensity over the
            annotated volume.
        annotation_raw: 3D of region identifiers.
        gaussian_filter_stdv: standard deviation value for the gaussian filter to apply on the
            intensity scalar field before transformation. Defaults to 0.0, i.e., no filtering.
        copy: If True, makes deep copy of the input. Otherwise, transform in place. Defaults
            to True.

    Returns:
        The transformed marker intensity array.
    '''
    marker_intensity = marker_intensity.copy() if copy else marker_intensity
    if gaussian_filter_stdv > 0.0:
        marker_intensity = scipy.ndimage.filters.gaussian_filter(
            marker_intensity, sigma=gaussian_filter_stdv, output=marker_intensity
        )
    marker_intensity[annotation_raw == 0] = 0.0
    marker_intensity /= np.max(marker_intensity)  # pre-normalize
    marker_intensity[marker_intensity < 0.0] = 0.0
    epsilon = 1e-4  # Used as a safety for intensity values too close to 1.0.
    marker_intensity = -np.log(1.0 - marker_intensity / (1.0 + epsilon))

    marker_intensity /= np.max(marker_intensity)  # post-normalize
    return marker_intensity


# pylint: disable=fixme
# TODO: Re-assess and underline each density validation crtierion. Design an actual optimization
# strategy if appropriate.
def optimize_distance_to_line(
    line_direction_vector: NDArray[float],
    upper_bounds: NDArray[float],
    sum_constraint: float,
    threshold: float = 1e-7,
    max_iter: int = 45,
    copy: bool = True,
) -> NDArray[float]:
    '''
    Find inside a box the closest point to a line with prescribed coordinate sum.

    This function aims at solving the following (convex quadratic) optmization problem:

    Given a sum S >= 0.0, a line D in the non-negative orthant of the Euclidean N-dimensional
    space and a box B in this orthant (an N-dimensional vector with non-negative coordinates),
    find, if it exists, the point P in B which is the closest to D and whose coordinate sum is S.

    The point P exists if and only if the coordinate sum of B is not less than S.
    The proposed algorithm is iterative and starts with the end point of a direction vector of D.
    First, we uniformly rescale the point coordinates so that it belongs to the plane
     defined by the equation 'coordinate sum = S'. Second, we project the new point on the box
    boundary. The algorithm iterates the two previous steps until we get sufficiently close to B.

    Note: The convergence to the optimal solution is obvious in 2D, but can already fail in 3D.
    At the moment, the funcion only returns a feasible point.

    Args:
        line_direction_vector: N-dimensional float vector with non-negative coordinates.
        upper_bounds: N-dimensional float vector with non-negative coordinates. Defines the
            the box constraining the optimization process.
        sum_constraints: non-negative float number. The coordinate sum constraints imposed on
            the point P we are looking for.
        threshold: non-negative float value. If the coordinate sum of the current point
            is below `threshold`, the function returns the current point.
        max_iter: maximum number of iteration.
        copy: If True, the function makes a copy of the input `line_direction_vector`. Otherwise
            `line_direction_vector` is modified in-place and holds the optimal value.

    Returns: N-dimensional float vector with non-negative coordinates. The solution point of the
        optimization problem, if it exists, up to inacurracy due to threshold size or early
        termination of the algorithm. Otherwise a point on the boundary of the box B defined by
        `upper_bounds`.
    '''
    diff = float('inf')
    iter_ = 0
    point = line_direction_vector.copy() if copy else line_direction_vector
    while diff > threshold and iter_ < max_iter:
        point *= sum_constraint / np.sum(point)
        point = np.min([point, upper_bounds], axis=0)
        diff = np.abs(np.sum(point) - sum_constraint)
        iter_ += 1

    return point


def constrain_density(
    target_sum: float,
    density: NDArray[float],
    density_upper_bound: NDArray[float],
    max_density_mask: NDArray[bool] = None,
    zero_density_mask: NDArray[bool] = None,
    epsilon: float = 1e-3,
    copy: bool = True,
):
    '''
    Modify the input density, while respecting bound constraints, so that it sums to `target_sum`.

    The output density is kept as close as possible to the input in the following sense:
    the algorithm minimizes the distance of the line defined by the input vector under
    the upper bounds and sum constraints.

    Each voxel value of the output is bounded from above by the corresponding value of
    `maximum_density`.

    Additional constraints can be imposed:
        * the voxels in the optional `max_density_mask` are assigned their maximum values.
        * the voxels in the optional `zero_density_mask` are assigned the zero value.

    Args:
        target_sum: the value constraining the sum of all voxel values.
        density: float array of shape (W, H, D) with non-negative values. The array to modify.
        density_upper_bound: float array of shape (W, H, D) with non-negative values.
            The bounds imposed upon the voxel values of the ouput array.
        max_density_mask: Optional boolean array of shape (W, H, D) indicating which voxels should
            be assigned
            their maximum values.
        zero_density_mask: Optional boolean array of shape (W, H, D) indicating which voxels should
            be assigned the zero value.
        epsilon: tolerated error between the sum of the output and `target_sum`.

    Returns:
        float array of shape (W, H, D) with non-negative values.
        The output array values should not exceed those of `density_upper_bounds`.
        The sum of array values should be `epsilon`-close to `target_sum`.

    Raises:
       AtlasBuildingError if the problem is not feasible, i.e,
       if the target sum is greater than the sum of the voxels of `maximum_density` or if
       the largest possible contribution of voxels with non-zero density is less than `target_sum`.
    '''

    if target_sum < epsilon:
        if not copy:
            density[...] = 0.0
        else:
            density = np.zeros_like(density)

        return density

    max_subsum = 0
    if max_density_mask is not None:
        max_subsum = np.sum(density_upper_bound[max_density_mask])

    if target_sum < max_subsum - epsilon:
        raise AtlasBuildingToolsError(
            'The contribution of voxels with prescribed maximum density'
            ' exceeds the density upper bound. One of the two constraints cannot be fulfilled.'
        )

    zero_indices_subsum = 0
    if zero_density_mask is not None:
        zero_indices_subsum = np.sum(density_upper_bound[zero_density_mask])

    if np.sum(density_upper_bound) - zero_indices_subsum < target_sum - epsilon:
        raise AtlasBuildingToolsError(
            'The maximum contribution of voxels with non-zero density'
            ' is less than the target sum. The target sum cannot be reached.'
        )

    density = density.copy() if copy else density
    complement = None
    if max_density_mask is not None:
        density[max_density_mask] = density_upper_bound[max_density_mask]
        complement = max_density_mask
    if zero_density_mask is not None:
        density[zero_density_mask] = 0.0
        complement = np.logical_or(complement, zero_density_mask)

    if complement is None:
        complement = tuple([slice(0, None)] * 3)
    else:
        complement = ~complement

    line_direction_vector = density[complement]
    upper_bound = density_upper_bound[complement]

    # Find a density field respecting all the constraints and which is as close
    # as possible to the line defined by the input density wrt to Euclidean norm.
    density[complement] = optimize_distance_to_line(
        line_direction_vector, upper_bound, target_sum - max_subsum, copy=copy
    )

    if np.abs(np.sum(density) - target_sum) > epsilon:
        raise AtlasBuildingToolsError('The target sum could not be reached')

    return density


def get_group_ids(region_map: 'RegionMap') -> Dict[str, Set[int]]:
    '''
    Get AIBS structure ids for several region groups of interest.

    The groups below have been selected because specific count information is available
    in the scientific literature.

    Args:
        region_map: object to navigate the mouse brain regions hierarchy
            (instantied from AIBS 1.json).

    Returns:
        A dictionary whose keys are region group names and whose values are
        sets of structure identifiers.
    '''
    cerebellum_group_ids = region_map.find(
        'Cerebellum', attr='name', with_descendants=True
    ) | region_map.find('arbor vitae', attr='name', with_descendants=True)
    isocortex_group_ids = (
        region_map.find('Isocortex', attr='acronym', with_descendants=True)
        | region_map.find('Entorhinal area', attr='name', with_descendants=True)
        | region_map.find('Piriform area', attr='name', with_descendants=True)
    )
    purkinje_layer_ids = region_map.find(
        '@.*Purkinje layer', attr='name', with_descendants=True
    )
    fiber_tracts_ids = (
        region_map.find('fiber tracts', attr='acronym', with_descendants=True)
        | region_map.find('grooves', attr='name', with_descendants=True)
        | region_map.find('ventricular systems', attr='name', with_descendants=True)
        | region_map.find('Basic cell groups and regions', attr='name')
        | region_map.find('Cerebellum', attr='name')
    )
    molecular_layer_ids = region_map.find(
        '@.*molecular layer', attr='name', with_descendants=True
    )
    cerebellar_cortex_ids = region_map.find(
        'Cerebellar cortex', attr='name', with_descendants=True
    )

    return {
        'Cerebellum group': cerebellum_group_ids,
        'Isocortex group': isocortex_group_ids,
        'Fiber tracts group': fiber_tracts_ids,
        'Purkinje layer': purkinje_layer_ids,
        'Molecular layer': molecular_layer_ids,
        'Cerebellar cortex': cerebellar_cortex_ids,
    }


def get_region_masks(
    group_ids: Dict[str, Set[int]], annotation_raw: NDArray[float]
) -> Dict[str, NDArray[bool]]:
    '''
    Get the boolean masks of several region groups of interest.

    The groups below have been selected because specific count information is available
    in the scientific literature.

    Args:
        group_ids: a dictionary whose keys are group names and whose values are
            sets of AIBS structure identifiers.
        annotation_raw: integer array of shape (W, H, D) enclosing the AIBS annotation of
            the whole mouse brain.

    Returns:
        A dictionary whose keys are region group names and whose values are
        the boolean masks of these groups. Each boolean array is of shape (W, H, D) and
        encodes which voxels belong to the corresponding group.
    '''
    region_masks = {}
    region_masks['Cerebellum group'] = np.isin(
        annotation_raw, list(group_ids['Cerebellum group'])
    )
    region_masks['Isocortex group'] = np.isin(
        annotation_raw, list(group_ids['Isocortex group'])
    )
    region_masks['Rest'] = np.isin(
        annotation_raw,
        list({0} | group_ids['Cerebellum group'] | group_ids['Isocortex group']),
        invert=True,
    )

    return region_masks