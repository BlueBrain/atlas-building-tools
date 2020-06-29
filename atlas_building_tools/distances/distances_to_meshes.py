'''
Module containing free functions for the computation of
distances to boundary meshes with respect to voxels direction vectors.
'''

import logging
import warnings

from typing import Dict, List, Optional, TYPE_CHECKING, Tuple, Union
from nptyping import NDArray  # type: ignore

import numpy as np  # type: ignore
from scipy.interpolate import NearestNDInterpolator  # type: ignore
import trimesh  # type: ignore


from atlas_building_tools.utils import is_obtuse_angle
from atlas_building_tools.distances.utils import memory_efficient_intersection
from atlas_building_tools.direction_vectors.algorithms.utils import normalized
from atlas_building_tools.exceptions import AtlasBuildingToolsError

if TYPE_CHECKING:  # pragma: no cover
    from scipy.interpolate import (  # pylint: disable=ungrouped-imports
        LinearNDInterpolator,
    )
    from voxcell import VoxelData  # type: ignore


logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
L = logging.getLogger(__name__)


def distances_to_mesh_wrt_dir(
    mesh: 'trimesh.TriMesh',
    origins: NDArray[float],
    directions: NDArray[float],
    backward: bool = False,
) -> Tuple[NDArray[float], NDArray[bool]]:
    '''
    Compute the distances from `origins` to the input mesh along `directions`.

    The computation of distances is based on ray-mesh intersections.
    The function also reports which vectors do not point in roughly the same direction
    as the normal vector of the intersected face.

    Args:
        mesh: mesh to get distances to.
        origins(array(N, 3)): origins of direction vectors
        directions(array(N, 3)): directions of vectors to compute along.
        backward: if True, distances will be checked for negated
                  direction vectors. resulting distances will be negative.
                  The check for vector direction in relation to mesh normals
                  still uses unnegated directions.
                  This option is intended to be used to check the locations
                  of deeper boundaries (e.g. L5, L6 for L4 voxels)

    Returns:
        float array(N, ) array holding the distance of each voxel
                          to the nearest point on `mesh` along voxel's direction vector.
        bool array(N, ) True if the ray (origin, direction) intersects with the input mesh
                         such that its angle with the mesh normal is > pi/2.
                         False otherwise.
    '''
    sign = -1 if backward else 1

    # If available, embree provides a significant speedup
    ray = (
        trimesh.ray.ray_pyembree if trimesh.ray.has_embree else trimesh.ray.ray_triangle
    )

    intersector = ray.RayMeshIntersector(mesh)

    number_of_voxels = directions.shape[0]
    assert origins.shape[0] == number_of_voxels
    locations, ray_ids, triangle_ids = memory_efficient_intersection(
        intersector, origins, directions * sign
    )
    dist = np.full(number_of_voxels, np.nan)
    wrong_side = np.zeros(number_of_voxels, dtype=np.bool)

    if locations.shape[0] > 0:  # Non empty intersections
        dist[ray_ids] = sign * np.linalg.norm(locations - origins[ray_ids], axis=1)
        wrong_side[ray_ids] = is_obtuse_angle(
            directions[ray_ids], mesh.face_normals[triangle_ids]
        )
        # pylint: disable=logging-unsupported-format
        L.info(
            'Proportion of intersecting rays: {:.3%}'.format(
                ray_ids.shape[0] / number_of_voxels
            )
        )
    return dist, wrong_side


def _split_indices_along_layer(
    layers_volume: NDArray[int],
    layer: int,
    valid_direction_vectors_mask: NDArray[bool],
) -> Tuple[List[NDArray[int]], List[NDArray[int]]]:
    '''
    Separate in two groups the voxels in `layers_volume` according to
    their position with respect to `layer`.

    The function outputs two groups under the form of numpy indices:
    the indices of the voxels lying below `layer` (`layer`included) and those of the voxels
     lying above (`layer`excluded).

    Args:
        layers_volume: volume enclosed by the union of all layers.
            Each voxel is labelled by an integer representing a layer.
            The higher is the label, the deeper is the layer.
            The value 0 represents a voxel that lies outside this volume.
        layer: the layer label identifying which layer to use for splitting.
        valid_direction_vectors_mask: 3D boolean mask for the voxels with valid direction vectors.
            Voxels whose direction vectors are invalid, i.e., of the form (NaN, NaN, NaN) are
            skipped.

    Returns:
        (below_indices, above_indices): a pair of lists. Each list has length 3. An item
            in a list is a one-dimensional numpy array holding the
            indices of the coordinate corresponding to the item index.

    '''
    below_indices = np.nonzero(
        np.logical_and(layers_volume >= layer, valid_direction_vectors_mask)
    )
    above_mask = np.logical_and(layers_volume < layer, layers_volume > 0)
    above_indices = np.nonzero(np.logical_and(above_mask, valid_direction_vectors_mask))
    return below_indices, above_indices


# pylint: disable=too-many-arguments
def _compute_distances_to_mesh(
    directions: NDArray[float],
    dists: NDArray[float],
    any_obtuse_intersection: NDArray[bool],
    voxel_indices: List[NDArray[int]],
    mesh: 'trimesh.Trimesh',
    index: int,
    backward: bool = False,
    rollback_distance: int = 4,
) -> None:
    '''
        Compute distances from voxels to `mesh` along direction vectors.

        Computations are based on ray-mesh intersections.
        This funcion fill the `dists` array with the outcome.

        Args:
            directions(array(N, 3)): direction vectors to compute along.
            dists: dists: 3D distances array corresponding to layer `mesh_index` + 1.
                A distances array is float 3D numpy array which holds the distance
                of every voxel in the underlying volume (wrt to its direction vector) to a fixed
                layer mesh.
            any_obtuse_intersection: mask of voxels where the intersection with
                a mesh resulted in an obtuse angle between the face and the direction vector.
            voxel_indices: list of the form [X, Y, Z], where the items are 1D numpy arrays
                of the same length. These are the indices of the voxels for which
                the computation is requested.
            mesh: mesh representing the upper boundary of the layer with index
                `index`. The mesh is usually bigger than the upper boundary alone
                and rays are assumed to hit this upper boundary only.
            index: index of the mesh or its corresponding layer.
            backward: (Optional) If True, the direction vectors are used as is to cast rays.
                Otherwise, direction vectors are negated.
            rollback_distance: (Optional) how far to step back along the directions before
                computing distances. Should be >= the max Hausdorff distance of the meshes from the
                voxelized layers it represents. This offset for the ray origins allows to obtain
                more valid intersections for voxels close to the mesh. The default value 4 was found
                by trials and errors.

    '''
    if len(voxel_indices[0]) == 0:
        return

    # Adjusted ray origin: voxel position  -  an added buffer along direction
    sign = -1 if backward else 1
    origins = (
        np.transpose(voxel_indices) + 0.5 - directions * (sign * rollback_distance)
    )
    L.info(
        'Computing distances for the %s mesh with index %d ...',
        'lower' if backward is False else 'upper',
        index,
    )
    dist, wrong = distances_to_mesh_wrt_dir(
        mesh, origins, directions, backward=backward
    )
    dist -= sign * rollback_distance
    with np.errstate(invalid='ignore'):
        dist[(dist * sign) < 0] = 0

    # Set distances
    dists[voxel_indices] = dist
    any_obtuse_intersection[voxel_indices] += wrong
    return


def distances_from_voxels_to_meshes_wrt_dir(
    layers_volume: NDArray[int],
    layer_meshes: List[trimesh.Trimesh],
    directions: NDArray[float],
) -> Tuple[NDArray[float], NDArray[bool]]:
    '''
    For each voxel of the layers volume, compute the distance to each layer mesh along the
    the voxel direction vector.

    Args:
        layers_volume: volume enclosed by the union of all layers.
            Each voxel is labelled by an integer representing a layer.
            The higher is the label, the deeper is the layer.
            The value 0 represents a voxel that lies outside this volume.
        layer_meshes: list of meshes representing the upper boundaries of the layers.
        directions: array of shape (N, 3).
            The direction vectors of the voxels. Should be finite (not nan)
            wherever `layers_volume` > 0.

    Returns:
        Tuple (dists, any_obtuse_intersection).
        dists: 4D numpy array interpreted as a 1D array of 3D distances arrays, one for each layer.
            A distances array is float 3D numpy array which holds the distance
            of every voxel in `layers_volume` (wrt to its direction vector) to a fixed layer mesh.
        any_obtuse_intersection: mask of voxels where the intersection with
            a mesh resulted in an obtuse angle between the face and the direction vector.
    '''
    directions = normalized(directions)

    # dists is a list of 3D numpy arrays, one for each layer
    dists = np.full((len(layer_meshes),) + layers_volume.shape, np.nan)
    any_obtuse_intersection = np.zeros(layers_volume.shape, dtype=np.bool)

    invalid_direction_vectors_mask = np.logical_and(
        np.isnan(np.linalg.norm(directions, axis=-1)), (layers_volume > 0)
    )
    if np.any(invalid_direction_vectors_mask):
        warnings.warn(
            'NaN direction vectors assigned to {:.5%} of the voxels.'
            ' Consider interpolating invalid vectors beforehand.'.format(
                np.mean(invalid_direction_vectors_mask[layers_volume > 0])
            ),
            UserWarning,
        )
    valid_mask = ~invalid_direction_vectors_mask
    L.info('Computing distances for each of the %d meshes', len(layer_meshes))
    for mesh_index, mesh in enumerate(layer_meshes):
        below_indices, above_indices = _split_indices_along_layer(
            layers_volume, mesh_index + 1, valid_mask
        )
        for part, backward in [(below_indices, False), (above_indices, True)]:
            _compute_distances_to_mesh(
                directions[part],
                dists[mesh_index],
                any_obtuse_intersection,
                part,
                mesh,
                mesh_index,
                backward=backward,
            )

    return dists, any_obtuse_intersection


def fix_disordered_distances(distances: NDArray[float]) -> None:
    '''
    Meshes close to one another may intersect one another, leading to distances which do not match
    the layer order.
    Boundaries must be in the correct order for thicknesses to be computed.
    In these problematic cases, both distances must be set to the same value
    (an average of the two).

    The function mutates distances in-place.

    Args:
        distances:
            4D numpy array interpreted as a 1D array of 3D distances arrays, one for each layer.
            A distances array is a float 3D numpy array which holds the distance of every voxel
            (wrt to its direction vector) to a fixed layer mesh.
    '''
    for layer, distance in enumerate(distances[1:], 1):
        with np.errstate(invalid='ignore'):
            previous_is_deeper = distances[layer - 1] < distance
            means = np.mean(
                [
                    distances[layer - 1][previous_is_deeper],
                    distance[previous_is_deeper],
                ],
                axis=0,
            )
            distances[layer - 1, previous_is_deeper] = means
            distances[layer, previous_is_deeper] = means


def report_problems(
    distances: NDArray[float],
    obtuse_intersection: NDArray[bool],
    voxel_data: 'VoxelData',
    max_thicknesses: Optional[NDArray[float]] = None,
) -> Tuple[Dict[str, float], NDArray[bool]]:
    '''
    Reports the proportions of voxels subject to some distance-related problem.

    These problems are:
        * the ray issued by a voxel intersects at an obtuse angle
          with the normal to the boundary mesh.
        * no ray intersection with bottom boundary.
        * no ray intersecton with top boundary.
        * some layer thickness is over the double of the expected amount.

    Args:
        distances(numpy.ndarray): the distances of each voxel to each boundary,
            array of shape (number of boundaries, length, width, height).
        obtuse_intersection(numpy.ndarray): mask of the voxels issuing
            a ray that intersects with some boundary making an obtuse angle
            with the boundary normal vector.
        region_voxel_data(VoxelData): mask of the region to be checked,
            provided as a VoxelData object.
        max_thicknesses: (Optional) 1D float array, the maximum expected thickness for each layer.
            Defaults to None.

     Returns:
        (dict containing the proportions of voxels of each problem,
        mask of all voxels displaying at least one problem)
    '''
    report = {}
    mask = voxel_data.raw > 0
    do_not_intersect_bottom = np.logical_and(np.isnan(distances[-1]), mask)
    do_not_intersect_top = np.logical_and(np.isnan(distances[0]), mask)
    tolerance = voxel_data.voxel_dimensions[0] * 2
    report[
        'Proportion of voxels whose rays do not intersect with the bottom mesh'
    ] = np.mean(do_not_intersect_bottom[mask])
    report[
        'Proportion of voxels whose rays do not intersect with the top mesh'
    ] = np.mean(do_not_intersect_top[mask])
    report[
        'Proportion of voxels whose rays make an obtuse angle '
        'with the mesh normal at the intersection point'
    ] = np.mean(obtuse_intersection[mask])

    # Thickness check
    too_thick = np.full(mask.shape, False)
    if max_thicknesses is not None:
        for i, max_thickness in enumerate(max_thicknesses):
            with np.errstate(invalid='ignore'):
                excess = (distances[i] - distances[i + 1]) > (max_thickness + tolerance)
            too_thick = np.logical_or(too_thick, excess)
        report[
            'Proportion of voxels with a distance gap greater than the maximum thickness'
        ] = np.mean(too_thick[mask])

    problematic_volume = np.full(mask.shape, False)
    for problem in [
        obtuse_intersection,
        do_not_intersect_bottom,
        do_not_intersect_top,
        too_thick,
    ]:
        problematic_volume = np.logical_or(problematic_volume, problem)
    report['Proportion of voxels with at least one distance-related problem'] = np.mean(
        problematic_volume[mask]
    )
    return report, problematic_volume


Interpolator = Union[NearestNDInterpolator, 'LinearNDInterpolator']


def interpolate_volume(
    volume: NDArray[float],
    known_values_mask: NDArray[bool],
    unknown_values_mask: NDArray[bool],
    interpolator: Interpolator = NearestNDInterpolator,
) -> NDArray[float]:
    '''
    Interpolate `unknown_values_mask` based on `known_values_mask` using
    the `interpolator` algorithm.

    Args:
        volume: 3D float array.
        known_values_mask: 3D binary masks of the voxels
         which are assigned a known value.
        unknown_values_mask: 3D binary masks of the voxels
         which are not assigned a value yet.
        interpolator: the scipy interpolation algorithm.

    Returns:
        1D numpy array of interpolated values.

    '''
    nonzero_known_values = np.nonzero(known_values_mask)
    known_positions = np.transpose(nonzero_known_values)
    if len(known_positions) == 0:
        raise AtlasBuildingToolsError(
            'known_values_mask is empty, no values to use for interpolation'
        )
    known_values = np.array(volume)[nonzero_known_values]
    interpolated_values = interpolator(known_positions, known_values)(
        np.transpose(np.nonzero(unknown_values_mask))
    )
    return interpolated_values


def interpolate_nan_voxels(distances: NDArray[float], mask: NDArray[bool]) -> None:
    '''
    Replace in-mask nans with weighted means of nearby in-mask non-nan values
    mutates 'distances' in-place

    Args:
        distances: array of shape (n, x, y, z), that is the volumetric distance data
            where n stands for the number of layers augmented by 1.
        mask: array of shape (x, y, z), that is the voxels to be used
            as known values for interpolation.
            Voxels in this mask with a nan distance will be ignored.

    Returns:
        None (mutates distances in place).
    '''

    for distance in distances:
        nan_distance = np.isnan(distance)
        voxels_to_include = np.logical_and(~nan_distance, mask)
        voxels_to_change = np.logical_and(nan_distance, mask)
        interpolated = interpolate_volume(distance, voxels_to_include, voxels_to_change)
        distance[voxels_to_change] = interpolated
        assert np.all(np.isfinite(distance[np.nonzero(mask)]))
