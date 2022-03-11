"""
Generate a cylindrical subset of an atlas, a.k.a column

A common use of such a column is to get a centrally-located subset of a circuit (as a target) which
can be analyzed as being free from edge-effects. Another is to check whether a phenomenon which
differs between an older (column-shaped) circuit and a new (atlas-based) circuit is due to the size
and shape, by checking if the phenomenon resembles the old version when only a cylindrical subset
of the circuit is simulated.
"""
from typing import Optional

import numpy as np
from atlas_commons.typing import FloatArray  # type: ignore
from voxcell import VoxelData


def _positions(mask):
    return mask.indices_to_positions(np.transpose(np.nonzero(mask.raw)))


def _indices(voxeldata, positions):
    indices = voxeldata.positions_to_indices(positions)
    return tuple(indices[..., ax] for ax in range(indices.shape[-1]))


def get_column(center: FloatArray, direction: FloatArray, radius: float, source_mask: VoxelData):
    """
    Create a mask for a cylindrical subset of `source_mask`.

    Args:
        center: 3D coodinates of the cylinder center
        direction: 3D unit vector of the central axis of the cylinder
        radius: radius of the cylinder
        source_mask: 3D boolean mask of voxels which can be included in
            the cylinder

    Returns:
       mask of a cylindrical subset of `source_mask` under the form of VoxelData
       with underlying array of dtype np.uint8.
    """
    positions = _positions(source_mask)
    column_mask = _get_positions_in_column(center, direction, radius, positions)
    column_positions = positions[column_mask, :]
    mask = np.zeros(source_mask.raw.shape, dtype=np.uint8)
    mask[_indices(source_mask, column_positions)] = 1

    return source_mask.with_data(mask)


def get_central_column(
    direction_vectors: VoxelData,
    region_mask: VoxelData,
    radius: float,
    span_mask: Optional[VoxelData] = None,
) -> "VoxelData":
    """Get a mask of a cylindrical column centered on region.

    The column will be aligned with the value of `direction_vectors` at the centroid of the 3D
    region. If `span_region` is not None, the mask only includes voxels in the column which are
    also in `span_region`. Otherwise, the mask of the region will be used.

    Args:
        direction_vectors: unit vectors representing the principal axis (i.e. the height-axis) of
            cells. The VoxelData object holds a float array of shape (W, H, D, 3) where W, H and D
            are integer dimensions.
        region_mask: boolean mask of the region of interest. This is an array of shape (W, H, D)
            where W, H and D are the same as above.
        radius: radius of the column in um
        span_mask: boolean mask the column should span. This is an array of shape (W, H, D) with W
            , H and D as above. If not provided, assumed to be same as `region_mask`.

    Returns
        VoxelData holding a mask of the column with dtype equal to np.uint8
    """

    if span_mask is not None:
        source_mask = span_mask
    else:
        source_mask = region_mask

    center = np.mean(_positions(region_mask), axis=0)
    direction = direction_vectors.lookup(center)

    return get_column(center, direction, radius, source_mask)


def _get_positions_in_column(
    seed_position: FloatArray,
    unit_vector: FloatArray,
    radius: float,
    positions: FloatArray,
):
    """
    Returns a boolean mask of shape `positions.shape[0]` indicating which positions are inside the
    column defined by the first three arguments.

    Args:
        seed_position: 3D float vector holding the coordinates of a reference point
            located on the column axis.
        unit_vector: 3D unit float vector giving the direction of the column central axis.
        radius: radius in microns (um) of the clyinder-shaped column.
        positions: float array of shape (N, 3) where N is the number of 3D points whose coordinates
            are checked.

    Returns:
        boolean mask of shape (N,) with N = `positions.shape[0]`.

    """

    def dot(arr1, arr2):
        return np.sum(arr1 * arr2, axis=-1)

    differences = seed_position - positions
    parallel_component = dot(differences, unit_vector)[:, np.newaxis] * unit_vector[np.newaxis, :]
    perpendicular_component_size = np.linalg.norm(differences - parallel_component, axis=-1)
    column_positions_mask = perpendicular_component_size < radius

    return column_positions_mask
