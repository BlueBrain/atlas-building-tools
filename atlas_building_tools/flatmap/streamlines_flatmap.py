"""
Functions to flatten a laminar brain region by contracting along streamlines.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cgal_pybind
import numpy as np  # type: ignore
import trimesh
from atlas_direction_vectors.vector_field import interpolate

from atlas_building_tools.flatmap.utils import reconstruct_surface_mesh
from atlas_building_tools.utils import compute_boundary

if TYPE_CHECKING:  # pragma: no cover
    from atlas_commons.typing import BoolArray, NDArray  # type: ignore
    from voxcell import VoxelData  # type: ignore

L = logging.getLogger(__name__)


def compute_streamlines_intersections(
    layers: NDArray[np.uint8],
    direction_vectors: VoxelData,
    first_layer: np.uint8,
    second_layer: np.uint8,
):
    """
    Compute intersection points of streamlines with the surface boundary separating two layers.

    This function wraps the python-C++ binding of common/cgal-pybind (BBP) of the same name.

    The algorithm draws for each voxel in `layers` a polygonal line passing through it. This
    polygonal line follows the stream of `direction_vectors`; we call it a streamline.
    The first segment of the streamline L which crosses the boundary surface S between
    `first_layer` and `second_layer` is used to approximate the intersection point between L and S.
    The intersection point is approximated by the middle point of this segment.
    This process maps every voxel in `layers` to the intersection point of its streamline.
    The intersection point is set to Vector3(NAN, NAN, NAN) if no segment of the streamline
    crosses the boundary surface.

    Args:
        layers: VoxelData object holding a uint8_t array of shape (W, H, D) where W, H and D are
            the 3 dimensions of the array. The value of a voxel is the index of the layer it
            belongs to. A value of 0 indicates a void voxel.
        direction_vectors: VoxelData object holding float32 array of shape (W, H, D, 3) where
            (W, H, D) is the shape of `layers`. This is a  3D unit vector field defined over
            `layers` domain used to draw the streamlines.
        first_layer: layer index of the first layer.
        second_layer: layer index of the second layer. The layers with index `first_layer` and
            `second_layer` define a boundary surface to be intersected with the streamlines of
            `direction_vectors`.

    Returns: a float32 numpy array of shape (W, H, D, 3). This array holds for each voxel
        V of `layers` the intersection point of the streamline passing through V with the boundary
        surface between `first_layer` and `second_layer`.

    Raises: AtlasBuildingToolsError if the offset or the voxel_dimensions of `layers` and
        `direction_vectors` are different.
    """

    return cgal_pybind.compute_streamlines_intersections(
        np.asarray(layers, dtype=np.uint8),
        np.asarray(direction_vectors.offset, dtype=np.float32),
        np.asarray(direction_vectors.voxel_dimensions, dtype=np.float32),
        np.asarray(direction_vectors.raw, dtype=np.float32),
        first_layer,
        second_layer,
    )


def find_closest_vertices(
    volume_mask: BoolArray,
    voxel_to_point_map: NDArray[np.float32],
    boundary_mesh: trimesh.Trimesh,
) -> NDArray[int]:
    """
    Compute for each voxel in `volume_mask` the closest vertex in `boundary_mesh`.
    Returns an array mapping voxels to `boundary_mesh` vertex indices.

    Each voxel is assigned a mesh vertex index representing the intersection point of the
    direction vectors streamline passing through this voxel.

    Args:
        volume_mask: VoxelData object holding a 3D annotated volume of shape (W, H, D).
        voxel_to_point_map: a float32 numpy array of shape (W, H, D, 3). This array holds for each
            voxel V of `annotated_volume` the intersection point of the streamline passing
            through V with a boundary surface between two layers.
        boundary_mesh: Trimesh object representing a boundary surface between two layers.

    Returns:
        an integer array of shape (W, H, D) assigning to each voxel of `annotated_volume` a
        vertex index of `boundary_mesh`. Each vertex represents the intersection point of the
        streamline passing through the corresponding voxel with the boundary surface.
    """

    points = voxel_to_point_map[volume_mask]
    _, vertex_indices = trimesh.proximity.ProximityQuery(boundary_mesh).vertex(points)
    # An out-of-bound index equal to len(boundary_mesh.vertices) has been returned for every
    # (NaN, NaN, NaN) 3D point. Setting the vertex index to -1 for each point-less voxel.
    vertex_indices[np.isnan(points[..., 0])] = -1

    voxel_to_vertex_index_map = np.full(volume_mask.shape, -1, dtype=int)
    voxel_to_vertex_index_map[volume_mask] = vertex_indices

    return voxel_to_vertex_index_map


def compute_voxel_to_pixel_map(
    volume_mask: BoolArray,
    voxel_to_vertex_index_map: NDArray[int],
    boundary_mesh: trimesh.Trimesh,
    resolution: int = 500,
):
    """
    Compute a voxel-to-pixel map flattening the 3D `volume_mask`.

    The `voxel_to_vertex_index_map` is used to build a voxel-to-pixel map in the following way:
    - `boundary_mesh` is flattened into the 2D Euclidean space thanks to CGAL's authalic map
    (cgal-pybind binding)
    - `voxel_to_vertex_index_map` induces then a map from the voxels in `volume_mask`
        to the flattened mesh vertices.
    - the final flatmap is obtained by rounding the 2D coordinates of flat mesh vertices.

    Args:
        volume_mask: boolean 3D mask of shape (W, H, D) representing the volume of interest.
            (The integers W, H and D are the dimensions of the volume domain.)
        voxel_to_vertex_index_map: integer array of shape (W, H, D) mapping each voxel to either
            the index of a `boundary_mesh` vertex or to -1. The value -1 means that the voxel lies
            outside `volume_mask` or couldn't be assigned a vertex index successfully.
        boundary_mesh: 3D surface mesh representing the surface boundary of two layers.
            This mesh is to be flattened by CGAL's authalic transformation. Therefore it should
            have the topology pf 2D disk and in particular a boundary circle, i.e., the edges that
            belong to a single triangle form a cyclic graph.
        resolution: (Optional) target width of the output flatmap. The flatmap image will be
            constrained to fit in a 2D rectangle image whose width along the x-axis is
            at most `resolution`. Defaults to 500.

    Returns:
        flatmap, i.e., an integer array of shape (W, H, D, 2) mapping the voxels of `volume_mask`
        to a 2D rectangle whose edge length is `resolution`. The pixels (x, y) of the flatmap image
        lies in the square [0, resolution] x [0, resolution * aspect_ratio].
        The voxels lying outside `volume_mask` are mapped to (-1, -1). The other voxels are mapped
        to pixels with non-negative coordinates. The expected aspect ratio is 1.0, but can be
        different for some pathological edge cases.
    """
    flat_mesh = cgal_pybind.SurfaceMesh()
    flat_mesh.add_vertices([cgal_pybind.Point_3(v[0], v[1], v[2]) for v in boundary_mesh.vertices])
    flat_mesh.add_faces([tuple(f) for f in boundary_mesh.faces])
    vertices = np.array(flat_mesh.authalic()[0])

    # Re-center and re-scale 2D vertex coordinates to sit in [0, 1] x [0, aspect_ratio]
    vertices = vertices - np.min(vertices, axis=0)  # Make (0, 0) the bottom left-hand corner
    vertices = vertices / np.max(vertices[:, 0])  # Rescale along the x-axis

    flatmap = np.full(volume_mask.shape + (2,), -1.0, dtype=float)
    known_values_mask = voxel_to_vertex_index_map != -1
    flatmap[known_values_mask] = vertices[voxel_to_vertex_index_map[known_values_mask]]

    # Some streamlines (< 3% for the rat S1 region) hit the void before the surface boundary.
    # We take care of pixel-less voxels via interpolation.
    unknown_values_mask = (~(known_values_mask)) & volume_mask
    interpolate(
        flatmap,
        unknown_values_mask,
        known_values_mask,
        interpolator="nearest-neighbour",
    )

    return np.round(resolution * flatmap).astype(int)


def compute_flatmap(
    layers: NDArray[np.uint8],
    direction_vectors: VoxelData,
    first_layer: int,
    second_layer: int,
    resolution: int = 500,
) -> NDArray[int]:
    """
    Compute a voxel-to-pixel map flattening the 3D `layers` volume.

    Flatten the `layers` volume while locally minimizing the area distortion of the
    surface defined as the boundary surface S between the layers labeled with `first_layer` and
    `second_layer`. Considering the intersection of `direction_vectors` streamlines with S provides
    us with a voxel-to-(surface point) map. The final flatmap is obtained by applying the so-called
    authalic transformation of the CGAL library to S.

    This function relies on two python-C++ bindings located in common/cgal-pybind, namely:
    - the computation of streamlines intersections (computationally intensive task)
    - the authalic transformation (CGAL C++ implementation, see
        https://doc.cgal.org/latest/Surface_mesh_parameterization/classCGAL_1_1Surface__mesh__
        parameterization_1_1Discrete__authalic__parameterizer__3.html)

    Note: We expect the returned flatmap to roughly preserve areas when restricted to the
    surface S.

    Args:
        layers: uint8 numpy array of shape (W, H, D) where W, H and D are the integer dimensions
            of the underlying brain region domain. Voxels are labeld by 1-based layer indices.
            The index 0 is used to label voxels out of the brain region of interest.
        direction_vectors: VoxelData object holding a float array of shape (W, H, D, 3). This
            3D unit vector field defined over `layers` represents the `layers`'s fibers directions,
            Voxels out of the brain region of interest are assigned the vector
            [np.nan, np.nan, np.nan].
        first_layer: positive integer label of the first layer.
        second_layer: positive integer label of the second layer. The two layers define a boundary
            surface to be intersected with `direction_vectors` streamlines.
        resolution: (Optional) target width of the output flatmap. The flatmap image will be
            constrained to fit in a 2D rectangle image whose width along the x-axis is at most
            `resolution`. Defaults to 500.

    Note: The expected image aspect ratio is 1.0, but it can be different for some pathological
        edge cases.

    """
    L.info("Computing streamlines intersections ...")
    voxel_to_point_map = compute_streamlines_intersections(
        layers, direction_vectors, np.uint8(first_layer), np.uint8(second_layer)
    )
    L.info(
        "Finding the boundary voxels between the layers with index %s and %s",
        first_layer,
        second_layer,
    )
    boundary_mask = compute_boundary(
        layers == np.uint8(first_layer), layers == np.uint8(second_layer)
    )
    L.info("Reconstructing a surface mesh from boundary voxels ...")
    boundary_mesh = (
        reconstruct_surface_mesh(boundary_mask, direction_vectors).subdivide().subdivide()
    )
    volume_mask = layers > 0
    L.info("Finding the closest vertices on the reconstructed mesh ...")
    voxel_to_vertex_index_map = find_closest_vertices(
        volume_mask, voxel_to_point_map, boundary_mesh
    )
    L.info("Applying the CGAL authalic map ...")
    return compute_voxel_to_pixel_map(
        volume_mask, voxel_to_vertex_index_map, boundary_mesh, resolution
    )
