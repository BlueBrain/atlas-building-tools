"""
Utility functions to compute distances to boundaries.
"""
from typing import TYPE_CHECKING, Tuple

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

if TYPE_CHECKING:
    import trimesh  # type: ignore


def memory_efficient_intersection(
    intersector: "trimesh.ray.ray_triangle.RayMeshIntersector",
    ray_origins: NDArray[int],
    ray_directions: NDArray[float],
    chunk_length: int = 100000,
) -> Tuple[NDArray[int], NDArray[int], NDArray[int]]:
    """
    Split the computations of ray intersections using several chunks of a
    specified length.
    It is slower than getting intersections directly but costs less memory.

    Args:
        intersector: Ray-mesh intersector.
        chunk_length: the number of rays to calculate intersections for at a time.
        ray_origins: array of shape (N, 3). Origins to cast rays from.
        ray_directions: array of shape (N, 3). Directions in which to cast rays.

    Returns:
        A tuple (locations, ray_ids, tri_ids).
        locations: array (N, 3): locations of intersections.
        ray_ids: array (N, 1): ids of intersecting rays.
        tri_ids: array (N, 1): ids of mesh triangles intersecting a ray.
    """

    locations: NDArray[int] = np.array([[]], dtype=int)
    locations = np.reshape(locations, (0, 3))
    ray_ids: NDArray[int] = np.array([], dtype=int)
    tri_ids: NDArray[int] = np.array([], dtype=int)
    split_count = ray_origins.shape[0] // chunk_length
    split_count = max(split_count, 1)
    ray_ids_offset = 0
    for raypos, raydir in zip(
        np.array_split(ray_origins, split_count),
        np.array_split(ray_directions, split_count),
    ):
        locs, rays, tris = intersector.intersects_location(raypos, raydir, multiple_hits=False)
        rays = rays + ray_ids_offset
        locations = np.vstack([locations, locs])
        ray_ids = np.hstack([ray_ids, rays])
        tri_ids = np.hstack([tri_ids, tris])
        ray_ids_offset += len(raypos)

    return locations, ray_ids, tri_ids
