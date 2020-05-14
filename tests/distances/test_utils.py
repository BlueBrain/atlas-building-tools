import numpy as np
import numpy.testing as npt
import trimesh

import atlas_building_tools.distances.utils as tested


def test_memory_efficient_interesection():
    cube = trimesh.creation.box(extents=np.array([2.0, 2.0, 2.0]))
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(cube)
    direction_vectors = np.array(
        [
            [0, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [0, 0, -1],
        ]
    )
    origins = np.array(
        [
            [0, -2, 0],
            [0, 2, 0],
            [0, 2, 0.5],
            [-3, 0, 0],
            [3, 0, 0],
            [-3, 0.5, 0],
            [0, 0, -3],
            [0, 0, 3],
            [0.5, 0, 3],
        ]
    )
    expected_locations = np.array(
        [
            [0, -1, 0],
            [0, 1, 0],
            [0, 1, 0.5],
            [-1, 0, 0],
            [1, 0, 0],
            [-1, 0.5, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0.5, 0, 1],
        ]
    )
    chunk_length = 5
    locations, ray_ids, triangle_ids = tested.memory_efficient_intersection(
        intersector, chunk_length, origins, direction_vectors
    )
    npt.assert_array_equal(locations, expected_locations[ray_ids])

    ray_index = list(ray_ids).index(2)
    assert triangle_ids[ray_index] == 7
    ray_index = list(ray_ids).index(5)
    assert triangle_ids[ray_index] == 2
