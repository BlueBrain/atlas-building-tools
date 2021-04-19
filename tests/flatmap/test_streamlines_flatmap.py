"""test for streamlines_flatmap"""

import numpy as np
import numpy.testing as npt
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.flatmap.streamlines_flatmap as tested
from atlas_building_tools.flatmap.utils import reconstruct_surface_mesh


def get_layers():
    # Four rectangular layers of equal width surrounded by a 2-voxel-thick black margin
    layers = np.zeros((20, 3, 3), dtype=np.uint8)
    for i in range(5):
        layers[i * 5 : (i + 1) * 5, ...] = i + 1

    # The final shape is (24 ,7, 7)
    return np.pad(layers, 2, "constant", constant_values=0)


def get_vector_field():
    direction_vectors = np.full((24, 7, 7, 3), np.nan, dtype=np.float32)
    direction_vectors[2:-2, 2:-2, 2:-2] = [1.0, 0.0, 0.0]
    return VoxelData(
        direction_vectors,
        offset=[1.0, 2.0, 3.0],
        voxel_dimensions=np.array([3.0, 1.0, 1.0]),
    )


def get_boundary_mesh():
    boundary = np.zeros((24, 7, 7), dtype=bool)
    boundary[12, :, :] = True
    return reconstruct_surface_mesh(boundary, get_vector_field())


def test_compute_streamlines_intersections():
    voxel_to_point_map = tested.compute_streamlines_intersections(
        get_layers(),
        get_vector_field(),
        first_layer=np.uint8(2),
        second_layer=np.uint8(3),
    )

    npt.assert_allclose(voxel_to_point_map[3, 3, 3], [37, 5, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[4, 3, 3], [37, 5, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[5, 3, 3], [37, 5, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[17, 3, 3], [37, 5, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[20, 3, 3], [37, 5, 6], rtol=1e-2)

    npt.assert_allclose(voxel_to_point_map[2, 4, 3], [37, 6, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[3, 4, 3], [37, 6, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[4, 4, 3], [37, 6, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[13, 4, 3], [37, 6, 6], rtol=1e-2)
    npt.assert_allclose(voxel_to_point_map[15, 4, 3], [37, 6, 6], rtol=1e-2)


def test_find_closest_vertices():
    layers = get_layers()
    voxel_to_point_map = tested.compute_streamlines_intersections(
        layers,
        get_vector_field(),
        first_layer=np.uint8(2),
        second_layer=np.uint8(3),
    )
    boundary_mesh = get_boundary_mesh()
    voxel_to_vertex_index_map = tested.find_closest_vertices(
        layers > 0, voxel_to_point_map, boundary_mesh
    )
    assert voxel_to_vertex_index_map[3, 3, 3] != -1
    assert voxel_to_vertex_index_map[3, 3, 3] == voxel_to_vertex_index_map[4, 3, 3]
    assert voxel_to_vertex_index_map[2, 4, 3] != -1
    assert voxel_to_vertex_index_map[2, 4, 3] == voxel_to_vertex_index_map[15, 4, 3]
    assert voxel_to_vertex_index_map[17, 3, 4] != -1
    assert voxel_to_vertex_index_map[17, 3, 4] == voxel_to_vertex_index_map[3, 3, 4]
    assert voxel_to_vertex_index_map[0, 3, 4] == -1
    assert voxel_to_vertex_index_map[4, 3, 1] == -1
    assert voxel_to_vertex_index_map[14, 1, 4] == -1

    delta = np.linalg.norm(
        voxel_to_point_map[3, 3, 3] - boundary_mesh.vertices[voxel_to_vertex_index_map[3, 3, 3]]
    )
    assert delta <= 0.25
    delta = np.linalg.norm(
        voxel_to_point_map[2, 4, 3] - boundary_mesh.vertices[voxel_to_vertex_index_map[2, 4, 3]]
    )
    assert delta <= 0.25
    delta = np.linalg.norm(
        voxel_to_point_map[17, 3, 4] - boundary_mesh.vertices[voxel_to_vertex_index_map[17, 3, 4]]
    )
    assert delta <= 0.25


def test_compute_voxel_to_pixel_map():
    layers = get_layers()
    voxel_to_point_map = tested.compute_streamlines_intersections(
        layers,
        get_vector_field(),
        first_layer=np.uint8(2),
        second_layer=np.uint8(3),
    )
    boundary_mesh = get_boundary_mesh()
    voxel_to_vertex_index_map = tested.find_closest_vertices(
        layers > 0, voxel_to_point_map, boundary_mesh
    )
    voxel_to_pixel_map = tested.compute_voxel_to_pixel_map(
        layers > 0, voxel_to_vertex_index_map, boundary_mesh, resolution=25
    )

    assert np.all(voxel_to_pixel_map[get_layers() > 0] >= 0)
    assert np.max(voxel_to_pixel_map[..., 0]) <= 25
    npt.assert_allclose(voxel_to_pixel_map[(3, 3, 3)], [12.5, 12.5], rtol=0.15)


def test_compute_flatmap():
    flatmap = tested.compute_flatmap(get_layers(), get_vector_field(), 2, 3, resolution=25)

    assert np.all(flatmap[get_layers() > 0] >= 0)
    assert np.max(flatmap[..., 0]) <= 25
    npt.assert_allclose(flatmap[(3, 3, 3)], [12.5, 12.5], rtol=0.15)
