'''
Unit tests of placement hints utils.
'''
from pathlib import Path
import numpy as np
import numpy.testing as npt
import trimesh
from mock import patch

from tests.mocking_tools import MockxelData, Mocked_get_region_mask

import atlas_building_tools.placement_hints.utils as tested


def test_centroid_outfacing_mesh():
    # The centroid lies outside of the mesh
    vertices = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0.8]]
    faces = [[0, 1, 3], [0, 3, 2], [3, 1, 2], [0, 2, 4], [0, 4, 1], [1, 4, 2]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    expected_vertices = [[0, 0, 1], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
    expected_faces = [[2, 3, 0], [2, 0, 1], [0, 3, 1]]
    result = tested.centroid_outfacing_mesh(mesh)
    npt.assert_array_equal(result.vertices, expected_vertices)
    npt.assert_array_equal(result.faces, expected_faces)

    # The centroid lies inside of the mesh
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 0], [0, 3, 1]], dtype=float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    result = tested.centroid_outfacing_mesh(mesh)
    assert (
        len(
            set(tuple(v) for v in result.vertices).difference(
                set(tuple(v) for v in vertices)
            )
        )
        == 0
    )
    assert len(result.faces) == len(faces)


class Test_layers_volume:
    @staticmethod
    @patch(
        'atlas_building_tools.placement_hints.utils.get_region_mask',
        Mocked_get_region_mask(
            {'Isocortex': np.array([[[1, 1, 1]]]), 'L23': np.array([[[0, 1, 1]]])}
        ),
    )
    def test_layers_volume_specified_layers():
        volume = tested.layers_volume(
            'atlas/brain_regions.nrrd',
            'atlas/hierarchy.json',
            layers=['@.*23$'],
            region='Isocortex',
        )
        np.testing.assert_array_equal(volume, np.array([[[0, 1, 1]]]))

    @staticmethod
    @patch(
        'atlas_building_tools.placement_hints.utils.get_region_mask',
        Mocked_get_region_mask(
            {
                'Isocortex': np.array([[[0, 0, 1, 1, 1, 1]]]),
                'Layer1': np.array([[[0, 0, 1, 1, 0, 0]]]),
                'Layer2': np.array([[[0, 0, 0, 0, 1, 0]]]),
            }
        ),
    )
    def test_layers_volume_gets_layers_in_region():
        expected_volume = np.array([[[0, 0, 1, 1, 2, 0]]])
        volume = tested.layers_volume(
            '', '', region='Isocortex', layers=['@.*{}$'.format(i) for i in range(1, 3)]
        )
        npt.assert_array_equal(volume, expected_volume)

    @staticmethod
    @patch(
        'atlas_building_tools.placement_hints.utils.get_region_mask',
        Mocked_get_region_mask(
            {
                'Isocortex': np.array([[[0, 0, 1, 1, 1, 1]]]),
                'Layer1': np.array([[[1, 1, 0, 0, 0, 0]]]),
            }
        ),
    )
    def test_layers_volume_restrict_to_region():
        expected_volume = np.array([[[0, 0, 0, 0, 0, 0]]])
        volume = tested.layers_volume('', '', region='Isocortex', layers=['@.*1$'])
        npt.assert_array_equal(volume, expected_volume)


def test_save_placement_hints():
    voxel_size = 10  # um
    saved_files_dict = {}
    voxel_data_for_saving = MockxelData(
        saved_files_dict, np.array([[[0]]]), (voxel_size,) * 3
    )

    tested.save_placement_hints(
        np.array([[[[2, 3, 4, 5]]], [[[1, 1, 1, 1]]], [[[-2, -1, 0, 1]]]]),
        'output_directory',
        voxel_data_for_saving,
        layer_names=["astring", 1],
    )

    expected_dict = {
        str(Path('output_directory', '[PH]y.nrrd')): -np.array([[[-2, -1, 0, 1]]])
        * voxel_size,
        str(Path('output_directory', '[PH]astring.nrrd')): np.array(
            [[[[3, 4], [2, 4], [1, 4], [0, 4]]]]
        )
        * voxel_size,
        str(Path('output_directory', '[PH]1.nrrd')): np.array(
            [[[[0, 3], [0, 2], [0, 1], [0, 0]]]]
        )
        * voxel_size,
    }

    for filename, value in expected_dict.items():
        npt.assert_array_equal(saved_files_dict[filename], value)
