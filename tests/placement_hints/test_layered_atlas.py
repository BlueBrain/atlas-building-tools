'''
Unit tests for the layered_atlas module
'''
import tempfile
from pathlib import Path
import unittest
from typing import Optional
import numpy.testing as npt  # type: ignore
import numpy as np  # type: ignore

from voxcell import VoxelData  # type: ignore
from atlas_building_tools.placement_hints.layered_atlas import LayeredAtlas, DistanceProblem  # type: ignore

from tests.placement_hints.mocking_tools import IsocortexMock
import atlas_building_tools.placement_hints.layered_atlas as tested


class Test_Layered_Atlas(unittest.TestCase):
    isocortex_mock: Optional[IsocortexMock] = None
    layered_atlas: Optional[LayeredAtlas] = None

    @classmethod
    def setUpClass(cls):
        cls.isocortex_mock = IsocortexMock(
            padding=20, layer_thickness=15, x_thickness=35, z_thickness=25
        )
        cls.layered_atlas = tested.LayeredAtlas(
            'Isocortex',
            cls.isocortex_mock.annotation,
            cls.isocortex_mock.region_map,
            ['@.*{}[ab]?$'.format(i) for i in range(1, 7)],
        )

    def test_region(self):
        raw = self.layered_atlas.region.raw
        expected = self.isocortex_mock.annotation.raw.copy()
        expected[expected == self.isocortex_mock.background] = 0
        expected[expected > 0] = 1
        npt.assert_array_equal(raw, expected.astype(bool))

    def test_create_layers_volume(self):
        volume = self.layered_atlas.volume
        expected = self.isocortex_mock.annotation.raw.copy()
        for i, ids in enumerate(self.isocortex_mock.layer_ids):
            expected[np.isin(expected, ids)] = 6 - i
        expected[expected == 507] = 0
        npt.assert_array_equal(volume, expected)

    def test_create_layer_meshes(self):
        volume = self.layered_atlas.volume
        meshes = self.layered_atlas.create_layer_meshes(volume)
        for i, mesh in enumerate(meshes[:-1]):
            vertices = mesh.vertices
            assert np.all(vertices[:, 1] >= 0.8 * self.isocortex_mock.padding)
            assert np.all(
                vertices[:, 1]
                <= 1.2 * self.isocortex_mock.padding
                + (6 - i) * self.isocortex_mock.thickness
            )

    def test_compute_distances_to_layer_meshes(self):
        direction_vectors = np.zeros(
            self.isocortex_mock.annotation.raw.shape + (3,), dtype=float
        )
        direction_vectors[self.isocortex_mock.annotation.raw > 0] = (0.0, 1.0, 0.0)
        distances = tested.compute_distances_to_layer_meshes(
            'Isocortex',
            self.layered_atlas.annotation,
            self.layered_atlas.region_map,
            direction_vectors,
            self.layered_atlas.layer_regexps,
        )
        atlas = distances['layered_atlas']
        assert atlas.acronym == 'Isocortex'
        assert atlas.layer_regexps == self.layered_atlas.layer_regexps
        assert atlas.annotation == self.layered_atlas.annotation

        dist_info = distances['distances_to_layer_meshes'][:-1]
        for i, ids in enumerate(self.isocortex_mock.layer_ids):
            layer_mask = np.isin(atlas.annotation.raw, ids)
            layer_index = 6 - i
            for j, dist_to_upper_boundary in enumerate(dist_info):
                boundary_index = j + 1
                non_nan_mask = ~np.isnan(dist_to_upper_boundary)
                # No more than 10% of NaNs
                npt.assert_allclose(
                    np.count_nonzero(non_nan_mask), self.isocortex_mock.volume, rtol=0.1
                )
                mask = layer_mask & (~np.isnan(dist_to_upper_boundary))
                valid = np.count_nonzero(
                    dist_to_upper_boundary[mask]
                    <= (layer_index - boundary_index + 1)
                    * self.isocortex_mock.thickness
                )
                # Check that distances are respected for at least 65% of the voxel of each layer
                npt.assert_allclose(valid, np.count_nonzero(mask), rtol=0.35)
                valid = np.count_nonzero(
                    dist_to_upper_boundary[mask]
                    >= (layer_index - boundary_index) * self.isocortex_mock.thickness
                )
                npt.assert_allclose(valid, np.count_nonzero(mask), rtol=0.35)

    def test_save_problematic_voxel_mask(self):
        with tempfile.TemporaryDirectory() as tempdir:
            problematic_voxel_mask = np.zeros((2, 2, 2), dtype=bool)
            problematic_voxel_mask[0, 0, 0] = True
            problematic_voxel_mask[0, 1, 0] = True
            problems = {
                'before interpolation': { 'volume': problematic_voxel_mask.copy()},
                'after interpolation': {'volume': problematic_voxel_mask}
            }
            problematic_voxel_mask[0, 1, 0] = False
            expected_voxel_mask = np.full((2, 2, 2), np.uint8(DistanceProblem.NO_PROBLEM.value))
            expected_voxel_mask[0, 0, 0] = np.uint8(DistanceProblem.AFTER_INTERPOLATION.value)
            expected_voxel_mask[0, 1, 0] = np.uint8(DistanceProblem.BEFORE_INTERPOLATION.value)

            tested.save_problematic_voxel_mask(
                self.layered_atlas, problems, tempdir
            )
            volume_path = Path(tempdir, 'Isocortex_problematic_voxel_mask.nrrd')
            voxel_data = VoxelData.load_nrrd(volume_path)
            npt.assert_almost_equal(voxel_data.raw, expected_voxel_mask)
