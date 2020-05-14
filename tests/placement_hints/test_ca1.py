'''
Unit tests for the comptutation of the placement hints for the CA1 region of the mouse hippocampus.
'''
import unittest
import numpy as np  # type: ignore

from tests.placement_hints.mocking_tools import Ca1Mock
import atlas_building_tools.placement_hints.compute_placement_hints as tested


class Test_ca1(unittest.TestCase):
    result = None
    ca1_mock = None

    @classmethod
    def setUpClass(cls):
        cls.ca1_mock = Ca1Mock(
            padding=10, layer_thickness=30, x_thickness=35, z_thickness=30
        )
        direction_vectors = np.zeros(
            cls.ca1_mock.annotation.raw.shape + (3,), dtype=np.float
        )
        direction_vectors[cls.ca1_mock.annotation.raw > 0] = (0.0, -1.0, 0.0)
        cls.result = tested.compute_placement_hints(
            cls.ca1_mock.region_map,
            cls.ca1_mock.annotation,
            'CA1',
            ['CA1so', 'CA1sp', 'CA1sr', 'CA1slm'],
            direction_vectors,
            flip_direction_vectors=True,
            has_hemispheres=False,
        )

    def test_distances_report(self):
        distances_report = self.result[1]

        # Because of the simplified ca1 model used for the tests,
        # bad voxels due to obtuse intersection angle are numerous.
        # Indeed, most of the voxels lying in the boundary with the ca1 complement
        # will issue rays that won't intersect with the target boundary meshes in the expected way.
        assert (
            distances_report[
                'Proportion of voxels whose rays make an obtuse angle with the mesh normal at the intersection point'
            ]
            <= 0.15
        )
        del distances_report[
            'Proportion of voxels whose rays make an obtuse angle with the mesh normal at the intersection point'
        ]

        assert (
            distances_report[
                'Proportion of voxels with at least one distance-related problem'
            ]
            <= 0.2
        )
        del distances_report[
            'Proportion of voxels with at least one distance-related problem'
        ]

        for proportion in distances_report.values():
            assert proportion <= 0.06

    def test_problematic_volume(self):
        problematic_volume = self.result[1]
        assert np.count_nonzero(problematic_volume) / self.ca1_mock.volume <= 0.05
