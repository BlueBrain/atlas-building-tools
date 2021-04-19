"""
Unit tests for the comptutation of the mouse iscortex placement hints.
"""
import unittest

import numpy as np  # type: ignore

import atlas_building_tools.placement_hints.compute_placement_hints as tested
from tests.placement_hints.mocking_tools import IsocortexMock


class Test_isocortex(unittest.TestCase):
    result = None
    isocortex_mock = None

    @classmethod
    def setUpClass(cls):
        cls.isocortex_mock = IsocortexMock(
            padding=10, layer_thickness=15, x_thickness=35, z_thickness=25
        )
        direction_vectors = np.zeros(cls.isocortex_mock.annotation.raw.shape + (3,), dtype=float)
        direction_vectors[cls.isocortex_mock.annotation.raw > 0] = (0.0, 1.0, 0.0)
        cls.result = tested.compute_placement_hints(
            cls.isocortex_mock.region_map,
            cls.isocortex_mock.annotation,
            "Isocortex",
            ["@.*{}[ab]?$".format(i) for i in range(1, 7)],
            direction_vectors,
            [210.639, 190.2134, 450.6398, 242.554, 670.2, 893.62],
        )

    def test_distances_report(self):
        distances_report = self.result[1]["before interpolation"]["report"]

        # Because of the simplified isocortex model used for the tests,
        # bad voxels due to obtuse intersection angle are numerous.
        # Indeed, most of the voxels lying in the boundary with the isocortex complement
        # will issue rays that won't intersect with the target boundary meshes in the expected way.
        assert (
            distances_report[
                "Proportion of voxels whose rays make an obtuse"
                " angle with the mesh normal at the intersection point"
            ]
            <= 0.15
        )
        del distances_report[
            "Proportion of voxels whose rays make an obtuse"
            " angle with the mesh normal at the intersection point"
        ]

        assert (
            distances_report["Proportion of voxels with at least one distance-related problem"]
            <= 0.2
        )
        del distances_report["Proportion of voxels with at least one distance-related problem"]

        for proportion in distances_report.values():
            assert proportion <= 0.075

    def test_problematic_volume(self):
        problematic_volume = self.result[1]["before interpolation"]["volume"]
        assert np.count_nonzero(problematic_volume) / self.isocortex_mock.volume <= 0.17

        problematic_volume = self.result[1]["after interpolation"]["volume"]
        assert np.count_nonzero(problematic_volume) / self.isocortex_mock.volume <= 0.14
