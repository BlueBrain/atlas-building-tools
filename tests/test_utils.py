'''test utils'''
from pathlib import Path

TEST_PATH = Path(Path(__file__).parent)

import pytest
import numpy as np
import numpy.testing as npt

from voxcell import RegionMap

import atlas_building_tools.utils as tested


class Test_load_region_map:
    def test_wrong_input(self):
        with pytest.raises(TypeError):
            tested.load_region_map()

        with pytest.raises(TypeError):
            tested.load_region_map([])

    def test_region_map_input(self):
        region_map = RegionMap.load_json(str(Path(TEST_PATH, '1.json')))
        assert isinstance(tested.load_region_map(region_map), RegionMap)

    def test_str_input(self):
        assert isinstance(
            tested.load_region_map(str(Path(TEST_PATH, '1.json'))), RegionMap
        )

    def test_dict_input(self):
        hierachy = {
            'id': 1,
            'acronym': 'root',
            'children': [{'id': 2, 'children': [{'id': 3}, {'id': 4}]}],
        }
        assert isinstance(tested.load_region_map(hierachy), RegionMap)


def test_get_region_mask():
    annotation = np.arange(27).reshape((3, 3, 3))
    expected = np.zeros((3, 3, 3), dtype=np.bool)
    expected[0, 0, 2] = True  # label 2, "SSp-m6b"
    expected[1, 0, 0] = True  # label 9, "SSp-tr6a"
    expected[2, 1, 1] = True  # label 22, "SSp-tr6a"

    # RegionMap instantiated from string
    mask = tested.get_region_mask(
        'Isocortex', annotation, str(Path(TEST_PATH, '1.json'))
    )
    npt.assert_array_equal(mask, expected)

    # True RegionMap object
    region_map = RegionMap.load_json(str(Path(TEST_PATH, '1.json')))
    mask = tested.get_region_mask('Isocortex', annotation, region_map)
    npt.assert_array_equal(mask, expected)
