'''Unit tests for split_23'''
from pathlib import Path
import json
import pytest

import numpy as np
import numpy.testing as npt


TEST_PATH = Path(Path(__file__).parent.parent)

from voxcell import RegionMap, VoxelData

from atlas_building_tools.exceptions import AtlasBuildingToolsError
import atlas_building_tools.region_splitter.isocortex_layer_23 as tested


def test_get_isocortex_hierarchy():
    with pytest.raises(AtlasBuildingToolsError):
        allen_hierarchy = {
            'root': [
                {
                    'id': 998,
                    'children': [
                        {
                            'id': 0,
                            'acronym': 'grey matter',
                            'children': [{'id': 1, 'acronym': 'End of the world'}],
                        }
                    ],
                }
            ]
        }
        tested.get_isocortex_hierarchy(allen_hierarchy)
    with pytest.raises(AtlasBuildingToolsError):
        allen_hierarchy = {
            'msg': [
                {
                    'id': 998,
                    'children': [
                        {
                            'id': 0,
                            'acronym': 'root',
                            'children': [{'id': 1, 'acronym': 'End of the world'}],
                        }
                    ],
                }
            ]
        }
        tested.get_isocortex_hierarchy(allen_hierarchy)


def test_get_isocortex_hierarchy_exception():
    with open(str(Path(TEST_PATH, '1.json'))) as h_file:
        allen_hierarchy = json.load(h_file)
        isocortex_hierarchy = tested.get_isocortex_hierarchy(allen_hierarchy)
        assert isocortex_hierarchy['acronym'] == 'Isocortex'


def test_create_id_generator():
    region_map = RegionMap.load_json(str(Path(TEST_PATH, '1.json')))
    id_generator = tested.create_id_generator(region_map)

    npt.assert_array_equal(
        (
            next(id_generator),
            next(id_generator),
            next(id_generator),
        ),
        (614454278, 614454279, 614454280),
    )


def test_edit_hierarchy():
    region_map = RegionMap.load_json(str(Path(TEST_PATH, '1.json')))
    isocortex_ids = region_map.find('Isocortex', attr='acronym', with_descendants=True)
    assert not isocortex_ids & (
        region_map.find('@.*3[ab]?$', attr='acronym')
        - region_map.find('@.*2/3$', attr='acronym')
    )
    # As of 2020.04.22, AIBS's Isocortex has 4 region ids in layer 2.
    initial_isocortex_layer_2_ids = isocortex_ids & region_map.find(
        '@.*2$', attr='acronym'
    )
    isocortex_layer_23_ids = isocortex_ids & region_map.find('@.*2/3$', attr='acronym')
    id_generator = tested.create_id_generator(region_map)
    layer_3_new_ids = {
        id_: next(id_generator) for id_ in list(isocortex_layer_23_ids)[:5]
    }
    isocortex_hierarchy = None
    with open(str(Path(TEST_PATH, '1.json'))) as h_file:
        allen_hierarchy = json.load(h_file)
        isocortex_hierarchy = tested.get_isocortex_hierarchy(allen_hierarchy)
    tested.edit_hierarchy(isocortex_hierarchy, layer_3_new_ids)
    modified_region_map = RegionMap.from_dict(isocortex_hierarchy)
    assert not modified_region_map.find('@.*2/3$', attr='acronym')
    isocortex_layer_2_ids = modified_region_map.find('@.*2$', attr='acronym')
    isocortex_layer_3_ids = modified_region_map.find('@.*3$', attr='acronym')
    npt.assert_array_equal(
        list(isocortex_layer_3_ids),
        (614454278, 614454279, 614454280, 614454281, 614454282),
    )
    npt.assert_array_equal(
        sorted(isocortex_layer_23_ids),
        sorted(isocortex_layer_2_ids - initial_isocortex_layer_2_ids),
    )


def test_split_isocortex_layer_23():
    ratio = 2.0 / 5.0
    padding = 1
    x_width = 1
    y_width = 55
    z_width = 55
    layer_3_top = padding + int((1.0 - ratio) * y_width)
    raw = np.zeros(
        (2 * padding + x_width, 2 * padding + y_width, 2 * padding + z_width), dtype=int
    )
    # The actual AIBS ids as of 2020.04.21.
    layer_23_ids = [
        41,
        113,
        163,
        180,
        201,
        211,
        219,
        241,
        251,
        269,
        288,
        296,
        304,
        328,
        346,
        412,
        427,
        430,
        434,
        492,
        556,
        561,
        582,
        600,
        643,
        657,
        667,
        670,
        694,
        755,
        806,
        821,
        838,
        854,
        888,
        905,
        943,
        962,
        965,
        973,
        1053,
        1066,
        1106,
        1127,
        12994,
        182305697,
        312782554,
        312782582,
        312782608,
        312782636,
        480149210,
        480149238,
        480149266,
        480149294,
        480149322,
    ]
    layer_2_ids = [195, 524, 606, 747]
    for i, id_ in enumerate(layer_23_ids):
        raw[
            padding : (padding + x_width), padding : (padding + y_width), padding + i
        ] = id_
    for i, id_ in enumerate(layer_2_ids):
        raw[
            padding : (padding + x_width),
            (layer_3_top + 1) : (padding + y_width),
            padding + i,
        ] = id_

    direction_vectors = np.full(raw.shape + (3,), np.nan)
    direction_vectors[
        padding : (padding + x_width),
        padding : (padding + y_width),
        padding : (padding + z_width),
        :,
    ] = [0.0, 1.0, 0.0]

    isocortex_data = VoxelData(raw, (1.0, 1.0, 1.0))
    allen_hierarchy = None
    with open(str(Path(TEST_PATH, '1.json'))) as h_file:
        allen_hierarchy = json.load(h_file)
    tested.split(allen_hierarchy, isocortex_data, direction_vectors, ratio)
    isocortex_hierarchy = tested.get_isocortex_hierarchy(allen_hierarchy)
    modified_region_map = RegionMap.from_dict(isocortex_hierarchy)
    assert not modified_region_map.find('@.*2/3$', attr='acronym')
    isocortex_layer_2_ids = modified_region_map.find('@.*2$', attr='acronym')
    isocortex_layer_3_ids = modified_region_map.find('@.*3$', attr='acronym')
    npt.assert_array_equal(
        np.unique(raw), sorted({0} | isocortex_layer_2_ids | isocortex_layer_3_ids)
    )
    layer_2_mask = np.isin(raw, list(isocortex_layer_2_ids))
    layer_3_mask = np.isin(raw, list(isocortex_layer_3_ids))
    npt.assert_array_equal(raw > 0, np.logical_or(layer_2_mask, layer_3_mask))

    layer_2_indices = np.where(np.isin(raw, list(isocortex_layer_2_ids)))
    layer_3_indices = np.where(np.isin(raw, list(isocortex_layer_3_ids)))
    assert np.count_nonzero(layer_2_indices[1] > layer_3_top) >= 0.95 * len(
        layer_2_indices[1]
    )
    assert np.count_nonzero(layer_3_indices[1] <= layer_3_top) >= 0.95 * len(
        layer_3_indices[1]
    )
