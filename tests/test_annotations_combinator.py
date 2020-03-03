'''test annotations_combinator'''
import numpy as np
import numpy.testing as npt

import voxcell

from atlas_building_tools.annotations_combinator import combine_annotations


def test_combine_annotations_fiber_tracts_are_merged():
    hierachy = {
        'id': 1,
        'acronym': 'root',
        'children': [{'id': 2, 'children': [{'id': 3}, {'id': 4}]}],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)
    shape = (2, 2, 2)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[0, :, :] = 3
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.zeros(shape)
    fiber_annotation_ccfv2[1, :, :] = 4
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[:, :, :] = 1
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.zeros(shape)
    expected_raw[0, :, :] = 3
    expected_raw[1, :, :] = 4

    ret = combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )

    npt.assert_array_equal(ret.raw, expected_raw)


def test_combine_annotations_non_leaf_ids_are_replaced():
    hierachy = {
        'id': 1,
        'acronym': 'root',
        'children': [{'id': 2, 'children': [{'id': 3}, {'id': 4}]}],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)
    shape = (2, 2, 2)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[0, :, :] = 3
    brain_annotation_ccfv2[1, :, :] = 4
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.copy(brain_annotation_ccfv2.raw)
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[0, :, :] = 3
    brain_annotation_ccfv3[1, :, :] = 1
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.zeros(shape)
    expected_raw[0, :, :] = 3
    expected_raw[1, :, :] = 4

    ret = combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )
    npt.assert_array_equal(ret.raw, expected_raw)


def test_combine_annotations_zeros_are_ignored():
    hierachy = {'id': 1, 'acronym': 'root', 'children': []}
    region_map = voxcell.RegionMap.from_dict(hierachy)
    shape = (2, 2, 2)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[:, :, :] = 1
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.copy(brain_annotation_ccfv2.raw)
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[:, :, :] = 1
    brain_annotation_ccfv3[0, 0, 0] = 0
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.copy(brain_annotation_ccfv3.raw)

    ret = combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )

    npt.assert_array_equal(ret.raw, expected_raw)


def test_combine_annotations():
    hierachy = {
        'id': 1,
        'acronym': 'root',
        'children': [
            {'id': 2, 'children': [{'id': 3}, {'id': 4}]},
            {
                'id': 5,
                'acronym': 'fibers',
                'children': [
                    {'id': 6, 'children': [{'id': 8}, {'id': 9}]},
                    {'id': 7, 'children': [{'id': 10}, {'id': 11}]},
                ],
            },
            {'id': 12},
        ],
    }
    region_map = voxcell.RegionMap.from_dict(hierachy)

    shape = (3, 3, 3)
    voxel_dimensions = (10.0, 10.0, 10.0)

    brain_annotation_ccfv2 = np.zeros(shape)
    brain_annotation_ccfv2[0:2, 0, :] = 3
    brain_annotation_ccfv2[2, 0, :] = 4
    brain_annotation_ccfv2[:, 2, :] = 12
    brain_annotation_ccfv2 = voxcell.VoxelData(brain_annotation_ccfv2, voxel_dimensions)

    fiber_annotation_ccfv2 = np.zeros(shape)
    fiber_annotation_ccfv2[:, 1, 0] = 8
    fiber_annotation_ccfv2[:, 1, 1] = 9
    fiber_annotation_ccfv2[0:2, 1, 2] = 10
    fiber_annotation_ccfv2[2, 1, 2] = 11
    fiber_annotation_ccfv2 = voxcell.VoxelData(fiber_annotation_ccfv2, voxel_dimensions)

    brain_annotation_ccfv3 = np.zeros(shape)
    brain_annotation_ccfv3[:, 0, :] = 2
    brain_annotation_ccfv3[:, 2, :] = 12
    brain_annotation_ccfv3[:, 1, 0] = 6
    brain_annotation_ccfv3[:, 1, 1] = 5
    brain_annotation_ccfv3 = voxcell.VoxelData(brain_annotation_ccfv3, voxel_dimensions)

    expected_raw = np.copy(brain_annotation_ccfv2.raw)
    expected_raw[:, 1, 0] = 8
    expected_raw[:, 1, 1] = 9
    expected_raw[:, 1, 2] = 0

    ret = combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )

    npt.assert_array_equal(ret.raw, expected_raw)
