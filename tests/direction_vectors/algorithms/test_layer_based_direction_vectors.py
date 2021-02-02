from tempfile import NamedTemporaryFile
import json
import pytest
import numpy as np
import numpy.testing as npt

from voxcell import RegionMap, VoxelData

from atlas_building_tools.direction_vectors.algorithms import (
    layer_based_direction_vectors as tested,
)


class Test_attributes_to_ids:
    def setup_method(self):
        self.hierarchy_json_dict = {
            'id': 0,
            'acronym': 'root',
            'name': 'root',
            'children': [
                {'id': 16, 'acronym': 'Isocortex', 'name': 'Isocortex'},
                {'id': 22, 'acronym': 'CB', 'name': 'Cerebellum'},
                {
                    'id': 1,
                    'acronym': "TMv",
                    'name': 'Tuberomammillary nucleus, ventral part',
                },
                {
                    'id': 23,
                    'acronym': 'TH',
                    'name': 'Thalamus',
                    'children': [
                        {
                            'id': 13,
                            'acronym': 'VAL',
                            'name': 'Ventral anterior-lateral complex of the thalamus',
                        }
                    ],
                },
            ],
        }

    def test_mixed_attributes(self):
        attributes = [
            ('id', 1),
            ('name', 'Isocortex'),
            ('id', 2),
            ('id', 2),
            ('id', 22),
            ('id', 3),
            ('id', 3),
            ('acronym', 'TH'),
            ('acronym', 'CB'),
        ]
        ids = tested.attributes_to_ids(self.hierarchy_json_dict, attributes)
        npt.assert_array_equal(sorted(ids), [1, 13, 16, 22, 23])
        attributes = [
            ('id', 1),
            ('name', 'Isocortex'),
            ('id', 2),
            ('id', 2),
            ('id', 22),
            ('id', 3),
            ('id', 3),
            ('name', 'Cerebellum'),
        ]
        ids = tested.attributes_to_ids(self.hierarchy_json_dict, attributes)
        npt.assert_array_equal(sorted(ids), [1, 16, 22])

    def test_hierarchy_from_file(self):
        attributes = [
            ('id', 1),
            ('acronym', 'Isocortex'),
            ('id', 2),
            ('id', 2),
            ('id', 22),
            ('id', 3),
            ('id', 3),
            ('acronym', 'VAL'),
            ('acronym', 'CB'),
        ]
        with NamedTemporaryFile(mode='w') as hierarchy_json:
            json.dump(self.hierarchy_json_dict, hierarchy_json)
            hierarchy_json.flush()
            hierarchy_json.seek(0)
            ids = tested.attributes_to_ids(hierarchy_json.name, attributes)
            npt.assert_array_equal(sorted(ids), [1, 13, 16, 22])


def check_direction_vectors(direction_vectors, inside, options=None):
    norm = np.linalg.norm(direction_vectors, axis=3)
    # Regiodesics can produce NaN vectors in the region of interest
    # (in its boundary).
    # We take this into account by specifying a non-strict NaN policy.
    if options is not None and not options.get('strict', True):
        inside = np.logical_and(~np.isnan(norm), inside)
    assert np.all(~np.isnan(direction_vectors[inside, :]))
    half = inside.shape[2] // 2
    bottom_hemisphere = np.copy(inside)
    bottom_hemisphere[:, :, half:] = False

    if options is None or options['opposite'] == 'target':
        # Vectors in the bottom hemisphere flow along the positive z-axis
        assert np.all(direction_vectors[bottom_hemisphere, 2] > 0.0)
    else:
        # Vectors in the top hemisphere flow along the negative z-axis
        assert np.all(direction_vectors[bottom_hemisphere, 2] < 0.0)
    top_hemisphere = np.copy(inside)
    top_hemisphere[:, :, :half] = False

    if options is None or options['opposite'] == 'target':
        # Vectors in the top hemisphere flow along the negative z-axis
        assert np.all(direction_vectors[top_hemisphere, 2] < 0.0)
    else:
        # Vectors in the bottom hemisphere flow along the positive z-axis
        assert np.all(direction_vectors[top_hemisphere, 2] > 0.0)

    # NaNs are expected outside `inside`
    assert np.all(np.isnan(norm[~inside]))
    # Non-NaN direction vectors have unit norm
    npt.assert_array_almost_equal(norm[inside], np.full(inside.shape, 1.0)[inside])


class Test_direction_vectors_for_hemispheres:
    @staticmethod
    def landscape_1():
        source = np.zeros((16, 16, 16), dtype=bool)
        source[3:15, 3:15, 3:7] = True
        inside = np.zeros_like(source)
        inside[3:15, 3:15, 6:10] = True
        target = np.zeros_like(source)
        target[3:15, 3:15, 9:13] = True
        return {'source': source, 'inside': inside, 'target': target}

    @staticmethod
    def landscape_2():
        source = np.zeros((16, 16, 16), dtype=bool)
        source[3:13, 3:13, 3] = True
        source[3:13, 3:13, 12] = True
        inside = np.zeros_like(source)
        inside[3:13, 3:13, 3:13] = True
        target = np.zeros_like(source)
        target[3:13, 3:13, 7:9] = True
        return {'source': source, 'inside': inside, 'target': target}

    @staticmethod
    def landscape_3():
        target = np.zeros((16, 16, 16), dtype=bool)
        target[3:13, 3:13, 3] = True
        target[3:13, 3:13, 12] = True
        inside = np.zeros_like(target)
        inside[3:13, 3:13, 3:13] = True
        source = np.zeros_like(target)
        source[3:13, 3:13, 7:9] = True
        return {'source': source, 'inside': inside, 'target': target}

    def test_invalid_option(self):
        with pytest.raises(ValueError):
            tested.direction_vectors_for_hemispheres(
                self.landscape_1(),
                'simple_blur_gradient',
                {'set_opposite_hemisphere_as': 'invalid'},
            )

    def test_simple_blur_without_hemispheres(self):
        l1 = self.landscape_1()
        direction_vectors = tested.direction_vectors_for_hemispheres(
            l1, 'simple_blur_gradient'
        )
        inside = l1['inside']
        assert np.all(~np.isnan(direction_vectors[inside, :]))
        assert np.all(
            direction_vectors[inside, 2] > 0.0
        )  # vectors flow along the positive z-axis
        norm = np.linalg.norm(direction_vectors, axis=3)
        # NaN are expected outside `inside`
        assert np.all(np.isnan(norm[~inside]))
        # Non-NaN direction vectors have unit norm
        npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    def test_regiodesics_without_hemispheres(self):
        l1 = self.landscape_1()
        direction_vectors = tested.direction_vectors_for_hemispheres(l1, 'regiodesics')
        inside = l1['inside']
        assert np.all(~np.isnan(direction_vectors[inside, :]))
        assert np.all(
            direction_vectors[inside, 2] > 0.0
        )  # vectors flow along the positive z-axis
        norm = np.linalg.norm(direction_vectors, axis=3)
        # NaN are expected outside `inside`
        assert np.all(np.isnan(norm[~inside]))
        # Non-NaN direction vectors have unit norm
        npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    def test_simple_blur_with_hemispheres_no_opposite(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(),
            'simple_blur_gradient',
            {'set_opposite_hemisphere_as': None},
        )
        check_direction_vectors(direction_vectors, self.landscape_2()['inside'])

    def test_regiodesics_with_hemispheres_no_opposite(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(), 'regiodesics', {'set_opposite_hemisphere_as': None},
        )
        check_direction_vectors(direction_vectors, self.landscape_2()['inside'])

    def test_simple_blur_with_opposite_hemisphere_as_target(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(),
            'simple_blur_gradient',
            {'set_opposite_hemisphere_as': 'target'},
        )
        check_direction_vectors(
            direction_vectors, self.landscape_2()['inside'], {'opposite': 'target'}
        )

    def test_regiodesics_with_opposite_hemisphere_as_target(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(), 'regiodesics', {'set_opposite_hemisphere_as': 'target'},
        )
        check_direction_vectors(
            direction_vectors, self.landscape_2()['inside'], {'opposite': 'target'}
        )

    def test_simple_blur_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_3(),
            'simple_blur_gradient',
            {'set_opposite_hemisphere_as': 'source'},
        )
        check_direction_vectors(
            direction_vectors, self.landscape_3()['inside'], {'opposite': 'source'}
        )

    def test_regiodesics_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_3(), 'regiodesics', {'set_opposite_hemisphere_as': 'source'},
        )
        check_direction_vectors(
            direction_vectors, self.landscape_3()['inside'], {'opposite': 'source'}
        )


class Test_compute_direction_vectors:
    @staticmethod
    def fake_hierarchy_json():
        return RegionMap.from_dict(
            {
                'id': 0,
                'children': [
                    {'id': 1},
                    {'id': 2},
                    {'id': 3},
                    {'id': 4},
                    {'id': 5},
                    {'id': 6},
                ],
            }
        )

    @staticmethod
    def voxel_data_1():
        raw = np.zeros((16, 16, 16), dtype=int)
        raw[3:15, 3:15, 3:6] = 1
        raw[3:15, 3:15, 6] = 2
        raw[3:15, 3:15, 7] = 3
        raw[3:15, 3:15, 8] = 4
        raw[3:15, 3:15, 9:12] = 5
        return VoxelData(raw, (1.0, 1.0, 1.0))

    @staticmethod
    def landscape_1():
        return {
            'source': [('id', 1), ('id', 2)],
            'inside': [('id', 1), ('id', 2), ('id', 3), ('id', 4), ('id', 5)],
            'target': [('id', 4), ('id', 5)],
        }

    @staticmethod
    def voxel_data_2():
        raw = np.zeros((16, 16, 16), dtype=int)
        raw[3:13, 3:13, 3] = 1
        raw[3:13, 3:13, 4:7] = 2
        raw[3:13, 3:13, 7] = 3
        raw[3:13, 3:13, 8] = 4
        raw[3:13, 3:13, 9:12] = 5
        raw[3:13, 3:13, 12] = 6
        return VoxelData(raw, (1.0, 1.0, 1.0))

    @staticmethod
    def landscape_2():
        return {
            'source': [('id', 1), ('id', 6)],
            'inside': [
                ('id', 1),
                ('id', 2),
                ('id', 3),
                ('id', 4),
                ('id', 5),
                ('id', 6),
            ],
            'target': [('id', 3), ('id', 4)],
        }

    @staticmethod
    def landscape_3():
        return {
            'source': [('id', 3), ('id', 4)],
            'inside': [
                ('id', 1),
                ('id', 2),
                ('id', 3),
                ('id', 4),
                ('id', 5),
                ('id', 6),
            ],
            'target': [('id', 1), ('id', 6)],
        }

    def test_raises_on_wrong_input(self):
        with pytest.raises(ValueError):
            tested.compute_direction_vectors(
                self.fake_hierarchy_json(), [], self.landscape_1()
            )

        with pytest.raises(ValueError):
            tested.compute_direction_vectors(
                self.fake_hierarchy_json(),
                self.voxel_data_1(),
                self.landscape_1(),
                algorithm='unknown_algorithm',
            )

    def test_simple_blur_without_hemispheres(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(), self.voxel_data_1(), self.landscape_1()
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(), self.landscape_1()['inside']
        )
        inside = np.isin(self.voxel_data_1().raw, ids)
        assert np.all(~np.isnan(direction_vectors[inside, :]))
        assert np.all(
            direction_vectors[inside, 2] > 0.0
        )  # vectors flow along the positive z-axis
        norm = np.linalg.norm(direction_vectors, axis=3)
        # NaNs are expected outside `inside`
        assert np.all(np.isnan(norm[~inside]))
        # Non-NaN direction vectors have unit norm
        npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    def test_simple_blur_without_hemispheres_from_file(self):
        with NamedTemporaryFile(mode='w') as nrrd_file:
            self.voxel_data_1().save_nrrd(nrrd_file.name)
            direction_vectors = tested.compute_direction_vectors(
                self.fake_hierarchy_json(), nrrd_file.name, self.landscape_1()
            )
            ids = tested.attributes_to_ids(
                self.fake_hierarchy_json(), self.landscape_1()['inside'],
            )
            inside = np.isin(self.voxel_data_1().raw, ids)
            assert np.all(~np.isnan(direction_vectors[inside, :]))
            assert np.all(
                direction_vectors[inside, 2] > 0.0
            )  # vectors flow along the positive z-axis
            norm = np.linalg.norm(direction_vectors, axis=3)
            # NaN are expected outside `inside`
            assert np.all(np.isnan(norm[~inside]))
            # Non-NaN Direction vectors have unit norm
            npt.assert_array_almost_equal(
                norm[inside], np.full((16, 16, 16), 1.0)[inside]
            )

    def test_simple_blur_with_hemispheres_no_opposite(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_2(),
            'simple_blur_gradient',
            {'set_opposite_hemisphere_as': None},
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(), self.landscape_2()['inside'],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside)

    def test_simple_blur_with_opposite_hemisphere_as_target(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_2(),
            'simple_blur_gradient',
            {'set_opposite_hemisphere_as': 'target'},
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(), self.landscape_2()['inside'],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside, {'opposite': 'target'})

    def test_simple_blur_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_3(),
            'simple_blur_gradient',
            {'set_opposite_hemisphere_as': 'source'},
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(), self.landscape_3()['inside'],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside, {'opposite': 'source'})

    def test_regiodesics_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_3(),
            'regiodesics',
            {'set_opposite_hemisphere_as': 'source'},
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(), self.landscape_2()['inside'],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside, {'opposite': 'source'})
