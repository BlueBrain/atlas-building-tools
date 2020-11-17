'''
Unit tests for inhibitory cell density computation
'''
import warnings
import numpy as np
from pathlib import Path
import numpy.testing as npt
from mock import patch
import pytest

from voxcell import RegionMap

from atlas_building_tools.exceptions import AtlasBuildingToolsError
import atlas_building_tools.densities.inhibitory_neuron_density as tested

TESTS_PATH = Path(__file__).parent.parent


def test_compute_inhibitory_neuron_density_internal():
    gad = np.array([[[0.0, 0.0, 0.5, 0.5]]])
    nrn1 = np.array([[[0.0, 0.0, 0.4, 0.6]]])
    neuron_density = np.array([[[4.0, 4.0, 2.0]]])
    expected = (np.array([[[0.0, 0.0, 1.0, 23.0 / 27.0]]]), 6)
    actual = tested._compute_inhibitory_neuron_density(gad, nrn1, neuron_density, 0.6)
    npt.assert_almost_equal(actual[0], expected[0])
    npt.assert_almost_equal(actual[1], expected[1])

    inhibitory_data = {
        'proportions': {'Cerebellum group': 0.2, 'Isocortex group': 0.2, 'Rest': 0.6,},
        'neuron_count': 6,
    }
    gad = np.array([[[0.0, 0.0, 0.5, 0.25, 0.25]]])
    nrn1 = np.array([[[0.0, 0.0, 0.4, 0.35, 0.25]]])
    region_masks = {
        'Cerebellum group': np.array([[[False, False, True, False, False]]]),
        'Isocortex group': np.array([[[False, False, False, True, False]]]),
        'Rest': np.array([[[False, False, False, False, True]]]),
    }
    actual = tested._compute_inhibitory_neuron_density(
        gad,
        nrn1,
        neuron_density,
        inhibitory_data=inhibitory_data,
        region_masks=region_masks,
    )
    expected = (np.array([[[0.0, 0.0, 25.0 / 63.0, 25.0 / 99.0, 1.0]]]), 6)
    npt.assert_almost_equal(actual[0], expected[0])
    npt.assert_almost_equal(actual[1], expected[1])


def test_compute_inhibitory_neuron_density_exceptions():
    gad = np.array([[[0.0, 0.0, 0.5, 0.25, 0.25]]])
    nrn1 = np.array([[[0.0, 0.0, 0.4, 0.35, 0.25]]])
    neuron_density = np.array([[[0.0, 0.0, 4.0, 4.0, 2.0]]])
    inhibitory_data = {
        'proportions': {'Cerebellum group': 0.2, 'Isocortex group': 0.2, 'Rest': 0.6,},
        'neuron_count': 6,
    }
    with pytest.raises(AtlasBuildingToolsError):
        tested._compute_inhibitory_neuron_density(
            gad, nrn1, neuron_density, inhibitory_data=inhibitory_data
        )

    with pytest.raises(AtlasBuildingToolsError):
        tested._compute_inhibitory_neuron_density(gad, nrn1, neuron_density)

    with warnings.catch_warnings(record=True) as w:
        tested._compute_inhibitory_neuron_density(
            gad,
            nrn1,
            neuron_density,
            inhibitory_data=inhibitory_data,
            inhibitory_proportion=0.079,
        )
        msg = str(w[0].message)
        assert 'Using ' in msg and 'only' in msg


def test_compute_inhibitory_neuron_density():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    gad = np.array([[[0.0, 0.0, 0.5, 0.25, 0.25]]])
    nrn1 = np.array([[[0.0, 0.0, 0.4, 0.35, 0.25]]])
    neuron_density = np.array([[[0.0, 0.0, 4.0, 4.0, 2.0]]])
    inhibitory_data = {
        'proportions': {'Cerebellum group': 0.2, 'Isocortex group': 0.2, 'Rest': 0.6,},
        'neuron_count': 6,
    }
    region_masks = {
        'Cerebellum group': np.array([[[False, False, True, False, False]]]),
        'Isocortex group': np.array([[[False, False, False, True, False]]]),
        'Rest': np.array([[[False, False, False, False, True]]]),
    }
    group_ids = {
        'Purkinje layer': set({1, 2}),
        'Cerebellar cortex': set({1, 6}),
        'Molecular layer': set({2, 6}),
    }
    annotation_raw = np.array([[[1, 2, 6, 3, 4]]], dtype=np.uint32)

    with patch(
        'atlas_building_tools.densities.inhibitory_neuron_density.get_group_ids',
        return_value=group_ids,
    ):
        with patch(
            'atlas_building_tools.densities.inhibitory_neuron_density.get_region_masks',
            return_value=region_masks,
        ):
            with patch(
                'atlas_building_tools.densities.inhibitory_neuron_density.compensate_cell_overlap',
                side_effect=[gad, nrn1],
            ):
                inhibitory_neuron_density = tested.compute_inhibitory_neuron_density(
                    region_map,
                    annotation_raw,
                    gad,
                    nrn1,
                    neuron_density,
                    inhibitory_data=inhibitory_data,
                )
                expected = np.array([[[0.0, 0.0, 4.0, 25.0 / 62.0, 99.0 / 62.0]]])
                npt.assert_almost_equal(inhibitory_neuron_density, expected)


def test_compute_inhibitory_density_large_input():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    shape = (20, 20, 20)
    annotation_raw = np.arange(8000).reshape(shape)
    neuron_density = np.random.random_sample(annotation_raw.shape).reshape(shape)
    neuron_density = 70000 * neuron_density / np.sum(neuron_density)
    gad = np.random.random_sample(annotation_raw.shape).reshape(shape)
    nrn1 = np.random.random_sample(annotation_raw.shape).reshape(shape)
    inhibitory_data = {
        'proportions': {
            'Cerebellum group': 0.4,
            'Isocortex group': 0.35,
            'Rest': 0.25,
        },
        'neuron_count': 30000,
    }
    inhibitory_neuron_density = tested.compute_inhibitory_neuron_density(
        region_map,
        annotation_raw,
        gad,
        nrn1,
        neuron_density,
        inhibitory_data=inhibitory_data,
    )

    assert np.all(inhibitory_neuron_density <= neuron_density)
    npt.assert_allclose(np.sum(inhibitory_neuron_density), 30000, rtol=1e-3)
    assert inhibitory_neuron_density.dtype == np.float64
