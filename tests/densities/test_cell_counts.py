"""
Unit tests for cell counts extraction
"""

from pathlib import Path

import atlas_building_tools.densities.cell_counts as tested

DATA_PATH = Path(Path(__file__).parent.parent.parent, "atlas_building_tools", "app", "data")


def test_cell_counts():
    assert tested.cell_counts() == {
        "Cerebellum group": 49170000,
        "Isocortex group": 23378142,
        "Rest": 38531858,
    }


def test_neuron_counts():
    assert tested.neuron_counts() == {
        "Cerebellum group": 42220000,
        "Isocortex group": 10097674,
        "Rest": 19442326,
    }


def test_glia_cell_counts():
    for group, glia_count in tested.glia_cell_counts().items():
        assert glia_count == tested.cell_counts()[group] - tested.neuron_counts()[group]


def test_inhibitory_neuron_counts():
    dataframe = tested.extract_inhibitory_neurons_dataframe(Path(DATA_PATH, "mmc1.xlsx"))
    assert tested.inhibitory_neuron_counts(dataframe) == {
        "Cerebellum group": 1861600.2000000002,
        "Isocortex group": 1587055.65,
        "Rest": 2252085.3000000003,
    }


def test_inhibitory_data():
    dataframe = tested.extract_inhibitory_neurons_dataframe(Path(DATA_PATH, "mmc1.xlsx"))
    assert tested.inhibitory_data(dataframe) == {
        "proportions": {
            "Cerebellum group": 0.044092851729038374,
            "Isocortex group": 0.15717041865285014,
            "Rest": 0.11583414967941594,
        },
        "neuron_count": 5700741.15,
    }
