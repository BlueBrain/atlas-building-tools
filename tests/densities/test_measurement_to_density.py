"""
Unit tests for overall cell density computation
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from mock import patch
from voxcell import RegionMap  # type: ignore

import atlas_building_tools.densities.measurement_to_density as tested

TESTS_PATH = Path(__file__).parent.parent


def get_hierarchy():
    return {
        "id": 8,
        "name": "Basic cell groups and regions",
        "acronym": "grey",
        "parent_structure_id": None,  # would be null in json
        "children": [
            {
                "id": 920,
                "acronym": "CENT",
                "name": "Central lobule",
                "parent_structure_id": 645,
                "children": [
                    {
                        "id": 976,
                        "acronym": "CENT2",
                        "name": "Lobule II",
                        "parent_structure_id": 920,
                        "children": [
                            {
                                "id": 10710,
                                "acronym": "CENT2mo",
                                "name": "Lobule II, molecular layer",
                                "parent_structure_id": 976,
                                "children": [],
                            },
                            {
                                "id": 10709,
                                "acronym": "CENT2pu",
                                "name": "Lobule II, Purkinje layer",
                                "parent_structure_id": 976,
                                "children": [],
                            },
                            {
                                "id": 10708,
                                "acronym": "CENT2gr",
                                "name": "Lobule II, granular layer",
                                "parent_structure_id": 976,
                                "children": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def region_map():
    return RegionMap.from_dict(get_hierarchy())


@pytest.fixture
def annotation():
    return np.array([[[920, 10710, 10710], [10709, 10708, 976], [10708, 10710, 10709]]])


def get_hierarchy_info():
    return pd.DataFrame(
        {
            "brain_region": [
                "Central lobule",
                "Lobule II",
                "Lobule II, granular layer",
                "Lobule II, Purkinje layer",
                "Lobule II, molecular layer",
            ],
            "child_id_set": [
                {920, 976, 10708, 10709, 10710},
                {976, 10708, 10709, 10710},
                {10708},
                {10709},
                {10710},
            ],
        },
        index=[920, 976, 10708, 10709, 10710],
    )


@pytest.fixture
def cell_density():
    return np.array([[[1.0, 1.0 / 3.0, 1.0 / 3.0], [0.5, 0.5, 1.0], [0.5, 1.0 / 3.0, 0.5]]])


@pytest.fixture
def cell_densities():
    densities = np.array([5.0 / 9.0, 4.0 / 8.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    hierarchy_info = get_hierarchy_info()
    return pd.DataFrame(
        {"brain_region": hierarchy_info["brain_region"], "cell density": densities},
        index=hierarchy_info.index,
    )


def test_get_hierarchy_info(region_map):
    pdt.assert_frame_equal(
        get_hierarchy_info(), tested.get_hierarchy_info(region_map, root="Central lobule")
    )


def test_get_parent_region(region_map):
    assert tested.get_parent_region("Lobule II", region_map) == "Central lobule"
    assert tested.get_parent_region("Lobule II, molecular layer", region_map) == "Lobule II"
    assert tested.get_parent_region("Basic cell groups and regions", region_map) is None


@pytest.fixture
def volumes(voxel_volume=2):
    hierarchy_info = get_hierarchy_info()
    volumes = voxel_volume * np.array([9.0, 8.0, 2.0, 2.0, 3.0])
    return pd.DataFrame(
        {"brain_region": hierarchy_info["brain_region"], "volume": volumes},
        index=hierarchy_info.index,
    )


def test_compute_region_volumes(annotation, volumes):
    pdt.assert_frame_equal(
        volumes,  # expected
        tested.compute_region_volumes(
            annotation, voxel_volume=2.0, hierarchy_info=get_hierarchy_info()
        ),
    )


@pytest.fixture
def cell_counts(voxel_volume=2):
    counts = voxel_volume * np.array([5.0, 4.0, 1.0, 1.0, 1.0])
    hierarchy_info = get_hierarchy_info()
    return pd.DataFrame(
        {"brain_region": hierarchy_info["brain_region"], "cell count": counts},
        index=hierarchy_info.index,
    )


def test_compute_cell_counts(annotation, cell_density, cell_counts):
    pdt.assert_frame_equal(
        cell_counts,  # expected
        tested.compute_region_cell_counts(
            annotation, cell_density, voxel_volume=2.0, hierarchy_info=get_hierarchy_info()
        ),
    )


def test_cell_count_to_density(region_map, volumes):
    measurements = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",
                "Lobule II",
                "Lobule II, granular layer",
            ],
            "measurement": [31047, 0.722, 28118.0],
            "standard_deviation": [5312, 0.722, 6753.9],
            "measurement_type": ["cell count", "volume", "cell count"],
            "measurement_unit": ["number of cells", "mm^3", "number of cells"],
            "source_title": ["Article 1", "Article 1", "Article 2"],
        }
    )
    ratio = 3.0 / 8.0  # 3 voxels with label 10710 / 8 voxels in Lobule II
    expected = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",
                "Lobule II",
                "Lobule II, granular layer",
            ],
            "measurement": [31047 / (0.722 * ratio), 0.722, 28118.0 / 4.0],
            "standard_deviation": [5312 / (0.722 * ratio), 0.722, 6753.9 / 4.0],
            "measurement_type": ["cell density", "volume", "cell density"],
            "measurement_unit": ["number of cells per mm^3", "mm^3", "number of cells per mm^3"],
            "source_title": ["Article 1", "Article 1", "Article 2"],
        }
    )
    tested.cell_count_to_density(measurements, volumes, region_map)
    pdt.assert_frame_equal(measurements, expected)


def test_cell_proportion_to_density(cell_densities):
    measurements = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",
                "Lobule II",
            ],
            "measurement": [0.5, 0.2],
            "standard_deviation": [0.25, 0.2],
            "measurement_type": ["cell proportion", "neuron proportion"],
            "measurement_unit": ["None", "None"],
            "source_title": ["Article 1", "Article 2"],
        }
    )
    expected = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",
                "Lobule II",
            ],
            "measurement": [0.5 / 3.0, 0.2],
            "standard_deviation": [0.25 / 3.0, 0.2],
            "measurement_type": ["cell density", "neuron proportion"],
            "measurement_unit": ["number of cells per mm^3", "None"],
            "source_title": ["Article 1", "Article 2"],
        }
    )
    tested.cell_proportion_to_density(measurements, cell_densities, "cell proportion")
    pdt.assert_frame_equal(measurements, expected)


def get_annotated_slices():
    return np.array(
        [
            [[920, 976], [10708, 10709]],
            [[10710, 10710], [10709, 10708]],
            [[976, 10708], [10710, 10709]],
        ]
    )


def test_get_average_voxel_count_per_slice():
    annotation = get_annotated_slices()

    actual = tested.get_average_voxel_count_per_slice({10710}, annotation, thickness=2)
    assert actual == 3.0

    actual = tested.get_average_voxel_count_per_slice(
        {976, 10708, 10709, 10710}, annotation, thickness=2
    )
    assert actual == 7.5


def test_cell_count_per_slice_to_density():
    measurements = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",  # 10710
                "Lobule II",  # 976 plus 10708, 10709 and 10710
            ],
            "measurement": [1.0, 2.0],
            "standard_deviation": [0.5, 0.25],
            "measurement_type": ["cell count per slice", "cell count per slice"],
            "measurement_unit": [
                "number of cells per 50-micrometer-thick slice",
                "number of cells per 50-micrometer-thick slice",
            ],
            "source_title": ["Article 1", "Article 2"],
        }
    )
    voxel_volume = (25 ** 3) * 1e-9
    expected = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",
                "Lobule II",
            ],
            "measurement": [1.0 / (3.0 * voxel_volume), 2.0 / (7.5 * voxel_volume)],
            "standard_deviation": [0.5 / (3.0 * voxel_volume), 0.25 / (7.5 * voxel_volume)],
            "measurement_type": ["cell density", "cell density"],
            "measurement_unit": ["number of cells per mm^3", "number of cells per mm^3"],
            "source_title": ["Article 1", "Article 2"],
        }
    )

    tested.cell_count_per_slice_to_density(
        measurements,
        get_annotated_slices(),
        voxel_dimensions=(25.0, 25.0, 25.0),
        voxel_volume=voxel_volume,
        hierarchy_info=get_hierarchy_info(),
    )
    pdt.assert_frame_equal(measurements, expected)


def get_input_data():
    cell_density = np.array(
        [[[0.1, 0.5], [0.5, 0.2]], [[0.5, 0.5], [0.2, 0.1]], [[0.2, 0.2], [0.5, 0.1]]]
    )
    neuron_density = np.array(
        [[[0.1, 0.3], [0.3, 0.2]], [[0.3, 0.3], [0.2, 0.1]], [[0.2, 0.1], [0.3, 0.1]]]
    )
    measurements = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",  # 10710
                "Lobule II",  # 976 plus 10708, 10709 and 10710
                "Lobule II, granular layer",  # 10708
                "Lobule II",
                "Lobule II, Purkinje layer",  # 10709
                "Lobule II",
                "Central lobule",
                "Lobule II, molecular layer",
                "lobule III, Purkinye Layer",  # not in AIBS
            ],
            "measurement": [1.0, 2.0, 0.5, 0.2, 1.0, 2.0, 3.0, 0.1, 1.0],
            "standard_deviation": [0.5, 0.25, 0.5, 0.2, 1.0, 2.0, 1.0, 0.01, 1.0],
            "measurement_type": [
                "cell count per slice",
                "cell count per slice",
                "cell proportion",
                "neuron proportion",
                "cell count",
                "cell count",
                "volume",
                "cell density",
                "cell density",
            ],
            "measurement_unit": [
                "number of cells per 50-micrometer-thick slice",
                "number of cells per 50-micrometer-thick slice",
                "None",
                "None",
                "None",
                "None",
                "mm^3",
                "number of cells per mm^3",
                "number of cells per mm^3",
            ],
            "source_title": [
                "Article 1",
                "Article 2",
                "Article 1",
                "Article 2",
                "Article 1",
                "Article 2",
                "Article 2",
                "Article 1",
                "Article 1",
            ],
        }
    )
    hierarchy = get_hierarchy()
    data = {
        "annotation": get_annotated_slices(),
        "cell_density": cell_density,
        "neuron_density": neuron_density,
        "voxel_dimensions": (25.0, 25.0, 25.0),
        "voxel_volume": (25 ** 3) * 1e-9,
        "hierarchy": hierarchy,
        "region_map": RegionMap.from_dict(hierarchy),
        "measurements": measurements,
    }

    return data


def get_expected_output():
    voxel_volume = get_input_data()["voxel_volume"]
    return pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",  # 10710
                "Lobule II",  # 976 plus 10708, 10709 and 10710
                "Lobule II, granular layer",  # 10708
                "Lobule II",
                "Lobule II, Purkinje layer",  # 10709
                "Lobule II",
                "Central lobule",
                "Lobule II, molecular layer",
            ],
            "measurement": [
                1.0 / (3.0 * voxel_volume),
                2.0 / (7.5 * voxel_volume),
                0.5 * (0.8 / 3.0),
                0.2 * (2.4 / 11.0),
                1.0 / (3 * voxel_volume),
                2.0 / (3.0 * 11.0 / 12.0),
                3.0,
                0.1,
            ],
            "standard_deviation": [
                0.5 / (3.0 * voxel_volume),
                0.25 / (7.5 * voxel_volume),
                0.5 * (0.8 / 3.0),
                0.2 * (2.4 / 11.0),
                1.0 / (3 * voxel_volume),
                2.0 / (3.0 * 11.0 / 12.0),
                1.0,
                0.01,
            ],
            "measurement_type": [
                "cell density",
                "cell density",
                "cell density",
                "cell density",
                "cell density",
                "cell density",
                "volume",
                "cell density",
            ],
            "measurement_unit": [
                "number of cells per mm^3",
                "number of cells per mm^3",
                "number of cells per mm^3",
                "number of cells per mm^3",
                "number of cells per mm^3",
                "number of cells per mm^3",
                "mm^3",
                "number of cells per mm^3",
            ],
            "source_title": [
                "Article 1",
                "Article 2",
                "Article 1",
                "Article 2",
                "Article 1",
                "Article 2",
                "Article 2",
                "Article 1",
            ],
        }
    )


def test_measurement_to_average_density():
    data = get_input_data()
    expected = get_expected_output()
    actual = tested.measurement_to_average_density(
        data["region_map"],
        data["annotation"],
        data["voxel_dimensions"],
        data["voxel_volume"],
        data["cell_density"],
        data["neuron_density"],
        data["measurements"],
    )
    pdt.assert_frame_equal(actual, expected)


def test_remove_non_density_measurements():
    measurements = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, molecular layer",
                "Lobule II",
                "Lobule II, granular layer",
            ],
            "measurement": [1.0, 2.0, 3.0],
            "standard_deviation": [1.0, 2.0, 3.0],
            "measurement_type": ["cell proportion", "volume", "cell density"],
            "measurement_unit": ["None", "mm^3", "number of cells per mm^3"],
            "source_title": ["Article 1", "Article 2", "Article 1"],
        }
    )
    expected = pd.DataFrame(
        {
            "brain_region": [
                "Lobule II, granular layer",
            ],
            "measurement": [3.0],
            "standard_deviation": [3.0],
            "measurement_type": ["cell density"],
            "measurement_unit": ["number of cells per mm^3"],
            "source_title": ["Article 1"],
        }
    )
    tested.remove_non_density_measurements(measurements)
    pdt.assert_frame_equal(measurements, expected)
