"""test cell_densities"""
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.cell_densities as tested
from atlas_building_tools.densities.cell_counts import (
    extract_inhibitory_neurons_dataframe,
    glia_cell_counts,
    inhibitory_data,
)
from tests.densities.test_excel_reader import (
    check_columns_na,
    check_non_negative_values,
    get_invalid_region_names,
)
from tests.densities.test_glia_densities import get_glia_input_data
from tests.densities.test_inhibitory_neuron_density import get_inhibitory_neuron_input_data

TEST_PATH = Path(Path(__file__).parent.parent)
DATA_PATH = Path(TEST_PATH.parent, "atlas_building_tools", "app", "data")


def _get_cell_density_result(runner):
    args = [
        "cell-density",
        "--hierarchy-path",
        str(Path(TEST_PATH, "1.json")),
        "--annotation-path",
        "annotation.nrrd",
        "--nissl-path",
        "nissl.nrrd",
        "--output-path",
        "overall_cell_density.nrrd",
    ]

    return runner.invoke(tested.app, args)


def test_cell_density():
    input_ = {
        "annotation": np.array(
            [
                [[512, 512, 1143]],
                [[512, 512, 1143]],
                [[477, 56, 485]],
            ]
        ),
        "nissl": np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
    }
    voxel_dimensions = [25] * 3
    runner = CliRunner()
    with runner.isolated_filesystem():
        for name, array in input_.items():
            VoxelData(array, voxel_dimensions=voxel_dimensions).save_nrrd(f"{name}.nrrd")
        result = _get_cell_density_result(runner)

        assert result.exit_code == 0
        voxel_data = VoxelData.load_nrrd("overall_cell_density.nrrd")
        assert voxel_data.raw.dtype == float

        # An error should be raised if annotation and nissl don't use the same voxel dimensions
        VoxelData(np.ones((3, 1, 3)), voxel_dimensions=[10] * 3).save_nrrd("nissl.nrrd")
        result = _get_cell_density_result(runner)
        assert "voxel_dimensions" in str(result.exception)


def _get_glia_cell_densities_result(runner):
    args = [
        "glia-cell-densities",
        "--annotation-path",
        "annotation.nrrd",
        "--hierarchy-path",
        str(Path(TEST_PATH, "1.json")),
        "--cell-density-path",
        "cell_density.nrrd",
        "--glia-density-path",
        "glia_density.nrrd",
        "--astrocyte-density-path",
        "astrocyte_density.nrrd",
        "--oligodendrocyte-density-path",
        "oligodendrocyte_density.nrrd",
        "--microglia-density-path",
        "microglia_density.nrrd",
        "--glia-proportions-path",
        "glia_proportions.json",
        "--output-dir",
        "densities",
    ]

    return runner.invoke(tested.app, args)


def test_glia_cell_densities():
    glia_cell_count = sum(glia_cell_counts().values())
    input_ = get_glia_input_data(glia_cell_count)
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_dimensions = (25, 25, 25)
        VoxelData(input_["annotation"], voxel_dimensions=voxel_dimensions).save_nrrd(
            "annotation.nrrd"
        )
        VoxelData(input_["cell_density"], voxel_dimensions=voxel_dimensions).save_nrrd(
            "cell_density.nrrd"
        )
        for (glia_type, unconstrained_density) in input_["glia_densities"].items():
            VoxelData(unconstrained_density, voxel_dimensions=voxel_dimensions).save_nrrd(
                glia_type + "_density.nrrd"
            )
        with open("glia_proportions.json", "w") as out:
            json.dump(input_["glia_proportions"], out)
        result = _get_glia_cell_densities_result(runner)

        assert result.exit_code == 0

        neuron_density = VoxelData.load_nrrd("densities/neuron_density.nrrd")
        assert neuron_density.raw.dtype == np.float64
        npt.assert_array_equal(neuron_density.shape, input_["annotation"].shape)

        oligodendrocyte_density = VoxelData.load_nrrd("densities/oligodendrocyte_density.nrrd")
        assert oligodendrocyte_density.raw.dtype == np.float64
        npt.assert_array_equal(neuron_density.shape, input_["annotation"].shape)

        # Check that an exception is thrown is voxel dimensions are not consistent
        VoxelData(input_["cell_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "cell_density.nrrd"
        )
        result = _get_glia_cell_densities_result(runner)
        assert "voxel_dimensions" in str(result.exception)


def _get_inh_and_exc_neuron_densities_result(runner):
    args = [
        "inhibitory-and-excitatory-neuron-densities",
        "--annotation-path",
        "annotation.nrrd",
        "--hierarchy-path",
        str(Path(TEST_PATH, "1.json")),
        "--gad1-path",
        "gad1.nrrd",
        "--nrn1-path",
        "nrn1.nrrd",
        "--neuron-density-path",
        "neuron_density.nrrd",
        "--output-dir",
        "densities",
    ]

    return runner.invoke(tested.app, args)


def test_inhibitory_and_excitatory_neuron_densities():
    inhibitory_df = extract_inhibitory_neurons_dataframe(Path(DATA_PATH, "mmc1.xlsx"))
    neuron_count = inhibitory_data(inhibitory_df)["neuron_count"]
    input_ = get_inhibitory_neuron_input_data(neuron_count)
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_dimensions = (25, 25, 25)
        for name in ["annotation", "neuron_density", "gad1", "nrn1"]:
            VoxelData(input_[name], voxel_dimensions=voxel_dimensions).save_nrrd(name + ".nrrd")

        result = _get_inh_and_exc_neuron_densities_result(runner)
        assert result.exit_code == 0

        inh_neuron_density = VoxelData.load_nrrd("densities/inhibitory_neuron_density.nrrd")
        assert inh_neuron_density.raw.dtype == np.float64
        npt.assert_array_equal(inh_neuron_density.shape, input_["annotation"].shape)

        exc_neuron_density = VoxelData.load_nrrd("densities/excitatory_neuron_density.nrrd")
        assert exc_neuron_density.raw.dtype == np.float64
        npt.assert_array_equal(exc_neuron_density.shape, input_["annotation"].shape)

        # Check that an exception is thrown is voxel dimensions are not consistent
        VoxelData(input_["neuron_density"], voxel_dimensions=(10, 10, 10)).save_nrrd(
            "neuron_density.nrrd"
        )
        result = _get_inh_and_exc_neuron_densities_result(runner)
        assert "voxel_dimensions" in str(result.exception)


def _get_compile_measurements_result(runner):
    args = [
        "compile-measurements",
        "--measurements-output-path",
        "measurements.csv",
        "--homogenous-regions-output-path",
        "homogenous_regions.csv",
    ]

    return runner.invoke(tested.app, args)


def test_compile_measurements():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = _get_compile_measurements_result(runner)
        assert result.exit_code == 0

        dataframe = pd.read_csv("measurements.csv")

        assert get_invalid_region_names(dataframe) == {
            "Nucleus of reunions",
            "Nucleus accumbens, core",
            "Nucleus accumbens, shell",
            "Kolliker-Fuse subnucleus",
            "Periaqueductal gray, dorsal lateral",
            "Periaqueductal gray, dorsal medial",
            "Periaqueductal gray, lateral",
            "Periaqueductal gray, ventral lateral",
            "Somatosensory cortex",
            "Superior Colliculus",
            "Prelimbic area, layer 4",
            "Medulla, unassigned",
            "Midbrain, motor related, other",
        }

        check_columns_na(dataframe)
        check_non_negative_values(dataframe)

        dataframe = pd.read_csv("homogenous_regions.csv")
        assert set(dataframe["cell type"]) == {"inhibitory", "excitatory"}
