"""test app/mtype_densities"""
import json
from pathlib import Path

import numpy.testing as npt
import yaml
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.cell_densities as tested
from tests.densities.test_mtype_densities import (
    DATA_PATH,
    create_excitatory_neuron_density,
    create_expected_cell_densities,
    create_inhibitory_neuron_density,
    create_slicer_data,
)


def get_result(runner):
    return runner.invoke(
        tested.mtype_densities,
        [
            "--annotation-path",
            "annotation.nrrd",
            "--hierarchy-path",
            "hierarchy.json",
            "--metadata-path",
            "metadata.json",
            "--direction-vectors-path",
            "direction_vectors.nrrd",
            "--mtypes-config-path",
            "config.yaml",
            "--output-dir",
            "output_dir",
        ],
    )


def test_mtype_densities():
    runner = CliRunner()
    with runner.isolated_filesystem():
        create_excitatory_neuron_density().save_nrrd("excitatory_neuron_density.nrrd")
        create_inhibitory_neuron_density().save_nrrd("inhibitory_neuron_density.nrrd")
        data = create_slicer_data()
        data["annotation"].save_nrrd("annotation.nrrd")
        data["annotation"].with_data(data["direction_vectors"]).save_nrrd("direction_vectors.nrrd")
        with open("metadata.json", "w") as file_:
            json.dump(data["metadata"], file_)
        with open("hierarchy.json", "w") as file_:
            json.dump(data["hierarchy"], file_)
        with open("config.yaml", "w") as file_:
            config = {
                "mtypeToProfileMapPath": str(DATA_PATH / "meta" / "mapping.tsv"),
                "layerSlicesPath": str(DATA_PATH / "meta" / "layers.tsv"),
                "densityProfilesDirPath": str(DATA_PATH / "mtypes"),
                "excitatoryNeuronDensityPath": "excitatory_neuron_density.nrrd",
                "inhibitoryNeuronDensityPath": "inhibitory_neuron_density.nrrd",
            }
            yaml.dump(config, file_)

        result = get_result(runner)
        assert result.exit_code == 0
        expected_cell_densities = create_expected_cell_densities()
        for mtype, expected_cell_density in expected_cell_densities.items():
            created_cell_density = VoxelData.load_nrrd(
                str(Path("output_dir", f"{mtype}_density.nrrd"))
            ).raw
            npt.assert_array_equal(created_cell_density, expected_cell_density)

        # No input density nrrd files
        with open("config.yaml", "w") as file_:
            config = {
                "mtypeToProfileMapPath": str(DATA_PATH / "meta" / "mapping.tsv"),
                "layerSlicesPath": str(DATA_PATH / "meta" / "layers.tsv"),
                "densityProfilesDirPath": str(DATA_PATH / "mtypes"),
            }
            yaml.dump(config, file_)

        result = get_result(runner)
        assert result.exit_code == 1
        assert "neuron density file" in str(result.exception)
