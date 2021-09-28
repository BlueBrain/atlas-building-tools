"""test app/mtype_densities"""
import json
from pathlib import Path

import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import yaml
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.mtype_densities as tested
from tests.densities.test_mtype_densities_from_map import create_from_probability_map_data
from tests.densities.test_mtype_densities_from_profiles import (
    DATA_PATH,
    create_excitatory_neuron_density,
    create_expected_cell_densities,
    create_inhibitory_neuron_density,
    create_slicer_data,
)


def get_result_from_profiles_(runner):
    return runner.invoke(
        tested.create_from_profile,
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


def test_mtype_densities_from_profiles():
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

        result = get_result_from_profiles_(runner)
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

        result = get_result_from_profiles_(runner)
        assert result.exit_code == 1
        assert "neuron density file" in str(result.exception)


def get_result_from_probablity_map_(runner):
    return runner.invoke(
        tested.create_from_probability_map,
        [
            "--annotation-path",
            "annotation.nrrd",
            "--hierarchy-path",
            "hierarchy.json",
            "--metadata-path",
            "metadata.json",
            "--mtypes-config-path",
            "config.yaml",
            "--output-dir",
            "output_dir",
        ],
    )


class Test_mtype_densities_from_probability_map:
    def setup_method(self, method):
        self.data = create_from_probability_map_data()

    def save_input_data_to_file(self):
        self.data["annotation"].save_nrrd("annotation.nrrd")
        with open("metadata.json", "w") as file_:
            json.dump(self.data["metadata"], file_)
        with open("hierarchy.json", "w") as file_:
            json.dump(self.data["hierarchy"], file_)
        with open("config.yaml", "w") as file_:
            config = {
                "probabilityMapPath": "probability_map.csv",
                "molecularTypeDensityPaths": {
                    "pv": "pv.nrrd",
                    "sst": "sst.nrrd",
                    "vip": "vip.nrrd",
                    "gad67": "gad67.nrrd",
                },
            }
            yaml.dump(config, file_)
        self.data["raw_probability_map"].insert(0, "mtype", self.data["raw_probability_map"].index)
        self.data["raw_probability_map"].to_csv("probability_map.csv", index=False)

        for type_, filepath in config["molecularTypeDensityPaths"].items():
            VoxelData(
                self.data["molecular_type_densities"][type_],
                voxel_dimensions=self.data["annotation"].voxel_dimensions,
            ).save_nrrd(filepath)

    def test_standardize_probability_map(self):
        pdt.assert_frame_equal(
            self.data["probability_map"],
            tested.standardize_probability_map(self.data["raw_probability_map"]),
        )
        probability_map = pd.DataFrame(
            {
                "ChC": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                "NGC-SA": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                "DLAC": [0.0] * 6,
                "SLAC": [0.0] * 6,
            },
            index=[
                "L1 Gad2-IRES-Cre",
                "L6a Htr3a-Cre_NO152",
                "L2/3 Pvalb-IRES-Cre",
                "L4 Htr3a-Cre_NO152",
                "L4 Vip-IRES-Cre",
                "L6b Htr3a-Cre_NO152",
            ],
        )
        actual = tested.standardize_probability_map(probability_map)
        expected = pd.DataFrame(
            {
                "chc": [0.5, 0.0, 0.0, 0.0],
                "ngc_sa": [0.5, 0.0, 0.0, 0.0],
                "lac": [0.0] * 4,
                "sac": [0.0] * 4,
            },
            index=["layer_1_gad67", "layer_6_htr3a", "layer_23_pv", "layer_4_htr3a"],
        )
        pdt.assert_frame_equal(actual, expected)

    def test_output(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.save_input_data_to_file()
            result = get_result_from_probablity_map_(runner)
            assert result.exit_code == 0

            chc = VoxelData.load_nrrd(str(Path("output_dir") / "no_layers" / "CHC_densities.nrrd"))
            assert chc.raw.dtype == float
            npt.assert_array_equal(chc.voxel_dimensions, self.data["annotation"].voxel_dimensions)

    def test_wrong_config(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.save_input_data_to_file()

            # No input density nrrd files
            with open("config.yaml", "w") as file_:
                config = {}
                yaml.dump(config, file_)

            result = get_result_from_probablity_map_(runner)
            assert result.exit_code == 1
            assert "missing" in str(result.exception)
            assert "probabilityMapPath" in str(result.exception)
