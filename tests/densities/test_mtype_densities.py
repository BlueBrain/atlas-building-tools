import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import yaml
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.densities.mtype_densities as tested

TESTS_PATH = Path(__file__).parent.parent
DATA_PATH = TESTS_PATH / "densities" / "data"


def create_placement_hints():
    placement_hints_2 = np.array([[[[3.0, 5.0], [3.0, 5.0], [3.0, 5.0], [3.0, 5.0], [3.0, 5.0]]]])
    placement_hints_3 = np.array([[[[0.0, 3.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0]]]])
    placement_hints_y = np.array([[[0.0, 1.0, 2.0, 3.0, 4.0]]])
    return {
        "layer_2": placement_hints_2,
        "layer_3": placement_hints_3,
        "y": placement_hints_y,
    }


def create_density_profile_collection():
    return tested.DensityProfileCollection.load(
        DATA_PATH / "meta" / "mapping.tsv",
        DATA_PATH / "meta" / "layers.tsv",
        DATA_PATH / "mtypes",
    )


def create_slice_voxel_indices():
    return [([0], [0], [slice_index]) for slice_index in range(5)]


def create_excitatory_neuron_density():
    return VoxelData(np.array([[[3.0, 1.0, 4.0, 2.0, 2.0]]]), voxel_dimensions=[1.0] * 3)


def create_inhibitory_neuron_density():
    return VoxelData(np.array([[[4.0, 3.0, 1.0, 2.0, 1.0]]]), voxel_dimensions=[1.0] * 3)


def create_placement_hints_config(dirpath: str):
    return {
        "layerPlacementHintsPaths": {
            "layer_2": str(Path(dirpath, "[PH]layer_2.nrrd")),
            "layer_3": str(Path(dirpath, "[PH]layer_3.nrrd")),
            "y": str(Path(dirpath, "[PH]y.nrrd")),
        }
    }


def create_input_volumetric_data(dirpath: str):
    create_excitatory_neuron_density().save_nrrd(
        str(Path(dirpath, "excitatory_neuron_density.nrrd"))
    )
    create_inhibitory_neuron_density().save_nrrd(
        str(Path(dirpath, "inhibitory_neuron_density.nrrd"))
    )
    placement_hints = create_placement_hints()
    config = create_placement_hints_config(dirpath)
    for layer, ph_array in placement_hints.items():
        VoxelData(ph_array, voxel_dimensions=[1.0] * 3).save_nrrd(
            config["layerPlacementHintsPaths"][layer]
        )

    with open(Path(dirpath, "placement_hints_config.yaml"), "w") as out:
        yaml.dump(config, out)

    return {
        "excitatory_neuron_density": str(Path(dirpath, "excitatory_neuron_density.nrrd")),
        "inhibitory_neuron_density": str(Path(dirpath, "inhibitory_neuron_density.nrrd")),
        "placement_hints_config": Path(dirpath, "placement_hints_config.yaml"),
    }


def create_expected_cell_densities():
    return {
        "L2_TPC:A": np.array([[[0.0, 0.0, 0.0, 2.0 * 10100.0 / 15800.0, 2.0 * 11000.0 / 17200.0]]]),
        "L23_BP": np.array([[[0.0, 3.0, 1.0, 2.0, 1.0]]]),
        "L3_TPC:B": np.array(
            [
                [
                    [
                        3.0 * 20500.0 / 82000.0,
                        1.0 * 50200.0 / 107600.0,
                        4.0 * 36500.0 / 93000.0,
                        0.0,
                        0.0,
                    ]
                ]
            ]
        ),
    }


def test_density_profile_collection_loading():
    density_profile_collection = None
    with warnings.catch_warnings(record=True) as w:
        density_profile_collection = create_density_profile_collection()

        msg = str(w[0].message)
        assert "No inhibitory cells assigned to slice 0 of layer_3" in msg

    expected_profile_data = {
        "layer_2": {
            "inhibitory": pd.DataFrame({"L23_BP": [1.0, 1.0]}).rename(index={0: 3, 1: 4}),
            "excitatory": pd.DataFrame(
                {
                    "L2_IPC": [5700.0 / 15800.0, 6200.0 / 17200.0],
                    "L2_TPC:A": [10100.0 / 15800.0, 11000.0 / 17200.0],
                }
            ).rename(index={0: 3, 1: 4}),
        },
        "layer_3": {
            "inhibitory": pd.DataFrame({"L23_BP": [0.0, 1.0, 1.0]}),
            "excitatory": pd.DataFrame(
                {
                    "L3_TPC:A": [
                        61500.0 / 82000.0,
                        57400.0 / 107600.0,
                        56500.0 / 93000.0,
                    ],
                    "L3_TPC:B": [
                        20500.0 / 82000.0,
                        50200.0 / 107600.0,
                        36500.0 / 93000.0,
                    ],
                }
            ),
        },
    }

    for layer in ["layer_2", "layer_3"]:
        for synapse_class in ["excitatory", "inhibitory"]:
            pdt.assert_frame_equal(
                density_profile_collection.profile_data[layer][synapse_class],
                expected_profile_data[layer][synapse_class],
            )


def test_load_placement_hints():
    with tempfile.TemporaryDirectory() as tempdir:
        placement_hints = create_placement_hints()
        placement_hints_paths = {
            name: str(Path(tempdir, f"[PH]{name}.nrrd")) for name in placement_hints
        }
        for ph_data, ph_path in zip(placement_hints.values(), placement_hints_paths.values()):
            VoxelData(ph_data, voxel_dimensions=[1.0] * 3).save_nrrd(ph_path)

        loaded = tested.DensityProfileCollection.load_placement_hints(placement_hints_paths)
        for actual, expected in zip(loaded.values(), placement_hints.values()):
            npt.assert_array_equal(actual, expected)


@patch(
    "atlas_building_tools.densities.mtype_densities.DensityProfileCollection.load_placement_hints",
    return_value=create_placement_hints(),
)
def test_compute_layer_slice_voxel_indices(mocked_loader):
    density_profile_collection = create_density_profile_collection()
    actual_voxel_indices = density_profile_collection.compute_layer_slice_voxel_indices(
        {
            "layer_2": "[PH]layer_2.nrrd",
            "layer_3": "[PH]layer_3.nrrd",
            "y": "[PH]y.nrrd",
        }
    )  # fake file paths

    expected = create_slice_voxel_indices()
    for slice_index in range(0, 5):
        npt.assert_array_equal(actual_voxel_indices[slice_index], expected[slice_index])


def test_create_mtype_density():
    density_profile_collection = create_density_profile_collection()
    with tempfile.TemporaryDirectory() as tempdir:
        density_profile_collection.create_density(
            "L2_TPC:A",
            "excitatory",
            create_excitatory_neuron_density(),
            create_slice_voxel_indices(),
            tempdir,
        )
        expected_cell_densities = create_expected_cell_densities()
        created_cell_density = VoxelData.load_nrrd(str(Path(tempdir, "L2_TPC:A_density.nrrd"))).raw
        npt.assert_array_equal(created_cell_density, expected_cell_densities["L2_TPC:A"])
        density_profile_collection.create_density(
            "L23_BP",
            "inhibitory",
            create_inhibitory_neuron_density(),
            create_slice_voxel_indices(),
            tempdir,
        )
        created_cell_density = VoxelData.load_nrrd(str(Path(tempdir, "L23_BP_density.nrrd"))).raw
        npt.assert_array_equal(created_cell_density, expected_cell_densities["L23_BP"])


def test_create_mtype_densities():
    density_profile_collection = create_density_profile_collection()
    with tempfile.TemporaryDirectory() as tempdir:
        paths = create_input_volumetric_data(tempdir)
        density_profile_collection.create_mtype_densities(
            paths["excitatory_neuron_density"],
            paths["inhibitory_neuron_density"],
            paths["placement_hints_config"],
            tempdir,
        )
        expected_cell_densities = create_expected_cell_densities()
        for mtype, expected_cell_density in expected_cell_densities.items():
            created_cell_density = VoxelData.load_nrrd(
                str(Path(tempdir, f"{mtype}_density.nrrd"))
            ).raw
            npt.assert_array_equal(created_cell_density, expected_cell_density)
