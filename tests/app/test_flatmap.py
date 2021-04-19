"""test flatmap"""
import json

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.flatmap as tested
from tests.flatmap.test_streamlines_flatmap import get_layers, get_vector_field


def get_hierarchy():
    return {
        "id": 315,
        "acronym": "Isocortex",
        "name": "Isocortex",
        "children": [
            {
                "id": 500,
                "acronym": "MO",
                "name": "Somatomotor areas",
                "children": [
                    {
                        "id": 1,
                        "acronym": "MO1",
                        "name": "Somatomotor areas, Layer 1",
                        "children": [],
                    },
                    {
                        "id": 2,
                        "acronym": "MO2",
                        "name": "Somatomotor areas, Layer 2",
                        "children": [],
                    },
                    {
                        "id": 3,
                        "acronym": "MO3",
                        "name": "Somatomotor areas, Layer 3",
                        "children": [],
                    },
                    {
                        "id": 4,
                        "acronym": "MO4",
                        "name": "Somatomotor areas, Layer 4",
                        "children": [],
                    },
                ],
            }
        ],
    }


def get_metadata():
    return {
        "layers": {
            "names": ["layer_1", "layer_2", "layer_3", "layer_4"],
            "queries": ["@.* Layer 1", "@.* Layer 2", "@.* Layer 3", "@.* Layer 4"],
            "attribute": "name",
        }
    }


def get_result(runner, first_layer="layer_2"):
    return runner.invoke(
        tested.app,
        [
            "streamlines-flatmap",
            "--annotation-path",
            "annotation.nrrd",
            "--direction-vectors-path",
            "direction_vectors.nrrd",
            "--hierarchy-path",
            "hierarchy.json",
            "--metadata-path",
            "metadata.json",
            "--output-path",
            "flatmap.nrrd",
            "--first-layer",
            first_layer,
            "--second-layer",
            "layer_3",
            "--resolution",
            25,
        ],
    )


def create_files():
    vector_field = get_vector_field()
    vector_field.save_nrrd("direction_vectors.nrrd")
    vector_field.with_data(get_layers()).save_nrrd("annotation.nrrd")
    with open("hierarchy.json", "w") as file_:
        json.dump(get_hierarchy(), file_)
    with open("metadata.json", "w") as file_:
        json.dump(get_metadata(), file_)


def test_flatmap():
    runner = CliRunner()
    with runner.isolated_filesystem():
        create_files()
        result = get_result(runner)
        assert result.exit_code == 0
        flatmap_voxel_data = VoxelData.load_nrrd("flatmap.nrrd")
        npt.assert_array_equal(flatmap_voxel_data.raw.shape, [24, 7, 7, 2])
        assert flatmap_voxel_data.raw.dtype == int
        npt.assert_allclose(flatmap_voxel_data.raw[(3, 3, 3)], [12.5, 12.5], rtol=0.15)


def test_flatmap_assertion_voxel_data():
    runner = CliRunner()
    with runner.isolated_filesystem():
        vector_field = get_vector_field()
        vector_field.save_nrrd("direction_vectors.nrrd")
        VoxelData(
            get_layers(), offset=vector_field.offset, voxel_dimensions=[1.0, 1.0, 1.0]
        ).save_nrrd("annotation.nrrd")
        with open("hierarchy.json", "w") as file_:
            json.dump(get_hierarchy(), file_)
        with open("metadata.json", "w") as file_:
            json.dump(get_metadata(), file_)

        # Test assertion when voxel dimensions don't match
        result = get_result(runner)
        assert "voxel_dimensions" in str(result.exception)

        # Restore previous annotation file
        vector_field.with_data(get_layers()).save_nrrd("annotation.nrrd")

        # Test assertion when some direction vector has NaN coordinates
        vector_field.raw[3, 3, 3] = [np.nan, 0.0, 0.0]
        vector_field.save_nrrd("direction_vectors.nrrd")
        result = get_result(runner)
        assert "NaN" in str(result.exception)

        # Test assertion raised when some direction vector is zero
        vector_field.raw[3, 3, 3] = [0.0, 0.0, 0.0]
        vector_field.save_nrrd("direction_vectors.nrrd")
        result = get_result(runner)
        assert "zero" in str(result.exception)


def test_flatmap_assertion_layer_names():
    runner = CliRunner()
    with runner.isolated_filesystem():
        create_files()
        # Test assertion raised when one of the layer names is unknown.
        result = get_result(runner, first_layer="layer 2")  # missing underscore
        assert "layer name" in str(result.exception)
