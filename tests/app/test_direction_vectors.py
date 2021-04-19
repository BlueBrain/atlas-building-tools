"""test app/direction_vectors"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.direction_vectors as tested
from tests.direction_vectors.test_thalamus import create_voxeldata

TEST_PATH = Path(Path(__file__).parent.parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))


def test_thalamus():
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_data = create_voxeldata(
            718,  # VPL, Ventral posterolateral nucleus of the thalamus
            709,  # VPM, Ventral posteromedial nucleus of the thalamus
        )
        voxel_data.save_nrrd("annotation.nrrd")
        result = runner.invoke(
            tested.thalamus,
            [
                "--annotation-path",
                "annotation.nrrd",
                "--hierarchy-path",
                HIERARCHY_PATH,
                "--output-path",
                "direction_vectors.nrrd",
            ],
        )
        assert result.exit_code == 0
        direction_vectors = VoxelData.load_nrrd("direction_vectors.nrrd")
        npt.assert_array_equal(direction_vectors.raw.shape, (12, 12, 12, 3))
        assert direction_vectors.raw.dtype == np.float32
