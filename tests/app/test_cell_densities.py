'''test cell_densities'''
from pathlib import Path

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.cell_densities as tested

TEST_PATH = Path(Path(__file__).parent.parent)


def _get_result(runner):
    args = [
        'cell-density',
        '--hierarchy-path',
        str(Path(TEST_PATH, '1.json')),
        '--annotation-path',
        'annotation.nrrd',
        '--nissl-path',
        'nissl.nrrd',
        '--output-path',
        'overall_cell_density.nrrd',
    ]

    return runner.invoke(tested.app, args)


def test_densities():
    input_ = {
        'annotation': np.array(
            [
                [[512, 512, 1143]],
                [[512, 512, 1143]],
                [[477, 56, 485]],
            ]
        ),
        'nissl': np.array(
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
            VoxelData(array, voxel_dimensions=voxel_dimensions).save_nrrd(
                f'{name}.nrrd'
            )
        result = _get_result(runner)

        assert result.exit_code == 0
        voxel_data = VoxelData.load_nrrd('overall_cell_density.nrrd')
        assert voxel_data.raw.dtype == float

        # An error should be raised if annotation and nissl don't use the same voxel dimensions
        VoxelData(np.ones((3, 1, 3)), voxel_dimensions=[10] * 3).save_nrrd('nissl.nrrd')
        result = _get_result(runner)
        assert 'voxel_dimensions' in str(result.exception)
