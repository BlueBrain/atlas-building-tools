'''test cell_positions'''
from pathlib import Path
import h5py
import yaml

import numpy as np
from click.testing import CliRunner

from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.cell_positions as tested


TEST_PATH = Path(Path(__file__).parent.parent)


def get_config():
    return {
        'inputDensityVolumePath': {
            'oligodendrocyte': 'oligodendrocyte.nrrd',
            'astrocyte': 'astrocyte.nrrd',
            'microglia': 'microglia.nrrd',
        }
    }


def test_cell_positions():
    astrocytes = np.zeros((3, 3, 3), dtype=int)
    astrocytes[0, :, :] = 100.0
    microglia = np.zeros((3, 3, 3), dtype=int)
    microglia[1, :, :] = 100.0
    oligodendrocytes = np.zeros((3, 3, 3), dtype=int)
    oligodendrocytes[2, :, :] = 100.0
    input_ = {
        'annotation': np.ones((3, 3, 3), dtype=int),
        'astrocyte': astrocytes,
        'microglia': microglia,
        'oligodendrocyte': oligodendrocytes,
    }
    cell_positions_config = get_config()
    voxel_dimensions = [25] * 3
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('cell_positions_config.yaml', 'w') as out:
            yaml.dump(cell_positions_config, out)
        for name, array in input_.items():
            VoxelData(array, voxel_dimensions=voxel_dimensions).save_nrrd(
                f'{name}.nrrd'
            )
        result = runner.invoke(
            tested.cmd,
            [
                '--annotation-path',
                'annotation.nrrd',
                '--config',
                'cell_positions_config.yaml',
                '--output-path',
                'cell_positions.h5',
            ],
        )
        assert result.exit_code == 0
        with h5py.File('cell_positions.h5', 'r') as file_:
            dataset = file_['positions']
            assert 'cell positions' in dataset.attrs['description']
            assert len(np.array(dataset)) == 2700


def test_cell_positions_exeception():
    input_ = {
        'annotation': np.ones((3, 3, 3), dtype=int),
        'astrocyte': np.ones((3, 3, 3), dtype=int),
        'microglia': np.ones((3, 3, 3), dtype=int),
        'oligodendrocyte': np.ones((3, 3, 3), dtype=int),
    }
    cell_positions_config = get_config()
    voxel_dimensions = [[25.0] * 3, [30.0] * 3, [25.0] * 3, [25.0] * 3]
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('cell_positions_config.yaml', 'w') as out:
            yaml.dump(cell_positions_config, out)
        for (name, array), vox_dim in zip(input_.items(), voxel_dimensions):
            VoxelData(array, voxel_dimensions=vox_dim).save_nrrd(f'{name}.nrrd')
        result = runner.invoke(
            tested.cmd,
            [
                '--annotation-path',
                'annotation.nrrd',
                '--config',
                'cell_positions_config.yaml',
                '--output-path',
                'cell_positions.h5',
            ],
        )
        assert 'different voxel dimensions' in str(result.exception)
        assert result.exit_code == 1

    offsets = [[0.0] * 3, [10.0] * 3, [0.0] * 3, [15.0] * 3]
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open('cell_positions_config.yaml', 'w') as out:
            yaml.dump(cell_positions_config, out)
        for (name, array), offset in zip(input_.items(), offsets):
            VoxelData(array, voxel_dimensions=[25.0] * 3, offset=offset).save_nrrd(
                f'{name}.nrrd'
            )
        result = runner.invoke(
            tested.cmd,
            [
                '--annotation-path',
                'annotation.nrrd',
                '--config',
                'cell_positions_config.yaml',
                '--output-path',
                'cell_positions.h5',
            ],
        )
        assert 'different offsets' in str(result.exception)
        assert result.exit_code == 1
