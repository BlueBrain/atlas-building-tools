'''test app/mtype_densities'''
from pathlib import Path

import numpy.testing as npt
from click.testing import CliRunner

from voxcell import VoxelData  # type: ignore

import atlas_building_tools.app.cell_densities as tested
from tests.densities.test_mtype_densities import (
    DATA_PATH,
    create_input_volumetric_data,
    create_expected_cell_densities,
)


def test_mtype_densities():
    runner = CliRunner()
    with runner.isolated_filesystem():
        paths = create_input_volumetric_data('.')
        result = runner.invoke(
            tested.mtype_densities,
            [
                '--excitatory-neuron-density-path',
                paths['excitatory_neuron_density'],
                '--inhibitory-neuron-density-path',
                paths['inhibitory_neuron_density'],
                '--placement-hints-config-path',
                paths['placement_hints_config'],
                '--mtype-to-profile-map-path',
                str(DATA_PATH / 'meta' / 'mapping.tsv'),
                '--layer-slices-path',
                str(DATA_PATH / 'meta' / 'layers.tsv'),
                '--density-profiles-dir',
                str(DATA_PATH / 'mtypes'),
                '--output-dir',
                'output_dir',
            ],
        )
        assert result.exit_code == 0
        expected_cell_densities = create_expected_cell_densities()
        for mtype, expected_cell_density in expected_cell_densities.items():
            created_cell_density = VoxelData.load_nrrd(
                str(Path('output_dir', f'{mtype}_density.nrrd'))
            ).raw
            npt.assert_array_equal(created_cell_density, expected_cell_density)
