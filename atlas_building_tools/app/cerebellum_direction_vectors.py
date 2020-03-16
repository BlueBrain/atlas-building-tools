'''Generate and save the direction vectors of the mouse cerebellum'''
import click

import voxcell

from atlas_building_tools.direction_vectors import cerebellum
from atlas_building_tools.app.utils import REQUIRED_PATH


@click.command()
@click.option(
    '--annotation-path',
    type=REQUIRED_PATH,
    required=True,
    help=('Path to the cerebellum annotation nrrd file.'),
)
@click.option('--output-path', required=True, help='path of file to write')
def cmd(annotation_path, output_path):
    '''Generate and save the direction vectors of the cerebellum'''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    direction_vectors = cerebellum.compute_cerebellum_direction_vectors(annotation)
    direction_vectors.save_nrrd(output_path)
