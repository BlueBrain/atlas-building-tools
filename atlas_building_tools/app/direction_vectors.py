'''Generate and save the direction vectors for different regions of the mouse brain'''
import logging
import click  # type: ignore

import voxcell  # type: ignore

from atlas_building_tools.direction_vectors import cerebellum as cerebellum_  # type: ignore
from atlas_building_tools.direction_vectors import isocortex as isocortex_  # type: ignore
from atlas_building_tools.app.utils import log_args, EXISTING_FILE_PATH, set_verbose  # type: ignore


L = logging.getLogger('Direction vectors')


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    '''Run the different direction vectors CLI'''
    set_verbose(L, verbose)


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the whole mouse brain annotation nrrd file.'),
)
@click.option('--output-path', required=True, help='Path of file to write.')
@log_args(L)
def cerebellum(annotation_path, output_path):
    '''Generate and save the direction vectors of the mouse Cerebellum'''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    dir_vectors = cerebellum_.compute_direction_vectors(annotation.raw)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the Isocortex annotation nrrd file.'),
)
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to hierarchy.json or AIBS 1.json.'),
)
@click.option('--output-path', required=True, help='Path of file to write.')
@log_args(L)
def isocortex(annotation_path, hierarchy_path, output_path):
    '''Generate and save the direction vectors of the mouse Isocortex'''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    dir_vectors = isocortex_.compute_direction_vectors(region_map, annotation)
    annotation.with_data(dir_vectors).save_nrrd(output_path)
