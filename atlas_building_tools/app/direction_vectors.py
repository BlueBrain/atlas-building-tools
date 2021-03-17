'''Generate and save the direction vectors for different regions of the mouse brain'''
import logging
import click  # type: ignore

import voxcell  # type: ignore

from atlas_building_tools.direction_vectors import cerebellum as cerebellum_  # type: ignore
from atlas_building_tools.direction_vectors import isocortex as isocortex_  # type: ignore
from atlas_building_tools.app.utils import log_args, EXISTING_FILE_PATH, set_verbose  # type: ignore

L = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    '''Run the different direction vectors CLI

    Direction vectors are 3D unit vectors associated to voxels of a brain region.
    They represent the directions of the fiber tracts and their streamlines are assumed
    to cross orthogonally layers in laminar brain regions.

    Direction vectors are used in are used in placement-algorithm to set cells orientations.

    Direction vectors are also used to compute placement hints (see the placement_hints module)
    and split layer 2/3 of the AIBS mouse isocortex.
    '''
    set_verbose(L, verbose)


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the whole AIBS mouse brain annotation nrrd file.'),
)
@click.option(
    '--output-path',
    required=True,
    help='Path of file to write the direction vectors to.',
)
@log_args(L)
def cerebellum(annotation_path, output_path):
    '''Generate and save the direction vectors of the AIBS mouse cerebellum

    This command relies on the computation of the gradient of Gaussian blur
    applied to specific parts of the cerebellum.

    The output file is an nrrd file enclosing a float array of shape (W, H, D, 3)
    where (W, H, D) is the shape the input annotation array.

    Note: At the moment, direction vectors are generated only for the following cerebellum
    subregions:\n
        - the flocculus\n
        - the lingula\n
    The vector [nan, nan, nan] is assigned to any voxel outside the above two regions.

    '''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    dir_vectors = cerebellum_.compute_direction_vectors(annotation.raw)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the annotation nrrd file.'),
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
    '''Generate and save the direction vectors of the mouse isocortex

    This command relies on Regiodesics.

    The output file is an nrrd file enclosing a float array of shape (W, H, D, 3)
    where (W, H, D) is the shape the input annotation array.

    The vector [nan, nan, nan] is assigned to any voxel out of the isocortex region.
    '''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    dir_vectors = isocortex_.compute_direction_vectors(region_map, annotation)
    annotation.with_data(dir_vectors).save_nrrd(output_path)
