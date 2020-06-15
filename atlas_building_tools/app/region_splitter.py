'''Refine annotations by splitting brain regions'''
import logging
import json
import click

import voxcell  # type: ignore

from atlas_building_tools.region_splitter import isocortex_layer_23

from atlas_building_tools.app.utils import log_args, EXISTING_FILE_PATH, set_verbose

L = logging.getLogger('Direction vectors')


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    '''Run the region splitter CLI'''
    set_verbose(L, verbose)


@app.command()
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the whole mouse brain hierarchy file.'),
)
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the whole mouse brain annotation nrrd file.'),
)
@click.option(
    '--direction-vectors-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the mouse isocortex direction vectors nrrd file.'),
)
@click.option('--output-hierarchy-path', required=True, help='path of file to write')
@click.option('--output-annotation-path', required=True, help='path of file to write')
@log_args(L)
def split_isocortex_layer_23(
    hierarchy_path,
    annotation_path,
    direction_vectors_path,
    output_hierarchy_path,
    output_annotation_path,
):
    '''Split the layer 2/3 in the mouse isocortex and save modified hierarchy and annotations.'''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    with open(hierarchy_path, 'r') as h_file:
        hierarchy = json.load(h_file)
    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)
    # Splits and updates in place hierarchy and annotations
    isocortex_layer_23.split(hierarchy, annotation, direction_vectors.raw)
    annotation.save_nrrd(output_annotation_path)
    with open(output_hierarchy_path, 'w') as out:
        json.dump(hierarchy, out, indent=1, separators=(',', ': '))
