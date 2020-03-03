'''Generate and save the combined annotation file'''
import click

import voxcell
from atlas_building_tools import annotations_combinator
from atlas_building_tools.app.utils import REQUIRED_PATH


@click.command()
@click.option(
    '--hierarchy',
    type=REQUIRED_PATH,
    required=True,
    help='Path to hierarchy.json or 1.json',
)
@click.option(
    '--brain-annotation-ccfv2',
    type=REQUIRED_PATH,
    required=True,
    help=('This brain annotation file contains the most complete annotation.'),
)
@click.option(
    '--fiber-annotation-ccfv2',
    type=REQUIRED_PATH,
    required=True,
    help='Fiber annotation is not included in the CCF-v2 2011 annotation files.',
)
@click.option(
    '--brain-annotation-ccfv3',
    type=REQUIRED_PATH,
    required=True,
    help=('More recent brain annotation file with missing leaf regions.'),
)
@click.option('--output-path', required=True, help='path of file to write')
def cmd(
    hierarchy,
    brain_annotation_ccfv2,
    fiber_annotation_ccfv2,
    brain_annotation_ccfv3,
    output_path,
):
    '''Generate and save the combined annotation file'''
    # The annotation file `brain_annotation_ccfv3` is the annotation file containing
    # the least complete annotation.

    # There are so far only two use cases, on for each resolution (10 um or 25 um).
    # For a resolution of 10 um, the path arguments should be the following:
    # - hierarchy = path to 1.json from AIBS
    # (http://api.brain-map.org/api/v2/structure_graph_download/1.json)
    # For a resolution of 10 um, the arguments should be the following.
    # - brain_annotation_ccfv2 = path to annotation_10_2011.nrrd
    # pylint: disable=line-too-long
    # (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/mouse_2011/annotation_10.nrrd)
    # - fiber_annotation = path to annotationFiber_10_2011.nrrd
    # (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/mouse_2011/annotationFiber_10.nrrd)
    # - brain_annotation_ccfv3 = path to annotation_10_2017.nrrd
    # (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd)

    region_map = voxcell.RegionMap.load_json(hierarchy)

    brain_annotation_ccfv2 = voxcell.VoxelData.load_nrrd(brain_annotation_ccfv2)
    fiber_annotation_ccfv2 = voxcell.VoxelData.load_nrrd(fiber_annotation_ccfv2)
    brain_annotation_ccfv3 = voxcell.VoxelData.load_nrrd(brain_annotation_ccfv3)

    combined_annotation = annotations_combinator.combine_annotations(
        region_map,
        brain_annotation_ccfv2,
        fiber_annotation_ccfv2,
        brain_annotation_ccfv3,
    )
    combined_annotation.save_nrrd(output_path)
