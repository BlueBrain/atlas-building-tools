'''Generate and save the placement hints of different regions of the mouse brain

See https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/index.html
for the specifications of the placement hints.
'''
import os
from pathlib import Path
import json
import logging
from typing import List, Optional
import click

import voxcell  # type: ignore

from atlas_building_tools.placement_hints.compute_placement_hints import (
    compute_placement_hints,
)
from atlas_building_tools.placement_hints.utils import save_placement_hints
from atlas_building_tools.placement_hints.layered_atlas import save_problematic_volume
from atlas_building_tools.app.utils import log_args, EXISTING_FILE_PATH, set_verbose  # type: ignore

L = logging.getLogger(__name__)


# pylint: disable=too-many-arguments, too-many-locals
def _placement_hints(
    annotation_path: str,
    hierarchy_path: str,
    direction_vectors_path: str,
    output_dir: str,
    region_acronym: str,
    layer_regexps: List[str],
    placement_hint_names: List[str],
    max_thicknesses: Optional[List[float]] = None,
    flip_direction_vectors: bool = False,
    has_hemispheres: bool = False,
) -> None:
    '''
    Compute the placement hints for a laminar region of the mouse brain.

    Args:
        annotation_path: path to the whole mouse brain annotation nrrd file.
        hierarchy_path: path to hierarchy.json.
        direction_vectors_path: path to the `region_arconym` direction vectors file, e.g.,
            direction_vectors.nrrd.
        output_dir: path to the output directory.
        region_acronym: acronym of the region for which the computation is requested.
            Example: 'CA1', 'Isocortex'.
        layer_regexps: list of regular expressions defining the layers of `region_acronym`.
        placement_hint_names: list of names to be used when saving the layer placement hints.
        max_thicknesses: (optional) thicknesses of `region_acronym` layers.
            Defaults to None, i.e., there will be no validity check with input from literature.
        flip_direction_vectors: (optional) if True, the input direction vectors are negated before
            use. This is required if direction vectors flaw from the top layer (shallowest) to the
            bottom layer (deepest). Otherwise, they are left unchanged. Defaults to false.
         has_hemispheres: (optional) If True, split the volume into halves along the z-axis and
            handle each of theses 'hemispheres' separately. Otherwise the whole volume is handled.
            Defaults to True.
    '''
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)
    (distances_info, distances_report, problematic_volume,) = compute_placement_hints(
        region_map,
        annotation,
        region_acronym,
        layer_regexps,
        direction_vectors.raw,
        max_thicknesses,
        flip_direction_vectors=flip_direction_vectors,
        has_hemispheres=has_hemispheres,
    )
    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    with open(Path(output_dir, 'distance_report.json'), mode='w+') as file_:
        json.dump(distances_report, file_)
    save_placement_hints(
        distances_info['distances_to_layer_meshes'],
        output_dir,
        distances_info['layered_atlas'].region,
        placement_hint_names,
    )
    # The problematic volume is a binary mask of the voxels for which distance computations failed.
    # For such voxels, distance information is not reliable.
    # See atlas_building_tools.distances.distances_to_meshes.report_problems.
    save_problematic_volume(
        distances_info['layered_atlas'], problematic_volume, output_dir
    )


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    '''Run the different placement hints CLI'''
    set_verbose(L, verbose)


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the whole mouse brain annotation nrrd file.'),
)
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to hierarchy.json.'),
)
@click.option(
    '--direction-vectors-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the CA1 direction vectors file, e.g., direction_vectors.nrrd.'),
)
@click.option(
    '--output-dir',
    required=True,
    help='path of the directory to write.' ' It will be created if it doesn\'t exist',
)
@log_args(L)
def ca1(annotation_path, hierarchy_path, direction_vectors_path, output_dir):
    '''Generate and save the placement hints for the CA1 region of the mouse hippocampus'''

    layer_regexps = ["CA1so", "CA1sp", "CA1sr", "CA1slm"]
    _placement_hints(
        annotation_path,
        hierarchy_path,
        direction_vectors_path,
        output_dir,
        'CA1',
        layer_regexps,
        layer_regexps,
        flip_direction_vectors=True,
        has_hemispheres=False,
    )


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to the whole mouse brain annotation nrrd file.'),
)
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('Path to hierarchy.json.'),
)
@click.option(
    '--direction-vectors-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        'Path to the isocortex direction vectors file, e.g., direction_vectors.nrrd.'
    ),
)
@click.option(
    '--output-dir',
    required=True,
    help='path of the directory to write.' ' It will be created if it doesn\'t exist',
)
@log_args(L)
def isocortex(annotation_path, hierarchy_path, direction_vectors_path, output_dir):
    '''Generate and save the placement hints of the mouse isocortex'''

    _placement_hints(
        annotation_path,
        hierarchy_path,
        direction_vectors_path,
        output_dir,
        'Isocortex',
        ['@.*{}[ab]?$'.format(i) for i in range(1, 7)],
        [f'layer_{i}' for i in range(1, 7)],
        # Layer thicknesses from J. Defilipe 2017 (unpublished), see Section 5.1.1.4
        # of the release report "Neocortex Tissue Reconstruction",
        # https://github.com/BlueBrain/ncx_release_report.git
        max_thicknesses=[210.639, 190.2134, 450.6398, 242.554, 670.2, 893.62],
        has_hemispheres=True,
    )
