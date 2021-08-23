"""Refine annotations by splitting brain regions"""
import json
import logging

import click
import voxcell  # type: ignore

from atlas_building_tools.app.utils import (
    EXISTING_FILE_PATH,
    common_atlas_options,
    log_args,
    set_verbose,
)
from atlas_building_tools.region_splitter import isocortex_layer_23

L = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the region splitter CLI"""
    set_verbose(L, verbose)


@app.command()
@common_atlas_options
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the mouse isocortex direction vectors nrrd file."),
)
@click.option("--output-hierarchy-path", required=True, help="Path of the json file to write")
@click.option("--output-annotation-path", required=True, help="Path of the nrrd file to write")
@log_args(L)
def split_isocortex_layer_23(
    annotation_path,
    hierarchy_path,
    direction_vectors_path,
    output_hierarchy_path,
    output_annotation_path,
):
    """Split the layer 2/3 of the AIBS mouse isocortex and save modified hierarchy and
    annotation files.

    Two new identifiers are created for each subregion of the AIBS mouse iscortex whose name
    and acronym ends with "2/3". The modification of the hierarchy file is independent
    of the input annotated volume.
    """
    L.info("Loading files ...")
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    with open(hierarchy_path, "r") as h_file:
        hierarchy = json.load(h_file)
    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)
    # Splits and updates in place hierarchy and annotations
    L.info("Splitting layer 2/3 in layer 2 and layer 3 ...")
    isocortex_layer_23.split(hierarchy, annotation, direction_vectors.raw)
    L.info("Saving modified hierarchy and annotation files ...")
    with open(output_hierarchy_path, "w") as out:
        json.dump(hierarchy, out, indent=1, separators=(",", ": "))
    annotation.save_nrrd(output_annotation_path)
