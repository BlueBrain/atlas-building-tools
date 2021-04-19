"""Generate a flatmap for different regions of the mouse brain

This module aims at computing a voxel-to-pixel map flattening a 3D annotated volume.

The available flattening transformation consists in 'projecting' each voxel of the volume onto a
the selected surface along the streamlines of the direction vectors of the volume fibers.
The mapped surface is subequently flattened while locally minimizing the area distortion.
The input volume can be for instance the primary somato sensory cortex of the rat SSCX atlas by
Paxinos & Watson, https://www.frontiersin.org/articles/10.3389/neuro.11.004.2007/full.
(BBP Atlas files, including hierarchy.json, reside in /gpfs/bbp.cscs.ch/project/proj39.)
"""
import json
import logging

import click  # type: ignore
import numpy as np
import voxcell  # type: ignore
from atlas_analysis.atlas import assert_meta_properties  # type: ignore

from atlas_building_tools.app.utils import EXISTING_FILE_PATH  # type: ignore
from atlas_building_tools.app.utils import log_args, set_verbose
from atlas_building_tools.exceptions import AtlasBuildingToolsError
from atlas_building_tools.flatmap.streamlines_flatmap import compute_flatmap
from atlas_building_tools.flatmap.utils import create_layers_volume

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the different flatmap CLI"""
    set_verbose(L, verbose)


@app.command()
@click.option(
    "--annotation-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the brain region annotation (nrrd file)."),
)
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the direction vectors of the brain region (nrrd file)."
        "This is a 3D unit vector field defined on the brain regions. It streamlines reflects"
        " the spatial distribution of fiber tracts."
    ),
)
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the hierarchy file of the brain region (json file)."
        "This is either the AIBS 1.json or a file with a similar structure."
    ),
)
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the metadata file of the brain region (json file). This file encloses the "
        "definition of the region layers by means of regular expressions that be read by "
        "voxcell.RegionMap.find. See atlas-building-tools/atlas_building_tools/data/metadata.json"
        " for an exmample."
    ),
)
@click.option(
    "--output-path",
    required=True,
    help=(
        "Path of the nrrd file to write. A flatmap nrrd "
        "file encloses the content of an integer numpy array of shape (W, H, D, 2) where W, H and D"
        " are the dimensions of the brain region domain. The last axis is used to encode pixel "
        "coordinates. A flatmap maps voxels of the brain region to pixels of a binary 2D image. "
        "Negative pixel coordinates are used for voxels out of the brain region but also or for "
        "voxels which could not be mapped to the image."
    ),
)
@click.option(
    "--first-layer",
    type=str,
    required=True,
    help=("Name of the first layer as listed in the metadata json file."),
)
@click.option(
    "--second-layer",
    type=str,
    required=True,
    help=(
        "Name of the second layer as listed in the metadata json file."
        " The first and second layers of the brain region are separated by a boundary surface."
        " The direction vectors streamlines intersect this surface, mapping thus voxels to surface"
        " points. The surface is eventually flattened by an authalic transformation."
    ),
)
@click.option(
    "--acronym",
    type=str,
    required=False,
    help=(
        "Acronym of the subregion to be flattened; acronyms are listed in the hierarchy json file."
        ' Examples: "S1", "Isocortex". Defaults to None.'
    ),
    default=None,
)
@click.option(
    "--resolution",
    type=int,
    required=False,
    help=(
        "target width of the output flatmap. The flatmap image will be constrained to fit in a 2D"
        " rectangle image whose width along the x-axis is `resolution`. Defaults to 500."
    ),
    default=500,
)
@log_args(L)
def streamlines_flatmap(
    annotation_path,
    direction_vectors_path,
    hierarchy_path,
    metadata_path,
    output_path,
    first_layer,
    second_layer,
    acronym,
    resolution,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Generate and save the flatmap of a brain region obtained by collapsing
    the direction vectors streamlines.\n

    The brain region to be flattened is assumed to be laminar, i.e., foliated by layers.
    The brain region is bundled with its 3D direction vectors whose streamlines are used to
    create a voxel-to-pixel mapping in the following way.
    For each voxel, a streamline passing through it is drawn. Here we call a streamline a polygonal
    line following the stream of the discrete `direction_vectors` field.
    The streamline intersects the boundary surface shared by `first_layer` and `second_layer`.
    The map assigning intersection points to voxels is finally post-composed with an authalic
    transformation, i.e., a transformation flattening the boundary surface while minimizing locally
    area distorsion.
    """

    L.info("Loading files ...")
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)

    # Check nrrd metadata consistency
    assert_meta_properties([annotation, direction_vectors])

    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    with open(metadata_path, "r") as file_:
        metadata = json.load(file_)

    subregion_ids = (
        region_map.find(acronym, attr="acronym", with_descendants=True)
        if acronym is not None
        else None
    )

    L.info("Labelling the layers of the input volume ...")
    layers = create_layers_volume(annotation.raw, region_map, metadata, subregion_ids)

    # Direction vectors are required to be valid when restricted to the specified region
    L.info("Checking direction vectors sanity ...")
    norms = np.linalg.norm(direction_vectors.raw[layers > 0], axis=1)
    if np.any(np.isnan(norms)):
        raise AtlasBuildingToolsError(
            "Some direction vectors of the specified region have NaN coordinates."
            " Streamlines cannot be drawn."
        )

    if np.any(norms == 0.0):
        raise AtlasBuildingToolsError(
            "Some direction vectors of the specified region are zero."
            "Streamlines cannot be drawned."
        )

    layer_names = metadata["layers"]["names"]
    for layer_name in [first_layer, second_layer]:
        if layer_name not in layer_names:
            raise AtlasBuildingToolsError(
                f"The layer name {layer_name} could not be found in the provided metadata"
            )

    L.info("Computing a flatmap with resolution %s ...", resolution)
    flatmap = compute_flatmap(
        layers,
        direction_vectors,
        layer_names.index(first_layer) + 1,
        layer_names.index(second_layer) + 1,
        resolution=resolution,
    )
    L.info("Saving the flatmap to %s...", output_path)
    annotation.with_data(flatmap).save_nrrd(output_path)
