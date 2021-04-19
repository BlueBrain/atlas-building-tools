"""Generate and save combined annotations or combined markers

Combination operates on two or more volumetric files with nrrd format.
"""
import json
import logging

import click
import pandas as pd
import voxcell  # type: ignore
import yaml

from atlas_building_tools.app.utils import EXISTING_FILE_PATH, log_args, set_verbose
from atlas_building_tools.combination import annotations_combinator, markers_combinator

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the combination CLI"""
    set_verbose(L, verbose)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
@app.command()
@click.option(
    "--hierarchy",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to hierarchy.json or 1.json",
)
@click.option(
    "--brain-annotation-ccfv2",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("This brain annotation file contains the most complete annotation."),
)
@click.option(
    "--fiber-annotation-ccfv2",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Fiber annotation is not included in the CCF-v2 2011 annotation files.",
)
@click.option(
    "--brain-annotation-ccfv3",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("More recent brain annotation file with missing leaf regions."),
)
@click.option("--output-path", required=True, help="path of file to write")
@log_args(L)
def combine_annotations(
    hierarchy,
    brain_annotation_ccfv2,
    fiber_annotation_ccfv2,
    brain_annotation_ccfv3,
    output_path,
):
    # pylint: disable=line-too-long
    """Generate and save the combined annotation file

    The annotation file `brain_annotation_ccfv3` is the annotation file containing
     the least complete annotation.\n

    There are so far only two use cases, one for each resolution (10 um or 25 um).
    For a resolution of 10 um, the path arguments should be the following:\n

     - hierarchy = path to 1.json from AIBS
    (http://api.brain-map.org/api/v2/structure_graph_download/1.json)\n
     - brain_annotation_ccfv2 = path to annotation_10_2011.nrrd
     (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/mouse_2011/annotation_10.nrrd)\n
     - fiber_annotation = path to annotationFiber_10_2011.nrrd
     (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/mouse_2011/annotationFiber_10.nrrd)\n
     - brain_annotation_ccfv3 = path to annotation_10_2017.nrrd
     (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd)\n
    """

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


@app.command()
@click.option(
    "--hierarchy",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to hierarchy.json or 1.json",
)
@click.option(
    "--brain-annotation",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the whole mouse rain annotation file."),
)
@click.option(
    "--config",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "Path to the gene markers configuration file."
        "This is a yaml file indicating which markers are used"
        " to identify the different glia cell types."
        "It contains the path to the gene marker volumes "
        "as well as their average expression intensities "
        "and the glia intensity output paths."
    ),
)
@log_args(L)
def combine_markers(hierarchy, brain_annotation, config):
    """Generate and save the combined glia files and the global celltype scaling factors

    This function performs the operations indicated by the formula of the
    'Glia differentiation' section in 'A Cell Atlas for the Mouse Brain' by
    C. Eroe et al., 2018.
     https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full.

     The output consists in:

     * A 3D volumetric file for each cell type (oligodendrocyte, astrocyte, microglia)
      representing the average density of each cell type, up to a uniform constant factor.\n

     * A 3D volumetric file representing the overall average density of the glia in the
      whole mouse brain up to the same uniform constant factor.\n

     * The global celltype scaling factors S_celltype of the 'Glia differentiation section in
      'A Cell Atlas for the Mouse Brain' by C. Eroe et al. 2018,
      i.e., the proportions of each glia cell type in the whole mouse brain.\n
    """
    hierarchy = voxcell.RegionMap.load_json(hierarchy)
    annotation = voxcell.VoxelData.load_nrrd(brain_annotation)
    config = yaml.load(open(config), Loader=yaml.FullLoader)
    glia_celltype_densities = pd.DataFrame(config["cellDensity"])
    combination_data = pd.DataFrame(config["combination"])
    volumes = pd.DataFrame(
        [
            [gene, voxcell.VoxelData.load_nrrd(path).raw]
            for (gene, path) in config["inputGeneVolumePath"].items()
        ],
        columns=["gene", "volume"],
    )
    combination_data = combination_data.merge(volumes, how="inner", on="gene")

    glia_intensities = markers_combinator.combine(
        hierarchy, annotation.raw, glia_celltype_densities, combination_data
    )

    for type_, output_path in config["outputCellTypeVolumePath"].items():
        annotation.with_data(glia_intensities.intensity[type_]).save_nrrd(output_path)

    annotation.with_data(glia_intensities.intensity["glia"]).save_nrrd(
        config["outputOverallGliaVolumePath"]
    )

    proportions = dict(glia_intensities.proportion.astype(str))
    with open(config["outputCellTypeProportionsPath"], "w") as out:
        json.dump(proportions, out, indent=1, separators=(",", ": "))
