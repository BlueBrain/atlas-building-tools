"""Generate and save the volumetric cell densities of the BBP mtypes

A BBP mtype is a morphological type, i.e., a BBP type string such as "NGC-SA", "CHC" or "DLAC"
for instance.
A density value is a non-negative float number corresponding to the number of cells in mm^3.
A density field, a.k.a volumetric density, is a 3D volumetric array assigning to each voxel
a density value, that is the mean cell density within this voxel.

The commands of this module create a density field for each mtype listed either in

- `app/data/mtypes/density_profiles/mapping.tsv`, or
- `app/data/mtypes/probability_map/probability_map.csv`

Volumetric density nrrd files are created for each mtype listed in either `mapping.tsv` or
`probability_map.csv`.

This module re-use the overall excitatory and inhibitory neuron densities
computed in mod:`app/cell_densities` in the first case.

In the second case, it re-uses the
computation of the densities of the neurons reacting to PV, SST, VIP and GAD67,
see also mod:`app/cell_densities`.

Note that excitatory mtypes are handled in the first but not in the second case.
"""
import json
import logging
import re

import click
import numpy as np
import pandas as pd
import yaml  # type: ignore
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_building_tools.app.utils import (
    ABT_PATH,
    DATA_PATH,
    EXISTING_FILE_PATH,
    assert_meta_properties,
    common_atlas_options,
    log_args,
    set_verbose,
)
from atlas_building_tools.densities.mtype_densities_from_map import check_probability_map_sanity
from atlas_building_tools.densities.mtype_densities_from_map import (
    create_from_probability_map as create_from_map,
)
from atlas_building_tools.densities.mtype_densities_from_profiles import DensityProfileCollection
from atlas_building_tools.exceptions import AtlasBuildingToolsError
from atlas_building_tools.utils import assert_metadata_content

MTYPES_PROFILES_REL_PATH = (DATA_PATH / "mtypes" / "density_profiles").relative_to(ABT_PATH)
MTYPES_PROBABILITY_MAP_REL_PATH = (DATA_PATH / "mtypes" / "probability_map").relative_to(ABT_PATH)
METADATA_PATH = DATA_PATH / "data" / "metadata"
METADATA_REL_PATH = METADATA_PATH.relative_to(ABT_PATH)

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the mtype densities CLI"""
    set_verbose(L, verbose)


@app.command()
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'isocortex_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "isocortex_metadata.json"),
)
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the mouse isocortex direction vectors file, e.g., `direction_vectors.nrrd`."),
)
@click.option(
    "--mtypes-config-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to the yaml configuration file. "
    f"See `{str(MTYPES_PROFILES_REL_PATH / 'README.rst')}` for an example.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to output directory. It will be created if it doesn't exist already.",
)
@log_args(L)
def create_from_profile(
    annotation_path,
    hierarchy_path,
    metadata_path,
    direction_vectors_path,
    mtypes_config_path,
    output_dir,
):  # pylint: disable=too-many-locals
    """
    Create neuron density nrrd files for the mtypes listed in the mapping tsv file.

    Somatosensory cortex layers are subdivided into slices (a.k.a bins). Each mtype in
    the mapping tsv file (see configuration file description) is assigned a density profile,
    that is, the list of the numbers of neurons with this mtype in each slice. From this, a
    relative density profile is derived, i.e. the list of the neuron proportions in each slice.
    Using the overall inhibitory neuron and excitatory neuron densities together with the relative
    density profiles, we obtain a volumetric neuron density for each mtype under the form of nrrd
    files.

    The streamlines of the direction vectors filed are used to divide layers into slices, i.e.,
    sublayers of equal thickness along the cortical axis. The number of slices per layer is
    specified by the field layerSlicesPath of the configuration file (defaults to `layers.tsv`).

    Neuron densities are expressed in number of neurons per voxel.

    The density profile datasets were obtained in
    "A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex"
    <https://www.frontiersin.org/articles/10.3389/fnana.2019.00078/full>.
    These datasets and associated metadata files can be found in
    :mod:`atlas_building_tools/app/data/mtypes/density_profiles`.
    """

    L.info("Collecting density profiles ...")

    config = yaml.load(open(mtypes_config_path), Loader=yaml.FullLoader)

    density_profile_collection = DensityProfileCollection.load(
        config["mtypeToProfileMapPath"],
        config["layerSlicesPath"],
        config["densityProfilesDirPath"],
    )

    L.info("Density profile collection successfully instantiated.")
    with open(metadata_path, "r") as file_:
        metadata = json.load(file_)
    region_map = RegionMap.load_json(hierarchy_path)

    annotation = VoxelData.load_nrrd(annotation_path)
    direction_vectors = VoxelData.load_nrrd(direction_vectors_path)
    voxeldata = [annotation, direction_vectors]

    inhibitory_neuron_density = None
    excitatory_neuron_density = None

    if "inhibitoryNeuronDensityPath" in config:
        inhibitory_neuron_density = VoxelData.load_nrrd(config["inhibitoryNeuronDensityPath"])
        voxeldata.append(inhibitory_neuron_density)
    if "excitatoryNeuronDensityPath" in config:
        excitatory_neuron_density = VoxelData.load_nrrd(config["excitatoryNeuronDensityPath"])
        voxeldata.append(excitatory_neuron_density)

    if inhibitory_neuron_density is None and inhibitory_neuron_density is None:
        raise AtlasBuildingToolsError(
            "No neuron density files were provided. Expected: excitatory neuron density, or"
            "inhibitory neuron density or both."
        )
    # Check metadata consistency
    assert_meta_properties(voxeldata)

    density_profile_collection.create_mtype_densities(
        annotation,
        region_map,
        metadata,
        np.asarray(direction_vectors.raw, dtype=np.float32),
        output_dir,
        excitatory_neuron_density,
        inhibitory_neuron_density,
    )


def _check_config_sanity(config: dict) -> None:
    """
    Check if `config` has the expected keys.

    Raises otherwise.

    Args:
        config: the dict to be checked.
    Raises: AtlasBuildingTools error on failure.
    """
    diff = {"probabilityMapPath", "molecularTypeDensityPaths"} - set(config.keys())
    if diff:
        raise AtlasBuildingToolsError(
            f"The following keys are missing from the configuration file: {list(diff)}"
        )


def standardize_probability_map(probability_map: "pd.DataFrame") -> "pd.DataFrame":
    """
    Standardize the labels of the rows and the columns of `probability_map` and
    remove unused rows.

    Output labels are all lower case.
    The underscore is the only delimiter used in an output label.
    The layer names refered to by output labels are:
        "layer_1," "layer_23", "layer_4", "layer_5" and "layer_6".
    Rows whose labels contain "VIP" or "6b" are removed.

    Row example: "L2/3 Pvalb-IRES-Cre" -> "layer_23_pv"
    Column example: "NGC-SA" -> "ngc_sa"

    Args:
        probability_map: probability_map:
            data frame whose rows are labeled by molecular types and layers (e.g.,
            "L6a Htr3a-Cre_NO152", "L2/3 Pvalb-IRES-Cre", "L4 Htr3a-Cre_NO152") and whose columns
            are labeled by mtypes (mtypes = morphological types, e.g., "NGC-SA", "ChC", "DLAC").

    Returns: a data frame complying with all the above constraints.
    """
    probability_map = probability_map.copy()

    # The rows referring to layer 6b or to the Vip molecular marker are not used by the algorithm
    # computing the mtype densities.
    mask = [("Vip" in row_label) or ("6b" in row_label) for row_label in probability_map.index]
    indices = probability_map.index[mask]
    probability_map.drop(indices, inplace=True)

    def standardize_row_label(row_label: str):
        """
        Lowercase labels and use explicit layer names.
        """
        splitting = re.split("-|_| ", row_label)[:-2]  # remove unused Creline information
        splitting[0] = splitting[0].replace("/", "")
        splitting[0] = splitting[0].replace("6a", "6")
        splitting[0] = splitting[0].replace("L", "layer_")
        splitting[1] = splitting[1].replace("Pvalb", "pv")
        # Although Gad2 = Gad65, see e.g. https://www.genecards.org/cgi-bin/carddisp.pl?gene=GAD2
        # Gad1 = Gad67 is taken as an acceptable substitute for density estimates.
        splitting[1] = splitting[1].replace("Gad2", "gad67")

        return "_".join(splitting).lower()

    def standardize_column_label(col_label: str):
        """
        Lowercase labels and use underscore as delimiter for composed
        molecular types such as NGC-DA or NGC-SA.

        Example: "NGC-SA" -> "ngc_sa"
        """
        col_label = col_label.replace("-", "_")

        return col_label.lower()

    bbp_mtypes_map = {"DLAC": "LAC", "SLAC": "SAC"}
    probability_map.rename(bbp_mtypes_map, axis="columns", inplace=True)
    probability_map.rename(standardize_column_label, axis="columns", inplace=True)
    probability_map.rename(standardize_row_label, axis="rows", inplace=True)

    return probability_map


@app.command()
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'isocortex_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "isocortex_metadata.json"),
)
@click.option(
    "--mtypes-config-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to the yaml configuration file. "
    f"See `{str(MTYPES_PROBABILITY_MAP_REL_PATH / 'README.rst')}` for an example.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to output directory. It will be created if it doesn't exist already.",
)
@log_args(L)
def create_from_probability_map(
    annotation_path,
    hierarchy_path,
    metadata_path,
    mtypes_config_path,
    output_dir,
):  # pylint: disable=too-many-locals
    """
    Create neuron density nrrd files for the mtypes listed in the probability mapping csv file.

    Neuron densities are expressed in number of neurons per voxel.

    The probability mapping was obtained in PUBLICATION by Y. Roussel et al.
    It is a mapping between BBP mtypes and the molecular types PV, SST, VIP, HTR3A and GAD67,
    (these are molecular markers of inhibitory neurons).
    It can be found in
    :mod:`atlas_building_tools/app/data/mtypes/probability_map`.

    Note: this command does not generate volumetric density files for excitatory neurons.
    """

    L.info("Loading configuration file ...")
    config = yaml.load(open(mtypes_config_path), Loader=yaml.FullLoader)
    _check_config_sanity(config)

    L.info("Loading probability mapping ...")
    probability_map = pd.read_csv(config["probabilityMapPath"])
    if "molecular_type" in probability_map.columns:
        probability_map.set_index("molecular_type", inplace=True)
    else:
        probability_map.set_index(probability_map.columns[0], inplace=True)

    # Remove useless lines, use lower case and "standardized" explicit label names
    probability_map = standardize_probability_map(probability_map)
    check_probability_map_sanity(probability_map)

    L.info("Loading brain region metadata ...")
    with open(metadata_path, "r") as file_:
        metadata = json.load(file_)
        assert_metadata_content(metadata)

    L.info("Loading hierarchy json file ...")
    region_map = RegionMap.load_json(hierarchy_path)

    L.info("Loading annotation nrrd file ...")
    annotation = VoxelData.load_nrrd(annotation_path)

    L.info("Loading volumetric densities of molecular types ...")
    molecular_type_densities = {
        molecular_type: VoxelData.load_nrrd(density_path)
        for (molecular_type, density_path) in config["molecularTypeDensityPaths"].items()
    }

    # Check metadata consistency
    voxeldata = [annotation] + list(molecular_type_densities.values())
    assert_meta_properties(voxeldata)

    L.info("Creating volumetric densities of mtypes specified in probability map ...")
    create_from_map(
        annotation,
        region_map,
        metadata,
        {
            molecular_type: density.raw
            for (molecular_type, density) in molecular_type_densities.items()
        },
        probability_map,
        output_dir,
    )
