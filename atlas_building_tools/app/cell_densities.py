"""Generate and save cell densities

A density value is a non-negative float number corresponding to the number of cells in mm^3.
A density field is a 3D volumetric array assigning to each voxel a density value, that is
the mean cell density within this voxel.

This script computes and saves the following cell densities under the form of density fields.

* overall cell density
* overall glia cell density and overall neuron density
* among glial cells:
    - astrocyte density
    - oligodendrocyte density
    - microglia density
* among neuron cells:
    - inhibitory neuron density
    - excitatory neuron density

Density estimates are based on datasets produced by in-situ hybridization experiments of the
 Allen Institute for Brain Science (AIBS). We used in particular AIBS genetic marker datasets and
  the Nissl volume of the Allen Mouse Brain Annotation Atlas.
Genetic marker stained intensity and Nissl stained intensity are assumed to be a good indicator
of the soma density in a population of interest.

It is assumed throughout that such intensities depend "almost" linearly on the cell density when
restricted to a brain region, but we shall not give a precise meaning to the word "almost".
"""

import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_building_tools.app.utils import (
    EXISTING_DIR_PATH,
    EXISTING_FILE_PATH,
    assert_properties,
    log_args,
    set_verbose,
)
from atlas_building_tools.densities.cell_counts import (
    extract_inhibitory_neurons_dataframe,
    glia_cell_counts,
    inhibitory_data,
)
from atlas_building_tools.densities.cell_density import compute_cell_density
from atlas_building_tools.densities.excel_reader import (
    read_homogenous_neuron_type_regions,
    read_measurements,
)
from atlas_building_tools.densities.glia_densities import compute_glia_densities
from atlas_building_tools.densities.inhibitory_neuron_density import (
    compute_inhibitory_neuron_density,
)
from atlas_building_tools.densities.measurement_to_density import (
    measurement_to_average_density,
    remove_non_density_measurements,
)
from atlas_building_tools.densities.mtype_densities import DensityProfileCollection

DATA_PATH = Path(Path(__file__).parent, "data")

L = logging.getLogger(__name__)


def _get_voxel_volume_in_mm3(voxel_data: "VoxelData") -> float:
    """
    Returns the voxel volume of `voxel_data` in mm^3.

    Note: the voxel_dimensions of `voxel_data` are assumed to be
    expressed in um (micron = 1e-6 m).

    Args:
        voxel_data: VoxelData object whose voxel volume will be computed.

    Returns:
        The volume in mm^3 of a `voxel_data` voxel.
    """
    return voxel_data.voxel_volume / 1e9


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the cell densities CLI"""
    set_verbose(L, verbose)


@app.command()
@click.option(
    "--annotation-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the whole mouse brain annotation file."),
)
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="The path to the hierarchy file, i.e., AIBS 1.json.",
)
@click.option(
    "--nissl-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the AIBS Nissl stains nrrd file."),
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path where to write the output cell density nrrd file."
    "A voxel value is a number of cells per mm^3",
)
@click.option(
    "--soma-radii",
    type=EXISTING_FILE_PATH,
    required=False,
    help="Optional path to the soma radii json file. If specified"
    ", the input nissl stain intensity is adjusted by taking regions soma radii into account."
    " See cell_detection module.",
    default=None,
)
@log_args(L)
def cell_density(annotation_path, hierarchy_path, nissl_path, output_path, soma_radii):
    """Compute and save the overall mouse brain cell density.\n

    The input Nissl stain volume of AIBS is turned into an actual density field complying with
    the cell counts of several regions.

    Density is expressed as a number of cells per mm^3.
    The output density field array is a float64 array of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation of the overall cell density is based on:\n
        * the Nissl stain intensity, which is supposed to represent the overall cell density, up to
            to region-dependent constant scaling factors.\n
        * cell counts from the scientific literature, which are used to determine a local \n
            linear dependency factor for each region where a cell count is available.\n
        * the optional soma radii, used to operate a correction.
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    nissl = VoxelData.load_nrrd(nissl_path)

    # Check nrrd metadata consistency
    assert_properties([annotation, nissl])

    region_map = RegionMap.load_json(hierarchy_path)
    if soma_radii is not None:
        with open(soma_radii, "r") as file_:
            soma_radii = json.load(file_)

    overall_cell_density = compute_cell_density(
        region_map,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        nissl.raw,
        soma_radii,
    )
    nissl.with_data(np.asarray(overall_cell_density, dtype=float)).save_nrrd(output_path)


@app.command()
@click.option(
    "--annotation-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the whole mouse brain annotation file."),
)
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="The path to the hierarchy file, i.e., AIBS 1.json.",
)
@click.option(
    "--cell-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the overall cell density nrrd file."),
)
@click.option(
    "--glia-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained overall glia cell density nrrd file."),
)
@click.option(
    "--astrocyte-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained astrocyte density nrrd file."),
)
@click.option(
    "--oligodendrocyte-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained oligodendrocyte density nrrd file."),
)
@click.option(
    "--microglia-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the unconstrained microglia density nrrd file."),
)
@click.option(
    "--glia-proportions-path",
    type=EXISTING_FILE_PATH,
    help="Path to the json file containing the different proportions of each glia type."
    "This file must hold a dictionary of the following form: "
    '{"astrocyte": <proportion>, "microglia": <proportion>, "oligodendrocyte": <proportion>,'
    ' "glia": 1.0}',
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Path to the directory where to write the output cell density nrrd files."
    " It will be created if it doesn't exist already.",
)
@log_args(L)
def glia_cell_densities(
    annotation_path,
    hierarchy_path,
    cell_density_path,
    glia_density_path,
    astrocyte_density_path,
    oligodendrocyte_density_path,
    microglia_density_path,
    glia_proportions_path,
    output_dir,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Compute and save the glia cell densities.\n

    Density is expressed as a number of cells per mm^3.
    The output density field arrays are float64 arrays of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation is based on:\n
        * an estimate of the overall cell density\n
        * estimates of unconstrained densities for the different glia cell types\n
        * glia cell counts from the scientific literature\n

    The cell counts and the overall cell density are used to constrain the glia cell densities\n
    so that:\n
        * they do not exceed voxel-wise the overall cell density\n
        * the density sums multiplied by the voxel volume match the provided cell counts\n

    An optimization process is responsible for enforcing these constraints while keeping\n
    the output densities as close as possible to the unconstrained input densities.\n
    Note: optimization is not fully implemented and the current process only returns a
    feasible point.

    The ouput glia densities are saved in the specified output directory under the following\n
    names:\n
        * glia_density.nrrd (overall glia density) \n
        * astrocyte_density.nrrd \n
        * oligodendrocyte_density.nrrd \n
        * microglia_density.nrrd \n

    In addition, the overall neuron cell density is inferred from the overall cell density and
    the glia cell density and saved in the same directory under the name:\n
        * neuron_density.
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    overall_cell_density = VoxelData.load_nrrd(cell_density_path)

    glia_densities = {
        "glia": VoxelData.load_nrrd(glia_density_path),
        "astrocyte": VoxelData.load_nrrd(astrocyte_density_path),
        "oligodendrocyte": VoxelData.load_nrrd(oligodendrocyte_density_path),
        "microglia": VoxelData.load_nrrd(microglia_density_path),
    }

    atlases = list(glia_densities.values())
    atlases += [annotation, overall_cell_density]
    assert_properties(atlases)

    region_map = RegionMap.load_json(hierarchy_path)
    with open(glia_proportions_path, "r") as file_:
        glia_proportions = json.load(file_)

    glia_densities = {
        glia_cell_type: voxel_data.raw for (glia_cell_type, voxel_data) in glia_densities.items()
    }

    glia_densities = compute_glia_densities(
        region_map,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        sum(glia_cell_counts().values()),
        glia_densities,
        overall_cell_density.raw,
        glia_proportions,
        copy=False,
    )

    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    neuron_density = overall_cell_density.raw - glia_densities["glia"]
    annotation.with_data(np.asarray(neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, "neuron_density.nrrd"))
    )
    for glia_type, density in glia_densities.items():
        annotation.with_data(np.asarray(density, dtype=float)).save_nrrd(
            str(Path(output_dir, f"{glia_type}_density.nrrd"))
        )


@app.command()
@click.option(
    "--annotation-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the whole mouse brain annotation file."),
)
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="The path to the hierarchy file, i.e., AIBS 1.json.",
)
@click.option(
    "--gad1-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the GAD marker nrrd file."),
)
@click.option(
    "--nrn1-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to Nrn1 marker nrrd file."),
)
@click.option(
    "--neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the overall neuron density nrrd file."),
)
@click.option(
    "--inhibitory-neuron-counts-path",
    type=EXISTING_FILE_PATH,
    required=False,
    default=Path(Path(__file__).parent, "data", "measurements", "mmc1.xlsx"),
    help=(
        "The path to the excel document mmc1.xlsx of the suplementary materials of "
        '"Brain-wide Maps Reveal Stereotyped Cell-Type- Based Cortical Architecture '
        'and Subcortical Sexual Dimorphism" by Kim et al., 2017. '
        "https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc1.xlsx. "
        "Defaults to atlas_building_tools/app/data/measurements/mmc1.xlsx."
    ),
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Path to the directory where to write the output cell density nrrd files."
    " It will be created if it doesn't exist already.",
)
@log_args(L)
def inhibitory_and_excitatory_neuron_densities(
    annotation_path,
    hierarchy_path,
    gad1_path,
    nrn1_path,
    neuron_density_path,
    inhibitory_neuron_counts_path,
    output_dir,
):  # pylint: disable=too-many-arguments
    """Compute and save the inhibitory and excitatory neuron densities.\n

    Density is expressed as a number of cells per mm^3.
    The output density field arrays are float64 arrays of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation is based on:\n
        * an estimate of the overall neuron density\n
        * estimates of unconstrained inhibitory and excitatory neuron densities provided by
        the GAD1 and Nrn1 markers intensities respectively.

    The overall neuron density and region-specific neuron counts from the scientific literature are
     used to constrain the inhibitory and excitatory neuron densities so that:\n
        * they do not exceed voxel-wise the overall neuron cell density\n
        * the ratio (inhibitory neuron count / excitatory neuron count) matches a prescribed value
        wherever it is constrained.

    An optimization process is responsible for enforcing these constraints while keeping\n
    the output densities as close as possible to the unconstrained input densities.\n

    Note: optimization is not fully implemented and the current process only returns a
    feasible point.

    The ouput densities are saved in the specified output directory under the following\n
    names:\n
        * inhibitory_neuron_density.nrrd \n
        * excitatory_neuron_density.nrrd \n
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    neuron_density = VoxelData.load_nrrd(neuron_density_path)

    assert_properties([annotation, neuron_density])

    region_map = RegionMap.load_json(hierarchy_path)
    inhibitory_df = extract_inhibitory_neurons_dataframe(inhibitory_neuron_counts_path)
    inhibitory_neuron_density = compute_inhibitory_neuron_density(
        region_map,
        annotation.raw,
        _get_voxel_volume_in_mm3(annotation),
        VoxelData.load_nrrd(gad1_path).raw,
        VoxelData.load_nrrd(nrn1_path).raw,
        neuron_density.raw,
        inhibitory_data=inhibitory_data(inhibitory_df),
    )

    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    annotation.with_data(np.asarray(inhibitory_neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, "inhibitory_neuron_density.nrrd"))
    )
    excitatory_neuron_density = neuron_density.raw - inhibitory_neuron_density
    annotation.with_data(np.asarray(excitatory_neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, "excitatory_neuron_density.nrrd"))
    )


@app.command()
@click.option(
    "--excitatory-neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to the excitatory neuron density nrrd file",
)
@click.option(
    "--inhibitory-neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="Path to the inhibitory neuron density nrrd file",
)
@click.option(
    "--placement-hints-config-path",
    type=EXISTING_FILE_PATH,
    help="Path to the placement hints config file (.yaml)",
)
@click.option(
    "--layer-slices-path",
    type=EXISTING_FILE_PATH,
    default=Path(Path(__file__).parent, "data", "mtypes", "meta", "layers.tsv"),
    help="Path to the layer slices file (.tsv).",
)
@click.option(
    "--mtype-to-profile-map-path",
    type=EXISTING_FILE_PATH,
    default=Path(Path(__file__).parent, "data", "mtypes", "meta", "mapping.tsv"),
    help="Path to the map which assigns a cell density profile to each mtype (.tsv)",
)
@click.option(
    "--density-profiles-dir",
    type=EXISTING_DIR_PATH,
    default=Path(Path(__file__).parent, "data", "mtypes"),
    help="Path to directory containing the cell density profiles",
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to output directory. It will be created if it doesn't exist already.",
)
@log_args(L)
def mtype_densities(
    excitatory_neuron_density_path,
    inhibitory_neuron_density_path,
    placement_hints_config_path,
    mtype_to_profile_map_path,
    layer_slices_path,
    density_profiles_dir,
    output_dir,
):  # pylint: disable=too-many-arguments
    """
    Create neuron density nrrd files for the mtypes listed in `mtype_to_profile_map_path`.

    Somatosensory cortex layers are subdivided into slices (a.k.a bins). Each mtype in
    `mtype_to_profile_map_path` (defaults to mapping.tsv) is assigned a density profile, that is,
    the list of the numbers of neurons with this mtype in each slice. From this, a relative density
    profile is derived, i.e. the list of the neuron proportions in each slice.
    Using the overall inhibitory neuron and excitatory neuron densities together with the relative
    density profiles, we obtain a volumetric neuron density for each mtype under the form of nrrd
    files.

    Placement hints are used to divide layers into slices, i.e., sublayers of equal thickness along
    the cortical axis. The number of slices per layer is specified in `layer_slices_path` (defaults
    to layers.tsv).

    Neuron densities are expressed in number of neurons per voxel.

    The density profile datasets were obtained in
    "A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex"
    by D. Keller et al., 2019., https://www.frontiersin.org/articles/10.3389/fnana.2019.00078/full.
    These datasets and associated metadata files can be found in
    atlas_building_tools/app/data/mtypes. This command uses the latter files used by default.
    """

    L.info("Collecting density profiles ...")

    density_profile_collection = DensityProfileCollection.load(
        mtype_to_profile_map_path, layer_slices_path, density_profiles_dir
    )

    L.info("Density profile collection successfully instantiated.")

    density_profile_collection.create_mtype_densities(
        excitatory_neuron_density_path,
        inhibitory_neuron_density_path,
        placement_hints_config_path,
        output_dir,
    )


@app.command()
@click.option(
    "--measurements-output-path",
    required=True,
    help="Path where the density-related measurement series will be written. CSV file whose columns"
    " are described in the main help section.",
)
@click.option(
    "--homogenous-regions-output-path",
    required=True,
    help="Path where the list of AIBS brain regions with homogenous neuron type (e.g., inhibitory"
    ' or excitatory) will be saved. CSV file with 2 columns: "brain_region" and "cell_type".',
)
@log_args(L)
def compile_measurements(
    measurements_output_path,
    homogenous_regions_output_path,
):
    """
    Compile the cell density related measurements of mmc3.xlsx and gaba_papers.xsls into a CSV file.

    In addition to various measurements found in the scientific literature, a list of AIBS mouse
    brain regions with homogenous neuron type is saved to `homogenous_regions_output_path`.

    Two input excel files containing measurements are handled:\n
    * mm3c.xls from the supplementary materials of
        'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and Subcortical
        Sexual Dimorphism' by Kim et al., 2017.
        https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc3.xlsx\n

    * atlas_building_tools/app/data/measurements/gaba_papers.xlsx, a compilation of measurements
        from the scientifc literature made by Rodarie Dimitri (BBP).\n

    This command extracts measurements from the above two files and gathers them into a unique
    CSV file with the following columns:\n

    * brain_region (str), a mouse brain region name, not necessarily compliant
      with AIBS 1.json file. Thus some filtering must be done when working with AIBS\n
      annotated files.\n
    * cell type (str, e.g, 'PV+' for cells reacting to parvalbumin, 'inhibitory neuron' for\n
    non-specific inhibitory neuron)\n
    * measurement (float)\n
    * standard_deviation (non-negative float)\n
    * measurement_type (str), see measurement types below\n
    * measurement_unit (str), see measurement units below\n
    * comment (str), a comment on how the measurement has been obtained\n
    * source_title (str), the title of the article where the measurement can be exracted\n
    * specimen_age (str, e.g., '8 week old', 'P56', '3 month old'), age of the mice used to obtain
      the measurement\n

    The different measurement types are, for a given brain region R and a given cell type T:\n
    * 'cell density', number of cells of type T per mm^3 in R\n
    * 'cell count', number of cells of type T in R\n
    * 'neuron proportion', number of cells of type T / number of neurons in R
    (a cell of type T is assumed to be a neuron, e.g., T = GAD67+)\n
    * 'cell proportion', number of cells of type T / number of cells in R\n
    * 'cell count per slice', number of cells of type T per slice of R\n

    Measurement units:
    * 'cell density': 'number of cells per mm^3'\n
    * 'neuron proportion': None (empty)\n
    * 'cell proportion': None (empty)\n
    * 'cell count per slice': e.g, number of cells per 50-micrometer-thick slice\n

    See atlas_building_tools/densities/excel_reader.py for more information.

    Note: This function should be deprecated once its output has been stored permanently as the
    the unique source of density-related measurements for the AIBS mouse brain. New measurements
    should be added to the stored file (Nexus).
    """

    L.info("Loading excel files ...")
    region_map = RegionMap.load_json(Path(DATA_PATH, "1.json"))  # Unmodified AIBS 1.json
    measurements = read_measurements(
        region_map,
        Path(DATA_PATH, "measurements", "mmc3.xlsx"),
        Path(DATA_PATH, "measurements", "gaba_papers.xlsx"),
        # The next measurement file has been obtained after manual extraction
        # of non-density measurements from the worksheets PV-SST-VIP and 'GAD67 densities'
        # of gaba_papers.xlsx.
        Path(DATA_PATH, "measurements", "non_density_measurements.csv"),
    )
    homogenous_regions = read_homogenous_neuron_type_regions(
        Path(DATA_PATH, "measurements", "gaba_papers.xlsx")
    )
    L.info("Saving to CSV files ...")
    measurements.to_csv(measurements_output_path, index=False)
    homogenous_regions.to_csv(homogenous_regions_output_path, index=False)


@app.command()
@click.option(
    "--annotation-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the whole mouse brain annotation file."),
)
@click.option(
    "--hierarchy-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help="The path to the hierarchy file, i.e., AIBS 1.json.",
)
@click.option(
    "--cell-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the to the overall cell density nrrd file."),
)
@click.option(
    "--neuron-density-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("The path to the overall neuron density nrrd file."),
)
@click.option(
    "--measurements-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        "The path to measurements.csv, the compilation of cell density "
        "related measurements of Dimitri Rodarie (BBP)."
    ),
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path where to write the output average cell densities (.csv file), that is, a data frame"
    " of the same format as the input measurements file (see --measurements-path )but comprising "
    "only measurements of type `cell density`.",
)
@log_args(L)
def measurements_to_average_densities(
    annotation_path,
    hierarchy_path,
    cell_density_path,
    neuron_density_path,
    measurements_path,
    output_path,
):  # pylint: disable=too-many-arguments
    """Compute and save average cell densities based on measurements and AIBS region volumes.\n

    Measurements from Dimitri Rodarie's compilation, together with volumes from the AIBS mouse brain
    (`annotation`) and precomputed volumetric cell densities (`cell_density_path` and
    `neuron_density_path`) are used to compute average cell densities in every AIBS region where
    sufficient information is available.

    The different cell types (e.g., PV+, SST+, VIP+ or overall inhibitory neurons) and
    brain regions under consideration are prescribed by the input measurements.

    Measurements can be cell densities in number of cells per mm^3 for instance.
    If several cell density measurements are available for the same region, the output dataframe
    records the average of these measurements.

    Measurements can also be cell counts, in which case the AIBS brain model volumes
    (`annotation_path`) are used in addition to compute average cell densities.

    For measurements such as cell proportions, neuron proportions or cell counts per slice, the
    brain-wide volumetric cell densities (`cell_density_path` or `neuron_density_path`) are used to
    compute average cell densities.

    If several combinations of measurements yield several average cell densities for the same
    region, then the output data frame records the average of these measurements.

    The ouput average cell densities are saved in under the CSV format as a dataframe with the same
    columns as the input data frame specified via `--measurements-path`.
    See :mod:`atlas_building_tools.app.densities.compile_measurements`.

    All output measurements are average cell densities of various cell types over AIBS brain
    regions expressed in number of cells per mm^3.
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    overall_cell_density = VoxelData.load_nrrd(cell_density_path)
    neuron_density = VoxelData.load_nrrd(neuron_density_path)

    assert_properties([annotation, overall_cell_density, neuron_density])

    region_map = RegionMap.load_json(hierarchy_path)
    measurements_df = pd.read_csv(measurements_path)
    average_cell_densities_df = measurement_to_average_density(
        region_map,
        annotation.raw,
        annotation.voxel_dimensions,
        _get_voxel_volume_in_mm3(annotation),
        overall_cell_density.raw,
        neuron_density.raw,
        measurements_df,
    )
    remove_non_density_measurements(average_cell_densities_df)
    average_cell_densities_df.to_csv(
        output_path,
        index=False,
    )
