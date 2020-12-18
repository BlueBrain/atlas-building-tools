'''Generate and save cell densities

A density value is a float number corresponding to the number of cells in a voxel.
A density field is a 3D volumetric array assigning to each voxel a density value.

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

It is assumed throughout that such intensities depend "almost" linearly
on the cell density when restricted to a brain region, but we shall not give a precise meaning to
 the word "almost".
'''

import os
import json
from pathlib import Path
import logging
import click
import numpy as np

from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_building_tools.app.utils import (
    log_args,
    EXISTING_DIR_PATH,
    EXISTING_FILE_PATH,
    set_verbose,
)

from atlas_building_tools.densities.cell_density import compute_cell_density
from atlas_building_tools.densities.glia_densities import compute_glia_densities
from atlas_building_tools.densities.inhibitory_neuron_density import (
    compute_inhibitory_neuron_density,
)
from atlas_building_tools.densities.cell_counts import (
    extract_inhibitory_neurons_dataframe,
    glia_cell_counts,
    inhibitory_data,
)
from atlas_building_tools.densities.mtype_densities import DensityProfileCollection

L = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    '''Run the cell densities CLI'''
    set_verbose(L, verbose)


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the whole mouse brain annotation file.'),
)
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help='The path to the hierarchy file, i.e., AIBS 1.json.',
)
@click.option(
    '--nissl-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the AIBS Nissl stains nrrd file.'),
)
@click.option(
    '--output-path',
    type=str,
    required=True,
    help='Path where to write the output cell density nrrd file.',
)
@click.option(
    '--soma-radii',
    type=EXISTING_FILE_PATH,
    required=False,
    help='Optional path to the soma radii json file. If specified'
    ', the input nissl stain intensity is adjusted by taking regions soma radii into account.',
    default=None,
)
@log_args(L)
def cell_density(annotation_path, hierarchy_path, nissl_path, output_path, soma_radii):
    """Compute and save the overall mouse brain cell density.\n

    The input Nissl stain volume of AIBS is turned into an actual density field complying with
    the cell counts of several regions.

    Density is expressed as a number of cells per voxel.
    The output density field array is a float64 array of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation of the overall cell density is based on:\n
        * the Nissl stain intensity, which is supposed to represent the overall cell density, up to
            to region-dependent constant scaling factors.\n
        * cell counts from the scientific literature, which are used to determine a local \n
            linear dependency factor each regions where a cell count is available.\n
        * the optional soma radii, used to operate a correction.
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    region_map = RegionMap.load_json(hierarchy_path)
    nissl = VoxelData.load_nrrd(nissl_path)
    if soma_radii is not None:
        with open(soma_radii, 'r') as file_:
            soma_radii = json.load(file_)

    overall_cell_density = compute_cell_density(
        region_map, annotation.raw, nissl.raw, soma_radii
    )
    nissl.with_data(np.asarray(overall_cell_density, dtype=float)).save_nrrd(
        output_path
    )


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the whole mouse brain annotation file.'),
)
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help='The path to the hierarchy file, i.e., AIBS 1.json.',
)
@click.option(
    '--cell-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the overall cell density nrrd file.'),
)
@click.option(
    '--glia-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the unconstrained overall glia cell density nrrd file.'),
)
@click.option(
    '--astrocyte-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the unconstrained astrocyte density nrrd file.'),
)
@click.option(
    '--oligodendrocyte-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the unconstrained oligodendrocyte density nrrd file.'),
)
@click.option(
    '--microglia-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the unconstrained microglia density nrrd file.'),
)
@click.option(
    '--glia-proportions-path',
    type=EXISTING_FILE_PATH,
    help='Path to the json file containing the different proportions of each glia type.'
    'This file must hold a dictionary of the following form: '
    '{"astrocyte": <proportion>, "microglia": <proportion>, "oligodendrocyte": <proportion>,'
    ' "glia": 1.0}',
)
@click.option(
    '--output-dir',
    type=str,
    required=True,
    help='Path to the directory where to write the output cell density nrrd files.'
    ' It will be created if it doesn\'t exist already.',
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

    Density is expressed as a number of cells per voxel.
    The output density field arrays are float64 arrays of shape (W, H, D) where (W, H, D)
    is the shape of the input annotated volume.

    The computation is based on:\n
        * an estimate of the overall cell density\n
        * estimates of unconstrained densities for the different glia cell types\n
        * glia cell counts from the scientific literature\n

    The cell counts and the overall cell density are used to constrain the glia cell densities\n
    so that:\n
        * they do not exceed voxel-wise the overall cell density\n
        * density sums match the provided cell counts\n

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
    region_map = RegionMap.load_json(hierarchy_path)
    overall_cell_density = VoxelData.load_nrrd(cell_density_path)
    with open(glia_proportions_path, 'r') as file_:
        glia_proportions = json.load(file_)

    glia_densities = {
        'glia': VoxelData.load_nrrd(glia_density_path).raw,
        'astrocyte': VoxelData.load_nrrd(astrocyte_density_path).raw,
        'oligodendrocyte': VoxelData.load_nrrd(oligodendrocyte_density_path).raw,
        'microglia': VoxelData.load_nrrd(microglia_density_path).raw,
    }

    glia_densities = compute_glia_densities(
        region_map,
        annotation.raw,
        sum(glia_cell_counts().values()),
        glia_densities,
        overall_cell_density.raw,
        glia_proportions,
        copy=False,
    )

    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    neuron_density = overall_cell_density.raw - glia_densities['glia']
    annotation.with_data(np.asarray(neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, 'neuron_density.nrrd'))
    )
    for glia_type, density in glia_densities.items():
        annotation.with_data(np.asarray(density, dtype=float)).save_nrrd(
            str(Path(output_dir, f'{glia_type}_density.nrrd'))
        )


@app.command()
@click.option(
    '--annotation-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the whole mouse brain annotation file.'),
)
@click.option(
    '--hierarchy-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help='The path to the hierarchy file, i.e., AIBS 1.json.',
)
@click.option(
    '--gad1-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the GAD marker nrrd file.'),
)
@click.option(
    '--nrn1-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to Nrn1 marker nrrd file.'),
)
@click.option(
    '--neuron-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=('The path to the overall neuron cell density nrrd file.'),
)
@click.option(
    '--inhibitory-neuron-counts-path',
    type=EXISTING_FILE_PATH,
    required=False,
    default=Path(Path(__file__).parent, 'data', 'mmc1.xlsx'),
    help=(
        'The path to the excel document mmc1.xlsx of the suplementary materials of '
        '"Brain-wide Maps Reveal Stereotyped Cell-Type- Based Cortical Architecture '
        'and Subcortical Sexual Dimorphism" by Kim et al., 2017. '
        'https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc1.xlsx. '
        'Defaults to atlas_building_tools/app/data/mmc1.xlsx.'
    ),
)
@click.option(
    '--output-dir',
    type=str,
    required=True,
    help='Path to the directory where to write the output cell density nrrd files.'
    ' It will be created if it doesn\'t exist already.',
)
@log_args(L)
def inhibitory_neuron_densities(
    annotation_path,
    hierarchy_path,
    gad1_path,
    nrn1_path,
    neuron_density_path,
    inhibitory_neuron_counts_path,
    output_dir,
):  # pylint: disable=too-many-arguments
    """Compute and save the inhibitory and excitatory neuron densities.\n

    Density is expressed as a number of cells per voxel.
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

    The ouput densities are saved in the specified output directory under the following\n
    names:\n
        * inhibitory_neuron_density.nrrd \n
        * excitatory_neuron_density.nrrd \n
    """

    annotation = VoxelData.load_nrrd(annotation_path)
    region_map = RegionMap.load_json(hierarchy_path)
    neuron_density = VoxelData.load_nrrd(neuron_density_path).raw
    inhibitory_df = extract_inhibitory_neurons_dataframe(inhibitory_neuron_counts_path)
    inhibitory_neuron_density = compute_inhibitory_neuron_density(
        region_map,
        annotation.raw,
        VoxelData.load_nrrd(gad1_path).raw,
        VoxelData.load_nrrd(nrn1_path).raw,
        neuron_density,
        inhibitory_data=inhibitory_data(inhibitory_df),
    )

    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    annotation.with_data(np.asarray(inhibitory_neuron_density, dtype=float)).save_nrrd(
        str(Path(output_dir, 'inhibitory_neuron_density.nrrd'))
    )
    excitatory_density = neuron_density - inhibitory_neuron_density
    annotation.with_data(np.asarray(excitatory_density, dtype=float)).save_nrrd(
        str(Path(output_dir, 'excitatory_neuron_density.nrrd'))
    )


@app.command()
@click.option(
    '--excitatory-neuron-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help='Path to the excitatory neuron density nrrd file',
)
@click.option(
    '--inhibitory-neuron-density-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help='Path to the inhibitory neuron density nrrd file',
)
@click.option(
    '--placement-hints-config-path',
    type=EXISTING_FILE_PATH,
    help='Path to the placement hints config file (.yaml)',
)
@click.option(
    '--layer-slices-path',
    type=EXISTING_FILE_PATH,
    default=Path(Path(__file__).parent, 'data', 'mtypes', 'meta', 'layers.tsv'),
    help='Path to the layer slices file (.tsv).',
)
@click.option(
    '--mtype-to-profile-map-path',
    type=EXISTING_FILE_PATH,
    default=Path(Path(__file__).parent, 'data', 'mtypes', 'meta', 'mapping.tsv'),
    help='Path to the map which assigns a cell density profile to each mtype (.tsv)',
)
@click.option(
    '--density-profiles-dir',
    type=EXISTING_DIR_PATH,
    default=Path(Path(__file__).parent, 'data', 'mtypes'),
    help='Path to directory containing the cell density profiles',
)
@click.option(
    '--output-dir',
    required=True,
    help='Path to output directory. It will be created if it doesn\'t exist already.',
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

    L.info('Collecting density profiles ...')

    density_profile_collection = DensityProfileCollection.load(
        mtype_to_profile_map_path, layer_slices_path, density_profiles_dir
    )

    L.info('Density profile collection successfully instantiated.')

    density_profile_collection.create_mtype_densities(
        excitatory_neuron_density_path,
        inhibitory_neuron_density_path,
        placement_hints_config_path,
        output_dir,
    )
