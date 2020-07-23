'''Generate and save combined annotations or combined markers

Combination operates on two or more volumetric files with nrrd format.

Lexicon:
    * AIBS stands for 'Allen Institute for Brain Science'
    (https://alleninstitute.org/what-we-do/brain-science/)
'''
import json
from pathlib import Path
import logging
import click


from atlas_building_tools import cell_detection
from atlas_building_tools.app.utils import (
    log_args,
    EXISTING_DIR_PATH,
    EXISTING_FILE_PATH,
    set_verbose,
)

L = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    '''Run the cell detection CLI'''
    set_verbose(L, verbose)


@app.command()
@click.option(
    '--input-dir',
    type=EXISTING_DIR_PATH,
    required=True,
    help=('This directory contains a list of svg files to convert.'),
)
@click.option(
    '--remove-strokes',
    is_flag=True,
    help='Removes the strokes surrounding colored areas.',
    default=False,
)
@click.option(
    '--output-dir',
    type=str,
    required=True,
    help='Directory where the png files will be saved. It will be '
    'created if it doesn\'t exist already.',
)
@log_args(L)
def svg_to_png(
    input_dir, remove_strokes, output_dir,
):
    '''Convert svg files into png files.\n

    Strokes are optionally removed.
    '''
    filepaths = [Path.resolve(f) for f in Path(input_dir).glob('*.svg')]
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    for filepath in filepaths:
        cell_detection.svg_to_png(
            filepath,
            Path(output_dir, filepath.name.replace('.svg', '.png')),
            remove_strokes=remove_strokes,
        )


@app.command()
@click.option(
    '--input-dir',
    type=EXISTING_DIR_PATH,
    required=True,
    help=(
        'Path to the image directory. This directory contains a list of svg files to parse for '
        'fill color and AIBS structure_id extractions.'
    ),
)
@click.option(
    '--output-path',
    type=str,
    required=True,
    help='Path where the output json file will be saved.',
)
@log_args(L)
def extract_color_map(
    input_dir, output_path,
):
    '''Extract the mapping of colors to structure ids and save to file.\n

     Extract the structure_id's and the corresponding pairs from each AIBS svg annotation file in
     of input directory. These pairs are structured under the form of a dict
     {hexdecimal_color_code: structure_id} and saved into a json file.
     See http://help.brain-map.org/display/api/Downloading+and+Displaying+SVG
     for information on AIBS svg files and how to fetch them.\n
     Output example:\n
        {\n
            "#188064": 893,\n
            "#11ad83": 849,\n
            "#40a666": 810,\n
        }
    '''
    filepaths = [Path.resolve(f) for f in Path(input_dir).glob('*.svg')]
    color_map = {}
    for filepath in filepaths:
        color_map.update(cell_detection.extract_color_map(filepath))
    with open(output_path, 'w') as out:
        json.dump(color_map, out, indent=1, separators=(',', ': '))


@app.command()
@click.option(
    '--input-dir',
    type=EXISTING_DIR_PATH,
    required=True,
    help=(
        'Path to the image directoy. This directory contains a list of jpg and png files whose '
        ' identical base names are brain section identifiers. The jpg files are nissle images. '
        'The png files hold colored annotation.'
    ),
)
@click.option(
    '--color-map-path',
    type=EXISTING_FILE_PATH,
    required=True,
    help=(
        'The path to the json file mapping hexdecimal color codes to AIBS structure ids.'
    ),
)
@click.option(
    '--output-path',
    type=str,
    required=True,
    help='Path where the output json file will be saved. This file encloses a dictionary '
    'whose keys are region identifiers (a.k.a structure id) and whose values are the corresponding '
    'average soma radii. Note that the computation is possible only for a small subset '
    ' of regions for which both a nissle image (jpg) and an annotation image (svg) exist.',
)
@log_args(L)
def compute_average_soma_radius(
    input_dir, color_map_path, output_path,
):
    '''Compute average soma radii for different regions and save to file.\n

     For each region where somas can been detected, the function estimates
     the radii of somas and save the mean value over each region.
    '''
    filepaths = [Path.resolve(f) for f in Path(input_dir).glob('*.jpg')]
    with open(color_map_path, 'r') as file_:
        color_map = json.load(file_)
        soma_radius_dict = cell_detection.compute_average_soma_radius(
            color_map, filepaths, delta=16, max_radius=10, intensity_threshold=0.1
        )
        with open(output_path, 'w+') as out:
            json.dump(soma_radius_dict, out, indent=1, separators=(',', ': '))
