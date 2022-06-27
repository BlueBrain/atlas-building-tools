"""Utilities for working with NRRD files"""
import logging
import sys

import click

import atlas_building_tools.utils
from atlas_building_tools.app.utils import EXISTING_FILE_PATH, log_args, set_verbose

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """utilities to work with NRRD files"""
    set_verbose(L, verbose)


@app.command()
@click.argument(
    "input-files",
    type=EXISTING_FILE_PATH,
    nargs=-1,
    required=True,
)
@click.option("--output", type=str, required=True, help="Output NRRD file")
@click.option(
    "--sentinel",
    type=str,
    required=True,
    help="Value for voxels that aren't to be copied (ex: 'nan', or 0)",
)
@log_args(L)
def merge(input_files, output, sentinel):
    """Merge multiple NRRD files together

    These are the NRRD files that will be merged to the output file
    Ones that come later override contents of earlier files
    They should all have the same offset, voxel size, and total shape.
    """
    try:
        ret = atlas_building_tools.utils.merge_nrrd(input_files, sentinel)
    except ValueError as exc:
        click.echo(click.style(str(exc), fg="red"))
        sys.exit(-1)

    ret.save_nrrd(output)
