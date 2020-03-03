''' Collection of tools for atlas building '''

import logging
import click

from atlas_building_tools.app import combine_annotations, cerebellum_direction_vectors
from atlas_building_tools.version import VERSION


@click.group('atlas-building-tools', help=__doc__.format(esc='\b'))
@click.option('-v', '--verbose', count=True, help='-v for INFO, -vv for DEBUG')
@click.version_option(VERSION)
def app(verbose=0):
    # pylint: disable=missing-docstring
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}[verbose]
    logging.basicConfig(level=level)


app.add_command(name='combine-annotations', cmd=combine_annotations.cmd)
app.add_command(name='cerebellum-direction-vectors', cmd=cerebellum_direction_vectors.cmd)


def main():
    '''main entry point'''
    app(obj={})  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
