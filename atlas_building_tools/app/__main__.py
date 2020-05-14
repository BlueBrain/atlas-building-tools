''' Collection of tools for atlas building '''

import logging
import click

from atlas_building_tools.version import VERSION

from atlas_building_tools.app import (
    combine_annotations,
    direction_vectors,
    orientation_field,
)


def main():
    ''' Collection of tools for atlas building '''
    logging.basicConfig(level=logging.INFO)
    app = click.Group(
        'atlas_building_tools',
        {
            'annotations-combinator': combine_annotations.cmd,
            'direction-vectors': direction_vectors.app,
            'orientation-field': orientation_field.cmd,
        },
    )
    app = click.version_option(VERSION)(app)
    app()
