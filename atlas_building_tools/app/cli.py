''' Collection of tools for atlas building '''

import logging
import click

from atlas_building_tools.version import VERSION

from atlas_building_tools.app import (
    cell_densities,
    cell_detection,
    combination,
    direction_vectors,
    orientation_field,
    placement_hints,
    region_splitter,
)


def main():
    ''' Collection of tools for atlas building '''
    logging.basicConfig(level=logging.INFO)
    app = click.Group(
        'atlas_building_tools',
        {
            'cell-densities': cell_densities.app,
            'cell-detection': cell_detection.app,
            'combination': combination.app,
            'direction-vectors': direction_vectors.app,
            'orientation-field': orientation_field.cmd,
            'placement-hints': placement_hints.app,
            'region-splitter': region_splitter.app,
        },
    )
    app = click.version_option(VERSION)(app)
    app()
