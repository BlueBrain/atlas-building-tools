"""The atlas-building-tools command line launcher"""

import logging

import click

from atlas_building_tools.app import (
    cell_densities,
    combination,
    direction_vectors,
    flatmap,
    orientation_field,
    placement_hints,
    region_splitter,
)
from atlas_building_tools.version import VERSION

L = logging.getLogger(__name__)

try:
    import cairosvg

except ImportError:
    cairosvg = None
    L.info("cairosvg not found. Disabling cell-detection CLI.")

if cairosvg:
    # pylint: disable=ungrouped-imports
    from atlas_building_tools.app import cell_detection


def cli():
    """The CLI entry point"""
    logging.basicConfig(level=logging.INFO)
    group = {
        "cell-densities": cell_densities.app,
        "combination": combination.app,
        "direction-vectors": direction_vectors.app,
        "flatmap": flatmap.app,
        "orientation-field": orientation_field.cmd,
        "placement-hints": placement_hints.app,
        "region-splitter": region_splitter.app,
    }
    if cairosvg:
        group["cell-detection"] = cell_detection.app

    app = click.Group("atlas_building_tools", group)
    app = click.version_option(VERSION)(app)
    app()
