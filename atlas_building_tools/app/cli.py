"""The atlas-building-tools command line launcher"""

import logging

import atlas_densities
import atlas_direction_vectors
import atlas_placement_hints
import atlas_splitter
import click
from atlas_densities.app import cell_densities, combination, mtype_densities
from atlas_direction_vectors.app import direction_vectors, orientation_field
from atlas_placement_hints.app import placement_hints
from atlas_splitter.app import layer_splitter

from atlas_building_tools.app import flatmap, nrrd
from atlas_building_tools.version import VERSION

L = logging.getLogger(__name__)

try:
    import cairosvg
except ImportError:
    cairosvg = None
else:
    from atlas_building_tools.app import cell_detection  # pylint: disable=ungrouped-imports


def cli():
    """The main CLI entry point"""
    logging.basicConfig(level=logging.INFO)
    group = {
        "cell-densities": cell_densities.app,
        "combination": combination.app,
        "direction-vectors": direction_vectors.app,
        "flatmap": flatmap.app,
        "mtype-densities": mtype_densities.app,
        "nrrd": nrrd.app,
        "orientation-field": orientation_field.cmd,
        "placement-hints": placement_hints.app,
        "region-splitter": layer_splitter.app,
    }
    help_str = "The main CLI entry point."
    if cairosvg:
        group["cell-detection"] = cell_detection.app
    else:
        help_str += "\n\nNote: the cell-detection CLI is disabled since cairosvg was not found."

    app = click.Group("atlas_building_tools", group, help=help_str)

    version_message = (
        f"atlas-building-tools: {VERSION}\n"
        f"    atlas_densities: {atlas_densities.__version__}\n"
        f"    atlas_direction_vectors: {atlas_direction_vectors.__version__}\n"
        f"    atlas_placement_hints: {atlas_placement_hints.__version__}\n"
        f"    atlas_splitter: {atlas_splitter.__version__}"
    )
    app = click.version_option(message=version_message)(app)
    app()
