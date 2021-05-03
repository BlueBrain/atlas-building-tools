"""app utils"""
import inspect
import logging
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import wraps

import click
import numpy as np

from atlas_building_tools.exceptions import AtlasBuildingToolsError

EXISTING_FILE_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)
EXISTING_DIR_PATH = click.Path(exists=True, readable=True, dir_okay=True, resolve_path=True)
LOG_DIRECTORY = "."


def set_verbose(logger, verbose):
    """Set the verbose level for the cli"""
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)])


class ParameterContainer(OrderedDict):
    """A dict class used to contain and display the parameters"""

    def __repr__(self):
        """Better printing than the normal OrderedDict"""
        return ", ".join(str(key) + ":" + str(val) for key, val in self.items())

    __str__ = __repr__


def log_args(logger, handler_path=None):
    """A decorator used to redirect logger and log arguments"""

    def set_logger(file_, logger_path=handler_path):

        if handler_path is None:
            logger_path = os.path.join(LOG_DIRECTORY, file_.__name__ + ".log")

        @wraps(file_)
        def wrapper(*args, **kw):
            logger.addHandler(logging.FileHandler(logger_path))
            param = ParameterContainer(inspect.signature(file_).parameters)
            for name, arg in zip(inspect.signature(file_).parameters, args):
                param[name] = arg
            for key, value in kw.items():
                param[key] = value
            date_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            logger.info(f"{date_str}:{file_.__name__} args:[{param}]")
            file_(*args, **kw)

        return wrapper

    return set_logger


# Atlas files consistency checks
# Copied from https://bbpcode.epfl.ch/code/#/admin/projects/nse/atlas-analysis,
# see atlas.py and utils.py.


def ensure_list(value):
    """Convert iterable / wrap scalar into list (strings are considered scalar)."""
    if isinstance(value, Iterable) and not isinstance(value, (str, Mapping)):
        return list(value)
    return [value]


def compare_all(data_sets, fun, comp):
    """Compares using comp all values extracted from data_sets using the fun access function

    Ex:
        compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose)
    """
    try:
        res = all(comp(fun(data_sets[0]), fun(other)) for other in data_sets[1:])
    except Exception as error_:
        raise AtlasBuildingToolsError("[compare_all] Bad operation during comparing") from error_
    return res


def assert_properties(atlases):
    """Assert that all atlases properties match

    Args:
        atlases: a list of voxeldata

    Raises:
        if one of the property is not shared by all data files
    """
    atlases = ensure_list(atlases)
    if not compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose):
        raise AtlasBuildingToolsError("Need to have the same shape for all files")
    if not compare_all(atlases, lambda x: x.voxel_dimensions, comp=np.allclose):
        raise AtlasBuildingToolsError("Need to have the same voxel_dimensions for all files")
    if not compare_all(atlases, lambda x: x.offset, comp=np.allclose):
        raise AtlasBuildingToolsError("Need to have the same offset for all files")


def assert_meta_properties(atlases):
    """Assert that all VoxelData metadata match

    Check that
        * VoxelData.shape
        * VoxelData.voxel_dimensions
        * VoxelData.offset

    is consistent accross the input VoxelData objects.

    For instance, it will not raise when comparing annotations with numpy shape
    (W, H, D) to direction vectors with numpy shape (W, H, D, 3).

    Args:
        atlases: a list of VoxelData objects

    Raises:
        if one of the above meta properties is not shared by all VoxelData objects.
    """
    atlases = ensure_list(atlases)
    if not compare_all(atlases, lambda x: x.shape, comp=np.allclose):
        raise AtlasBuildingToolsError("Need to have the same shape for all files")
    if not compare_all(atlases, lambda x: x.voxel_dimensions, comp=np.allclose):
        raise AtlasBuildingToolsError("Need to have the same voxel_dimensions for all files")
    if not compare_all(atlases, lambda x: x.offset, comp=np.allclose):
        raise AtlasBuildingToolsError("Need to have the same offset for all files")
