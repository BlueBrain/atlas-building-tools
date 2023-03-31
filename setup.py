#!/usr/bin/env python
import importlib.util
from pathlib import Path

from setuptools import find_namespace_packages, setup

spec = importlib.util.spec_from_file_location(
    "atlas_building_tools.version",
    "atlas_building_tools/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

setup(
    name="atlas-building-tools",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing tools for building atlases",
    long_description="Library containing tools for building atlases",
    url="https://bbpgitlab.epfl.ch/nse/atlas-building-tools",
    download_url="git@bbpgitlab.epfl.ch:nse/atlas-building-tools.git",
    license="BBP-internal-confidential",
    python_requires=">=3.7",
    install_requires=[
        "atlas-direction-vectors>=0.1.1",
        "atlas-splitter>=0.1.1",
        "atlas-placement-hints>=0.1.1",
        "atlas-densities>=0.1.1",
        "cgal_pybind>=0.1.4",
        "poisson-recon-pybind>=0.1.2",  # python3.9/3.10 wheels for >=0.1.2
        "voxcell>=3.0.0",
        "click>=7.0",
        "numpy>=1.15.0",
        "rtree>=0.8.3",  # soft dep required for trimesh to allow indexing
        # Since version 1.6.0, scipy.optimize.linprog has fast, new methods for large, sparse problems
        # from the HiGHS library. We use the "highs" method in the densities module.
        "scipy>=1.6.0",
        "trimesh>=2.38.10",
    ],
    extras_require={
        "cell-detection": [
            "cairosvg>=2.4.2",
            "scikit-image>=0.17.2",
            "Pillow>=7.1.2",
        ],
        "tests": ["pytest>=4.4.0", "cairosvg>=2.4.2"],
    },
    packages=find_namespace_packages(include=["atlas_building_tools*"]),
    include_package_data=True,
    entry_points={"console_scripts": ["atlas-building-tools=atlas_building_tools.app.cli:cli"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
