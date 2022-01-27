#!/usr/bin/env python

import imp

from setuptools import find_packages, setup

VERSION = imp.load_source("", "atlas_building_tools/version.py").__version__

setup(
    name="atlas-building-tools",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing tools for building atlases",
    url="https://bbpgitlab.epfl.ch/nse/atlas-building-tools",
    download_url="git@bbpgitlab.epfl.ch:nse/atlas-building-tools.git",
    license="BBP-internal-confidential",
    python_requires=">=3.6.0",
    install_requires=[
        "atlas-direction-vectors==0.1.0.dev0",
        "atlas-splitter>=0.1.0",
        "atlas-placement-hints>=0.1.0.dev0",
        "atlas-densities>=0.1.0.dev0",
        "click>=7.0",
        "cgal_pybind>=0.1.4",  # python3.9/3.10 wheels for >=0.1.4
        "cached-property>=1.5.2",
        "nptyping>=1.0.1",
        "numpy>=1.15.0",
        "openpyxl>=3.0.3",
        "pandas>=1.0.3",
        "Pillow>=7.1.2",
        "poisson-recon-pybind>=0.1.2",  # python3.9/3.10 wheels for >=0.1.2
        "pynrrd>=0.4.0",
        "PyYAML>=5.3.1",
        "rtree>=0.8.3",  # soft dep required for trimesh to allow indexing
        # Since version 1.6.0, scipy.optimize.linprog has fast, new methods for large, sparse problems
        # from the HiGHS library. We use the "highs" method in the densities module.
        "scipy>=1.6.0",
        "scikit-image>=0.17.2",
        "tqdm>=4.44.1",
        "trimesh>=2.38.10",
        "voxcell>=3.0.0",
        "xlrd>=1.0.0",
    ],
    extras_require={
        "cell-detection": "cairosvg>=2.4.2",
        "tests": ["pytest>=4.4.0", "mock>=2.0.0", "cairosvg>=2.4.2"],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["atlas-building-tools=atlas_building_tools.app.cli:cli"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
