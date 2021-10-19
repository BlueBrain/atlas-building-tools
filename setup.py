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
        "click>=7.0",
        "cgal_pybind>=0.1.1",
        "cached-property>=1.5.2",
        "networkx>=2.4",
        "nptyping>=1.0.1",
        # In numba>0.48.0, numba.utils does not exist anymore and this causes numba to
        # be left unused by numpy-quaternion (warning), see
        # https://bbpteam.epfl.ch/project/issues/browse/NSETM-1463
        "numba==0.48.0",
        "numpy>=1.15.0",
        # numpy-quaternion version is capped because of an issue similar to
        # https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import
        "numpy-quaternion<=2019.12.11.22.25.52",
        "openpyxl>=3.0.3",
        "pandas>=1.0.3",
        "Pillow>=7.1.2",
        "poisson-recon-pybind>=0.1.0",
        "pynrrd>=0.4.0",
        "PyYAML>=5.3.1",
        "rtree>=0.8.3",
        "scipy>=1.4.1",
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
