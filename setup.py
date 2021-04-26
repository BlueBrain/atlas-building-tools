#!/usr/bin/env python

import imp

from setuptools import find_packages, setup

VERSION = imp.load_source("", "atlas_building_tools/version.py").__version__

setup(
    name="atlas_building_tools",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing tools for building atlases",
    url="https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
    download_url="ssh://bbpcode.epfl.ch/nse/atlas-building-tools",
    license="BBP-internal-confidential",
    python_requires=">=3.6.0",
    install_requires=[
        "click>=7.0",
        "cgal_pybind>=0.0.2",
        "cached-property>=1.5.2",
        "networkx>=2.5",
        "nptyping==1.0.1",
        "numba>=0.48.0",
        "numpy>=1.15.0",
        # numpy-quaternion version is capped because of an issue similar to
        # https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import
        "numpy-quaternion<=2019.12.11.22.25.52",
        "openpyxl>=3.0.5",
        "pandas>=1.0.3",
        "Pillow>=7.1.2",
        "poisson-recon-pybind>=0.1.0",
        "pynrrd>=0.4.0",
        "PyYAML>=5.3.1",
        "rtree>=0.9.4",
        "scipy>=1.4.1",
        "scikit-image>=0.17.2",
        "tqdm>=4.44.1",
        "trimesh>=3.6.18",
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
