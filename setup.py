#!/usr/bin/env python

import imp

from setuptools import setup, find_packages

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
    python_requires='>=3.6.0',
    install_requires=[
        'cairosvg>=2.4.2',
        'click>=7.0',
        'dataclasses>=0.6',
        'h5py>=2.10.0',
        'lazy>=1.4',
        'networkx',
        'nptyping==1.0.1',
        'numba>=0.48.0',
        'numpy>=1.19',
        'numpy-quaternion>=2019.10.3.10.26.21',
        'pandas>=1.0.3',
        'pathlib>=1.0.1',
        'Pillow>=7.1.2',
        'pynrrd>=0.4.0',
        'PyYAML>=5.3.1',
        'rtree',
        'scipy>=1.4.1',
        'scikit-image>=0.17.2',
        'tqdm>=4.44.1',
        'trimesh>=3.6.18',
        'typing>=3.7.4.1',
        'voxcell>=2.7.1',
        'xlrd>=1.0.0',
    ],
    tests_require=['pytest>=4.4.0', 'mock>=2.0.0', 'rtree'],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': ['atlas-building-tools=atlas_building_tools.app.cli:main']
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
