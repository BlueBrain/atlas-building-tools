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
        'click>=7.0',
        'numpy>=1.16.3',
        'voxcell>=2.6.3.dev1',
    ],
    tests_require=[
        'pytest>=4.4.0',
        'mock>=2.0.0'
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'atlas-building-tools=atlas_building_tools.app.__main__:main'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
