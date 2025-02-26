.. warning::
   The Blue Brain Project concluded in December 2024, so development has ceased under the BlueBrain GitHub organization.
   Future development will take place at: https://github.com/openbraininstitute/atlas-building-tools

Overview
=========

This project contains the tools to create the data files the `BBP Cell Atlas`_ is built on.
The creation of atlas files is the first step towards the `creation of a circuit`_.

The tools implementation is based on the methods of `A Cell Atlas for the Mouse Brain`_ by Csaba Eroe et al., 2018.
The source code was originally written by Csaba Eroe, Dimitri Rodarie, Hugo Dictus, Lu Huanxiang, Wajerowicz Wojciech and Jonathan Lurie.

Atlas building tools operate on data files coming from the `Allen Institute for Brain Science (AIBS)`_.
These data files were obtained via experiments performed on P56 wild-type mouse brains.

The tools allow to:

* combine AIBS annotation files to reinstate missing mouse brain regions
* combine several AIBS gene marker datasets, to be used as hints for the spatial distribution of glia cells
* split the layer 2/3 of the AIBS mouse isocortex into layer 2 and layer 3
* assign direction vectors or orientations to voxels in a selected brain region
* compute distances between voxels and region boundaries, i.e., the so-called placement hints to be used by the `placement-algorithm`_
* compute cell densities for several cell types including neurons and glia cells in the whole mouse brain
* flatten a laminar brain region by collapsing the streamlines of its fiber tracts direction field, i.e., create a flat map to be used by `white-matter-projections`_

Tools can be used through a command line interface.

Currently, atlas-building-tools is mainly a wrapper, for backwards compatibility, around:

* https://github.com/BlueBrain/atlas-densities
* https://github.com/BlueBrain/atlas-direction-vectors
* https://github.com/BlueBrain/atlas-placement-hints
* https://github.com/BlueBrain/atlas-splitter

Installation
============

This python project depends on:

* the python package Rtree_, which might require the separate installation of the C++ library libspatialindex_ (see instructions below)
* the BBP C++ toolkits Regiodesics_ and Ultraliser_
* the BBP python-C++ bindings cgal-pybind_

The remaining installation instructions are only relevant if you want to install `atlas-buidling-tools` on BB5
with its latest sources. Prior to running


.. code-block:: bash

    git clone git@bbpgitlab.epfl.ch:nse/atlas-building-tools.git
    cd atlas-building-tools
    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ -e .

you need to install the dependencies listed below.

Regiodesics
-----------

This python project depends on Regiodesics_, a BBP C++ toolkit by the Viz Team.

To load Regiodesics on the BB5 cluster, run the following command:

.. code-block:: bash

    module load regiodesics/0.1.2

For an installation of Regiodesics from the sources, install cmake (version >= 3.5 required), boost and OpenSceneGraph_ first.
The installation of Regiodesics is as follows.

.. code-block:: bash

    git clone https://bbpgitlab.epfl.ch/nse/archive/regiodesics
    cd Regiodesics
    git submodule update --init
    mkdir build
    cd build
    cmake ..
    make -j
    cd ..
    export PATH=$PATH:$PWD/build/bin


Ultraliser
----------

This python project depends on Ultraliser_, a BBP C++ toolkit by the Viz Team.

To load Ultraliser on the BB5 cluster, run the following command:

.. code-block:: bash

    module load unstable ultraliser/0.2.0


For an installation of Ultraliser from the sources, install cmake (version >= 3.5 required) and proceed as follows.

.. code-block:: bash

    git clone https://github.com/BlueBrain/ultraliser
    cd ultraliser
    mkdir build
    cd build
    cmake ..
    make -j
    cd ..
    export PATH=$PATH:$PWD/build/bin


Rtree
-----

This python project depends on Rtree_, a python package which requires
the libspatialindex_ library, a C++ dependency.

If you are using conda_, then libspatialindex should be installed automatically with Rtree.

If this is not the case, you can install libspatialindex via brew_ on MacOS or via apt-get_ on Ubuntu systems.

On the BB5 cluster, install rtree and its dependency libspatialindex with:

.. code-block:: bash

    module load unstable py-rtree/0.8.3


cgal-pybind
-----------
The BBP python project cgal-pybind_ contains python bindings for several functions of the
CGAL_ C++ library. The algorithm of atlas-building-tools which creates a flat map uses specifically
CGAL's `authalic map`_.

On the BB5 cluster, install cgal-pybind with:

.. code-block:: bash

    module load unstable py-cgal-pybind/0.0.2

poisson-recon-pybind
--------------------
The BBP python project poisson-recon-pybind_ contains python bindings for the reconstruction
surface algorithm of PoissonRecon_.

On the BB5 cluster, install poisson-recon-pybind with:

.. code-block:: bash

    module load unstable py-poisson-recon-pybind/0.1.0


Instructions for developers
===========================

Run the following commands before submitting your code for review:

.. code-block:: bash

    cd atlas-building-tools
    isort -l 100 --profile black atlas_building_tools tests setup.py
    black -l 100 atlas_building_tools tests setup.py

These formatting operations will help you pass the linting check `testenv:lint` defined in
`tox.ini`.

Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license see LICENSE.txt.

Copyright © 2020-2024 Blue Brain Project/EPFL


.. _`Allen Institute for Brain Science (AIBS)`: https://alleninstitute.org/what-we-do/brain-science/
.. _`A Cell Atlas for the Mouse Brain`: https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full
.. _apt-get: https://askubuntu.com/questions/428772/how-to-install-specific-version-of-some-package
.. _`authalic map`: https://doc.cgal.org/latest/Surface_mesh_parameterization/classCGAL_1_1Surface__mesh__parameterization_1_1Discrete__authalic__parameterizer__3.html
.. _`BBP Cell Atlas`: https://portal.bluebrain.epfl.ch/resources/models/cell-atlas/
.. _brew: https://brew.sh/
.. _cgal-pybind: https://bbpgitlab.epfl.ch/nse/cgal-pybind
.. _CGAL: https://www.cgal.org/
.. _conda: https://docs.conda.io/en/latest/
.. _libspatialindex: https://libspatialindex.org/
.. _OpenSceneGraph: http://www.openscenegraph.org/
.. _`placement-algorithm`: https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/index.html
.. _PoissonRecon: https://github.com/mkazhdan/PoissonRecon
.. _`poisson-recon-pybind`: https://bbpgitlab.epfl.ch/nse/poisson-recon-pybind
.. _Regiodesics: https://bbpgitlab.epfl.ch/nse/archive/regiodesics
.. _Rtree: https://pypi.org/project/Rtree/
.. _Ultraliser: https://github.com/BlueBrain/ultraliser
.. _white-matter-projections: https://bbpgitlab.epfl.ch/nse/white-matter-projections
.. _`creation of a circuit`: https://bbpteam.epfl.ch/documentation/projects/circuit-build/latest/tutorial.html
