.. |name| replace:: Atlas-Building-Tools

Welcome to |name| documentation!
==========================================

Introduction
============


This repository contains the tools used to:

* combine AIBS annotation files to reinstate missing brain regions
* assign direction vectors or orientations to voxels in a selected brain region
* compute distances of voxels to region boundaries (placement hints)
* compute densities of cells in each voxel of a brain region
* compute neuron 3D positions in a selected brain region



Installation
============

Besides the python package Rtree_, which might require the separate installation of
the C++ library libspatialindex_ (see instructions below), this project depends on two
BBP C++ toolkits, namely Regiodesics_ and Ultraliser_.

Regiodesics
-----------

This python project depends on Regiodesics_, a BBP C++ toolkit by the Viz Team.

To load Regiodesics on the BB5 cluster, run the following command:

.. code-block:: bash

    module load regiodesics/0.1.1

where 0.1.1 can be replaced by any deployed git tag.

For an installation of Regiodesics from the sources, install cmake (version >= 3.5 required),
boost and OpenSceneGraph_ first.
The installation of Regiodesics is as follows.

.. code-block:: bash

    git clone https://$USER@bbpcode.epfl.ch/code/a/viz/Regiodesics
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

    module load ultraliser


For an installation of Ultraliser from the sources, install cmake (version >= 3.5 required) and proceed as follows.

.. code-block:: bash

    git clone https://$USER@bbpcode.epfl.ch/code/a/viz/Ultraliser
    cd Ultraliser
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

On the BB5 cluster, the following commands install libspatialindex:

.. code-block:: bash

    git clone https://github.com/BlueBrain/spack.git spack --depth 1
    source spack/share/spack/setup-env.sh
    spack install py-rtree
    spack load py-rtree


.. _apt-get: https://askubuntu.com/questions/428772/how-to-install-specific-version-of-some-package
.. _brew: https://brew.sh/
.. _conda: https://docs.conda.io/en/latest/
.. _libspatialindex: https://libspatialindex.org/
.. _OpenSceneGraph: http://www.openscenegraph.org/
.. _Regiodesics: https://bbpcode.epfl.ch/browse/code/viz/Regiodesics/tree/
.. _Rtree: https://pypi.org/project/Rtree/
.. _Ultraliser: https://bbpcode.epfl.ch/browse/code/viz/Ultraliser/tree/