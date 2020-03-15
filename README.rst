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
This python project depends on Regiodesics, a BBP C++ toolkit by the Viz Team,
see Regiodesics_.

To load Regiodesics on the BB5 cluster, you only need to run the following command:

.. code-block:: bash

    module load regiodesics/0.1.1

where 0.1.1 can be replaced by any deployed git tag.

On your desktop computer, you should install cmake (version >= 3.5), boost and OpenSceneGraph_ first.
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


.. _Regiodesics: https://bbpcode.epfl.ch/browse/code/viz/Regiodesics/tree/
.. _OpenSceneGraph:  http://www.openscenegraph.org/