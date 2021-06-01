
Description
===========

This folder contains the density profiles corresponding to PUBLICATION.
These files are used to create density nrrd files (a.k.a. density fields) for each of the neuron morphological types
(mtypes) listed in m_types/meta/mappting.tsv, see app/cell_densities.py. These profiles apply only to the mouse isocortex.

A cell density profile (see e.g. mtypes/BP.dat) is a series of non-negative numbers. Each number corresponds to a number of
cells in a specfic slice of the mouse isocortex. The six mouse iscortex layers are split into slices according to mtypes/meta/layers.tsv
. For instance layer 6 is divided into 35 slices of equal thickness and the first 35 values of each profile set the number of cells in each of these slices. Slice are parallel to layer boundaries, i.e.,
orthogonal to "cortical depth".

The command `atlas-building-tools cell-densities mtype-densites <OPTIONS>` creates mtypes volumetric densites based
on the above data together with the atlas placement hints. Placement hints are passed to the command via a path
to a yaml file of the following form:

.. code:: yaml

    'layerPlacementHintsPaths':
        'layer_1': '[PH]layer_1.nrrd'
        'layer_2': '[PH]layer_2.nrrd'
        'layer_3': '[PH]layer_3.nrrd'
        'layer_4': '[PH]layer_4.nrrd'
        'layer_5': '[PH]layer_5.nrrd'
        'layer_6': '[PH]layer_6.nrrd'
        'y': '[PH]y.nrrd'


The keys of `layerPlacementHintsPaths` are required to be either `y` or of the
form `layer_<index>` where `index` is a 1-based index.