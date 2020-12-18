
Description
===========

This folder contains the density profiles corresponding to PUBLICATION.
These files are used to create density nrrd files (a.k.a. density fields) for each of the neuron morphological types
(mtypes) listed in m_types/meta/mappting.tsv, see app/cell_densities.py. These profiles apply only to the mouse isocortex.

A cell density profile (see e.g. mtypes/BP.dat) is a series of non-negative numbers. Each number corresponds to a number of
cells in a specfic slice of the mouse isocortex. The six mouse iscortex layers are split into slices according to mtypes/meta/layers.tsv
. For instance layer 6 is divided into 35 slices of equal thickness and the first 35 values of each profile set the number of cells in each of these slices. Slice are parallel to layer boundaries, i.e.,
orthogonal to "cortical depth".