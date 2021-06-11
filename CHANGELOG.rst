Changelog
=========

Version 0.1.2
--------------
- Upgrades the dependency on cgal-pybind in setup.py: cgal-pybind>=0.1.1 [`BBPP82-614`_]
- Uses a volume slicer based on lengths of streamlines to compute of m-type volumetric densities [`BBPP82-614`_]
- Stores metadata json files for mouse isocortex, CA1 and thalamus in app/data/metadata [`NSETM-1474`_]
- Uses metadata json files in app/data/metadata as default files for the computation of placement hints [`NSETM-1474`_]
- Removes the proportion of voxels with the so-called "obtuse angle" issue from the distance report generated after distance interpolation [`NSETM-1474`_]
- mtype densities: supports mapping.tsv file with no inhibitory synapse class [`NSETM-1487`_]
- Adds CLI which estimates average cell densities using a linear fitting on a point cloud (average marker intensity, average cell density) [`NSETM-1475`_]
- Adds a CLI which turns non-density measurements of measurements.csv into average cell densities (number of cells per mm^3) [`NSETM-1475`_]
- Edits measurements.csv: adds missing unit for "cell count per slice" measurements and fix comments of "volume" measurements [`NSETM-1475`_]
- Renames the columns of every measurement CSV files: spaces are replaced by underscores (the same applies to key variables) [`NSETM-1475`_]
- Moves measurement files into app/data/measurements [`NSETM-1475`_]
- mtype densities: supports mapping.tsv file with no inhibitory synapse class [`NSETM-1487`_]
- Change the name atlas_building_tools to atlas-building-tools in setup.py (NSE standard)
- Adds CLI to interpolate NaN direction vectors by valid ones but also direction vectors specified by a mask [`NSETM-1343`_]
- The flatmap CLI has changed: it consumes now a json metadata file defining the ROI and its layers


Version 0.1.1 (2010-05-14)
--------------------------

- Uses cgal-pydind==0.1.0 and lowers the version required for openpyxl, rtree and trimesh [`NSETM-1454`_]
- Adds 3 consistency checks for distances to layer boundaries wrt to direction vectors (placement hints) [`NSETM-1343`_]
- Enlarges the mask of distance-wise problematic voxels with voxels whose distances are inconsistent according to the above checks [`NSETM-1343`_]
- Interpolates invalid distances by valid ones, restricting valid values to one hemisphere and one layer [`NSETM-1343`_]
- Adds a the label 3 to the mask of problematic voxels: 3 denotes a new problem caused by interpolation [`NSETM-1343`_]
- Fixes runtime error of atlas-building-tools cell-densities glia-cell-densities [`NSETM-1463`_]
- Fixes runtime error of atlas-building-tools cell-densities inhibitory-neuron-densities [`NSETM-1463`_]
- Renames the inhibitory-neuron-densities CLI with inhibitory-and-excitatory-neuron-densities [`NSETM-1463`_]
- Adds cell count and volume measurements from app/data/gaba_papers.xlsx (codes 5 and 6 in PV-SST-VIP worksheet) to app/data/measurements.csv

Version 0.1.0 (2021-04-27)
--------------------------
- CLI to combine AIBS ccfv2 and ccfv3 mouse annotation files
- CLI to split the layer 2/3 AIBS mouse isocortex into layer 2 and layer 3
- CLI to compute the direction vectors of the AIBS mouse isocortex, CA1 and thalamus regions
- CLI to compute the placement hints of the AIBS mouse isocortex, CA1 and thalamus regions
- CLI to detect cells and estimate cell radii inside png files from AIBS IHS experiments
- CLI to compute volumetric cell densities of astrocytes, microglia, oligodendrocytes, inhibitory and excitatory neurons
- CLI to compute volumetric cell densities of m_types specified in app/data/mtypes
- CLI to compute a flat map based on the streamlines of a laminar brain region
- CLI to turn the excel measurement compilation gaba_papers.xlsx of D. Rodarie into a CSV file.


.. _`BBPP82-614`: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-614
.. _`NSETM-1487`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1487
.. _`NSETM-1475`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1475
.. _`NSETM-1474`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1474
.. _`NSETM-1454`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1354
.. _`NSETM-1343`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1343
.. _`NSETM-1463`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1463
