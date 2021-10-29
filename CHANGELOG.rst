Changelog
=========


Version 0.1.6
-------------
- Creates excitatory mtype densities using a taxonomy, composition and the excitatory neuron density [`CA-20`_]
- Documents the limitations of the algorithm splitting the mouse isocortex's layer 2/3 with warnings and error messages [`NSETM-1658`_]

Version 0.1.5 (2021-10-20)
--------------------------
- Fixes application crash caused by wrong regular expression in the default metadata json file of the mouse isocortex [`NSETM-1616`_]
- Fixes application crash caused by empty bottom or empty top shell passed to Regiodesics [`NSETM-1616`_]
- Integrates Dimitri Rodarie's linear program minimizing distances to density estimates [`BBPP82-628`_]
- Fixes wrong assert condition and wrong log in the layered_atlas module  [`NSETM-1616`_]
- Re-uses the identifiers of the nodes whose names end with "layer 2" or "layer 3" (but not "layer 2/3") when splitting layer 2/3 in AIBS 1.json [`BBPP82-640`_]
- Fixes wrong sanity check for measurement types and addition of spurious NaN or empty columns in density dataframes [`BBPP82-630`_]
- Addresses the troubles caused by duplicate region names in brain regions hierarchy [`BBPP82-630`_]
- Integrate the computation of mtype volumetric densities by Yann Roussel (BBP) [`NSETM-1574`_]
- Uses sphinx's autodoc and sphinx-click to generated HTML documentation of the command line interface [`NSETM-1484`_]
- Makes a clearer separation between the splitting of layer 2/3 in terms of hierarchy (json) and the splitting the annotated AIBS volume (nrrd) [`NSETM-1513`_]

Version 0.1.4 (2021-08-12)
--------------------------
- Migration from gerrit to gitlab: `https://bbpgitlab.epfl.ch/nse/atlas-building-tools` [`NSETM-1562`_]
- Uses the volume slicer of `nse/cgal-pybind` to split layer 2/3 of the AIBS mouse isocortex [`NSETM-1513`_]
- Creates and re-uses a common atlas option group [`NSETM-1513`_]
- Makes average inhibitory neuron densities consistent and create refined volumetric density files [`NSETM-1506`_]
- Turns warning on missing cairosvg module into a note in the main CLI help [`NSETM-1513`_]

Version 0.1.3 (2021-07-01)
--------------------------
- Adds a function creating a boolean mask of cylinder-shaped subregion of a 3D volume [`NSETM-1320`_]

Version 0.1.2 (2021-06-22)
--------------------------
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

Version 0.1.1 (2021-05-14)
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

.. _`CA-20`: https://bbpteam.epfl.ch/project/issues/browse/CA-20
.. _`NSETM-1658`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1658
.. _`BBPP82-628`: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-628
.. _`NSETM-1616`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1616
.. _`BBPP82-640`: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-640
.. _`NSETM-1574`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1574
.. _`BBPP82-630`: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-630
.. _`NSETM-1484`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1484
.. _`NSETM-1562`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1562
.. _`NSETM-1513`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1513
.. _`NSETM-1506`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1506
.. _`NSETM-1320`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1320
.. _`BBPP82-614`: https://bbpteam.epfl.ch/project/issues/browse/BBPP82-614
.. _`NSETM-1487`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1487
.. _`NSETM-1475`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1475
.. _`NSETM-1474`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1474
.. _`NSETM-1454`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1354
.. _`NSETM-1343`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1343
.. _`NSETM-1463`: https://bbpteam.epfl.ch/project/issues/browse/NSETM-1463
