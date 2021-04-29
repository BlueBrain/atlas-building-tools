Changelog
=========

Version 0.1.1
--------------
- Adds cell count and volume measurements from app/data/gaba_papers.xlsx (code 6 in PV-SST-VIP worksheet) to app/data/measurements.csv

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
