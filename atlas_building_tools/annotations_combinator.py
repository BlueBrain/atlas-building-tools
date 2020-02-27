"""
Module responsible for the combination of different annotation files.

An annotation file is a volumetric nrrd file (voxellized 3D image) whose
voxels are 'annotated', i.e., voxels are assigned integer identifiers defining brain regions.
The hierarchy of brain regions and their identifiers are described in the ontology structure graph.
This graph is provided as a file with json format, often referred to as the hierarchy.json file.

Annotations combination is the process by which a more recent annotation file
(say, annotation/ccf_2017/annotation_10.nrrd) is combined with less recent annotation files
(say, annotation/ccf_2011/annotation_10.nrrd) because some regions are missing
in the more recent file.

Annotations combination was introduced when AIBS released their CCF v3 Mouse Atlas in 2017,
whose annotation file has missing regions with respect to the CCF v2 Mouse Atlas of 2011.
So far, annotations combination handles only to this use case.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


def combine_annotations(
    region_map, brain_annotation_ccfv2, fiber_annotation_ccfv2, brain_annotation_ccfv3
):
    """Combine `brain_annotation_ccfv2` with `brain_annotation_ccfv3` to reinstate missing regions.

    The ccfv2 brain annotation file contains the most complete set
    of brain regions while the ccfv3 brain annotation file is a more recent version
    of the brain annotation where some regions are missing.

    The ccfv2 fiber annotation file is required because fiber tracts
    are missing from the ccfv2 brain annotation file,
    whereas they are already included in the ccfv3 brain annotation file.

    These assumptions are based on the use case
    annotation 2011 (Mouse CCF v2) / annotation 2017 (Mouse CCF v3).

    Each annotation file has a resolution, eithe 10 um or 25 um.
    The input files and the output file should all have the same resolution.

    Args:
        region_map(voxcell.RegionMap): region map corresponding to the ccfv2/v3 annotations
        brain_annotation_ccfv2(voxcell.VoxelData): reference annotation file.
        fiber_annotation_ccfv2(voxcell.VoxelData): fiber annotation.
        brain_annotation_ccfv3(voxcell.VoxelData): new annotation.

    Returns:
        VoxelData object holding the combined annotation 3D array.
    """
    fiber_mask = fiber_annotation_ccfv2.raw > 0
    brain_annotation_ccfv2.raw[fiber_mask] = fiber_annotation_ccfv2.raw[fiber_mask]

    def leaf_nodes(region_map, ids):
        return [id_ for id_ in ids if region_map.is_leaf_id(id_)]

    brain_annotation_ccfv3_non_zero_mask = brain_annotation_ccfv3.raw > 0
    new_unique_ids = np.unique(
        brain_annotation_ccfv3.raw[brain_annotation_ccfv3_non_zero_mask]
    )
    new_unique_ids = new_unique_ids[np.nonzero(new_unique_ids)]  # removes the zero id
    leaves = leaf_nodes(region_map, new_unique_ids)
    missing_ids = np.isin(brain_annotation_ccfv2.raw, new_unique_ids, invert=True)
    combination_mask = (
        brain_annotation_ccfv3_non_zero_mask
        & np.isin(brain_annotation_ccfv3.raw, leaves, invert=True)
        & missing_ids
    )
    raw = brain_annotation_ccfv3.raw.copy()
    raw[combination_mask] = brain_annotation_ccfv2.raw[combination_mask]
    return brain_annotation_ccfv2.with_data(raw)
