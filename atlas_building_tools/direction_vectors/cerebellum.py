"""Function computing the direction vectors of the AIBS mouse cerebellum

The algorithm creates a scalar field with low values in surfaces where fiber tracts are incoming
and high values where fiber tracts are outgoing. The direction vectors are given by the gradient
of this scalar field.

Note: At the moment, direction vectors are generated only for the following cerebellum subregions:
    - the flocculus
    - the lingula
"""
from typing import TYPE_CHECKING, Dict, List

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

import atlas_building_tools.direction_vectors.algorithms.blur_gradient as blur_gradient
from atlas_building_tools.direction_vectors.algorithms.blur_gradient import (
    RegionShading,
    compute_initial_field,
)
from atlas_building_tools.exceptions import AtlasBuildingToolsError

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore


def compute_direction_vectors(region_map: "RegionMap", annotation: "VoxelData") -> NDArray[float]:
    """
    Computes cerebellum's direction vectors as the normalized gradient of a custom scalar field.

    The computations are restricted to the flocculus and the lingula subregions.

    The output direction vector field is computed as the normalized gradient
    of a custom scalar field. This scalar field resembles a distance field in
    the neighborhood of the molecular layer.

    Afterwards, a Gaussian filter is applied and the normalized gradient of the
    blurred scalar field is returned.

    Note: For now, direction vectors are only computed for the flocculus and lingula subregions.
        A voxel lying outside these two regions will be assigned a 3D vector
        with np.nan coordinates.

    Arguments:
        region_map: hierarchy data structure of the AIBS atlas
        annotation: integer array of shape (W, H, D) holding the annotation of the whole mouse
         brain.

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """
    flocculus_direction_vectors = _flocculus_direction_vectors(region_map, annotation)
    lingula_direction_vectors = _lingula_direction_vectors(region_map, annotation)

    # Assembles flocculus and lingula direction vectors.
    direction_vectors = flocculus_direction_vectors
    lingula_mask = np.logical_not(np.isnan(lingula_direction_vectors))
    direction_vectors[lingula_mask] = lingula_direction_vectors[lingula_mask]

    return direction_vectors


def _flocculus_direction_vectors(
    region_map: "RegionMap", annotation: "VoxelData"
) -> NDArray[float]:
    """Returns the directin vectors for the flocculus subregions

    name: cerebellum related fiber tracts, acronym: cbf,  identifier = 960
    name: Flocculus, granular layer, acronym: FLgr, identifier: 10690
    name: Flocculus, purkinje layer, acronym: FLpu, identifier: 10691
    name: Flocculus molecular layer, acronym: FLmo, identifier: 10692
    """
    flocculus_regions = ["FLgr", "FLpu", "FLmo"]

    flocculus_region_to_weight = {
        "cbf": -5,
        "FLgr": -1,
        "FLpu": 0,
        "FLmo": 1,
        "outside_of_brain": 3,
    }

    return _region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        region_acronyms=flocculus_regions,
        region_to_weight=flocculus_region_to_weight,
        boundary_region="FLmo",
    )


def _lingula_direction_vectors(region_map: "RegionMap", annotation: "VoxelData") -> NDArray[float]:
    """Returns direction vectors for the lingula subregions

    name: cerebellum related fiber tracts, acronym: cbf,  identifier = 960
    name: Lingula, granular layer, acronym: LINGgr, identifier: 10705
    name: Lingula, purkinje layer, acronym: LINGpu, identifier: 10706
    name: Lingula molecular layer, acronym: LINGmo, identifier: 10707
    """
    lingula_regions = ["LINGgr", "LINGpu", "LINGmo"]

    lingula_region_to_weight = {"cbf": -5, "LINGgr": -1, "LINGpu": 0, "LINGmo": 1}

    return _region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        region_acronyms=lingula_regions,
        region_to_weight=lingula_region_to_weight,
        boundary_region="LINGmo",
    )


def _region_direction_vectors(
    region_map: "RegionMap",
    annotation: "VoxelData",
    region_acronyms: List[str],
    region_to_weight: Dict[str, int],
    boundary_region: str,
) -> NDArray[float]:
    """Calculate the direction vectors of a region

    Arguments:
        region_map: RegionMap to navigate the brain regions hierarchy.
        annotation: VoxelData containing the brain ids.
        region_acronyms: A list of the region acronyms of interest.
        region_to_weight: A dictionary the keys of which are region acronyms and the
            values weights for the shading algorithm. The special key `outside_of_brain` will
            be mapped to the identifier 0.
        boundary_region: The boundary region acronym.

    Returns:
        Vector field of 3D unit vectors over the isocortex volume with the same shape
        as the input one. Voxels outside the Isocortex have np.nan coordinates.

    Notes:
        The function allows to specify the parent acronyms instead of listing all the children
        for `region_acronyms` and `region_to_weight`. The hierarchies are resolved by including
        both parent and children in the region acronyms, propagating the weight of the parent to
        the children.

    """
    roi_ids = _acronyms_to_flattened_identifiers(region_map, region_acronyms)
    boundary_id = _acronyms_to_flattened_identifiers(region_map, [boundary_region])[0]

    complement_shading = [
        RegionShading(
            ids=roi_ids,
            boundary_region=boundary_id,
            boundary_offset=1,
            limit_distance=4,
            invert=True,
        )
    ]

    identifier_to_weight = _build_region_weight_map(region_map, region_to_weight)
    initial_field = compute_initial_field(annotation.raw, identifier_to_weight, complement_shading)
    return blur_gradient.compute_direction_vectors(annotation.raw, initial_field, roi_ids)


def _acronyms_to_flattened_identifiers(
    region_map: "RegionMap", region_acronyms: List[str]
) -> List[int]:
    """Returns a sorted list of the identifiers corresponding to the `region_acronyms`. If
    an acronym has children, the children are added to the list along with the parent.
    """
    flattened_identifiers = set()

    for acronym in region_acronyms:

        identifiers = region_map.find(acronym, attr="acronym", with_descendants=True)

        if not identifiers:
            raise AtlasBuildingToolsError(f"No identifiers found for acronym {acronym}")

        flattened_identifiers.update(identifiers)

    return sorted(flattened_identifiers)


def _build_region_weight_map(
    region_map: "RegionMap", region_to_weight: Dict[str, int]
) -> Dict[int, int]:
    """Returns a dictionary the keys of which are identifiers and the values of which are weights.
    If an acronym has children, they are added to the dictionary and the weight of the parent is
    assigned to them.
    """
    identifier_to_weight = {}

    for acronym, weight in region_to_weight.items():

        # outside_of_brain is a special name that is not an acronym but it is used
        # for determining the weight for the voxels that are outside of the brain
        if acronym == "outside_of_brain":
            identifier_to_weight[0] = weight
            continue

        for identifier in _acronyms_to_flattened_identifiers(region_map, [acronym]):

            if identifier in identifier_to_weight:
                raise AtlasBuildingToolsError(
                    f"Acronym {acronym} added from parent region already exists."
                )

            identifier_to_weight[identifier] = weight

    return identifier_to_weight
