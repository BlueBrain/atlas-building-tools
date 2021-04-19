"""
Generic script to compute direction vectors of a laminar region.

This script supports the case of regions separated in two hemispheres such as the isocortex or
the thalamus. It allows the use of two different lower-level algorithms computing directions
vectors. Both algorithms are based on the identification of a source and a target region:
a simple blur gradient approach and Regiodesics.

These two algorithms are appropriate when the fibers of the brain region
follow streamlines which start from and end to specific surfaces. The region
from where fibers originate is referred to as the source region.
The region where fibers end is referred to as the target region.
In Regiodesics terminology, these correspond respectively to the bottom and top
shells, a.k.a lower and upper shells.

This script is used to compute the mouse isocortex and the mouse thalamus directions vectors.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_building_tools.direction_vectors.algorithms import regiodesics, simple_blur_gradient
from atlas_building_tools.utils import load_region_map, split_into_halves

ALGORITHMS: Dict[str, Callable] = {
    "simple_blur_gradient": simple_blur_gradient.compute_direction_vectors,
    "regiodesics": regiodesics.compute_direction_vectors,
}


def attributes_to_ids(
    region_map: Union[str, dict, RegionMap],
    attributes: List[Union[Tuple[str, str], Tuple[str, int]]],
) -> List[int]:
    """
    Make a list of region identifiers out of hierarchy attributes including descendants.

    Args:
        region_map: the hierachy description of the region of
            interest. Either a path to hierarch.json file or a dict.
        attributes: list of pairs (attribute: str, value: Union[str, int])
            where `attribute` is either 'id', 'acronym' or 'name' and `value` is a value for
            of `attribute`.

    Return:
        duplicate-free list of region identifiers corresponding to the
        input attribute values.

    """
    region_map = load_region_map(region_map)
    ids = set()
    for (attribute, value) in attributes:
        ids |= region_map.find(value, attribute, ignore_case=False, with_descendants=True)
    return list(ids)


def direction_vectors_for_hemispheres(
    landscape: Dict[str, NDArray[bool]],
    algorithm: str,
    hemisphere_options: Optional[Dict[str, Union[str, None]]] = None,
    **kwargs: Union[int, float, str]
) -> NDArray[np.float32]:
    """
    Compute direction vectors for each of the two hemispheres.

    Arguments:
        landscape: dict of the form
            {'source': NDArray[bool], 'inside': NDArray[bool], 'target': NDArray[bool]}
            where the value corresponding to
                'source' is the 3D binary mask of the source region, i.e.,
                    the region where the fibers originate from,
                'inside' is the 3D binary mask of the region where direction vectors
                    are computed,
                'target' is the 3D binary mask of the fibers target region.
        algorithm: the algorithm to use to generate direction vectors
                   (either 'simple_blur_gradient' or 'regiodesics').
        hemisphere_options(None|dict): None or a dict of the form
            {'set_opposite_hemisphere_as': <str>} or {'set_opposite_hemisphere_as': None}.
            If `hemisphere_options` is None, i.e., the default value, the region of interest
            is not split into hemispheres. Otherwise, the string value corresponding to
            'set_opposite_hemisphere_as' indicates, if for each hemisphere the opposite hemisphere
            should be included as a source or a target for the former. The possible values are
            'source', 'target' or None. If the value is None, the opposite hemisphere is not used,
            neither as source, nor as target.
        kwargs: (optional) Options specific to the underlying algorithm.
            For regiodesics.compute_direction_vectors, the option regiodesics_path=str can be used
            to indicate where the regiodesics executable is located. Otherwise this function will
            attempt to find it by means of distutils.spawn.find_executable.
            For simple_blur_gradient.direction_vectors, the option sigma=float can be used to
            specify the standard deviation of the Gaussian blur while source_weight=float,
            target_weight=float can be used to set custom weights in the source and target regions.

    Returns:
        Array holding a vector field of unit vectors defined on the `inside` 3D volume. The shape
        of this array is (W, L, D, 3) if the shape of `inside` is (W, L, D).
        Outside the `inside` volume, the returned direction vectors have np.nan coordinates.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("algorithm must be one of {}".format(ALGORITHMS.keys()))

    set_opposite_hemisphere_as = (
        hemisphere_options["set_opposite_hemisphere_as"] if hemisphere_options is not None else None
    )
    if set_opposite_hemisphere_as not in {None, "source", "target"}:
        raise (
            ValueError(
                'Argument "set_opposite_hemisphere_as" should be None, "source" or'
                ' "target". Got {}'.format(set_opposite_hemisphere_as)
            )
        )
    hemisphere_masks = [landscape["inside"]]
    if hemisphere_options is not None:
        # We assume that the region of interest has two hemispheres
        # which are symetric wrt the plane z = volume.shape[2] // 2.
        hemisphere_masks = split_into_halves(  # type: ignore
            np.ones(landscape["inside"].shape, dtype=bool)
        )

    direction_vectors = np.full(landscape["inside"].shape + (3,), np.nan, dtype=np.float32)
    for hemisphere in hemisphere_masks:
        source = (
            np.logical_or(landscape["source"], ~hemisphere)
            if set_opposite_hemisphere_as == "source"
            else np.logical_and(landscape["source"], hemisphere)
        )
        target = (
            np.logical_or(landscape["target"], ~hemisphere)
            if set_opposite_hemisphere_as == "target"
            else np.logical_and(landscape["target"], hemisphere)
        )
        direction_vectors[hemisphere] = ALGORITHMS[algorithm](
            source, np.logical_and(landscape["inside"], hemisphere), target, **kwargs
        )[hemisphere]

    return direction_vectors


Attribute = Union[Tuple[str, str], Tuple[str, int]]
AttributeList = List[Attribute]


def compute_direction_vectors(
    region_map: Union[str, dict, RegionMap],
    brain_regions: Union[str, VoxelData],
    landscape: Dict[str, AttributeList],
    algorithm: str = "simple_blur_gradient",
    hemisphere_options: Optional[Dict[str, Union[str, None]]] = None,
    **kwargs: Union[int, float, str]
) -> NDArray[np.float32]:
    """
    Computes within `inside` direction vectors that originate from `source` and end in `target`.

    Args:
        region_map: a path to hierarchy.json or dict made of such a file or a
            RegionMap object. Defaults to None.
        brain_regions: full annotation array from which the region of interest `inside` will be
            extracted.
        landscape: landscape: dict of the form
            {source': AttributeList, 'inside': AttributeList, 'target': AttributeList}
            where the value corresponding to
                'source' is a list of acronyms or of integer identifiers defining the
                    source region of fibers.
                'inside' is a list of acronyms or of integer identifiers defining the
                    the region where the direction vectors are computed.
                'target' is a list of acronyms or of integer identifiers defining the
                    the region where the fibers end.
        algorithm: name of the algorithm to be used for the computation
            of direction vectors. One of `regiodesics` or `simple_blur_gradient`.
            Defaults to `simple_blur_gradient`.
        hemisphere_options: None or a dict of the form
            {'set_opposite_hemisphere_as': str} or {'set_opposite_hemisphere_as': None}.
            If `hemisphere_options` is None, i.e., the default value, the region of interest
            is not split into hemispheres. Otherwise, the string value corresponding to
            'set_opposite_hemisphere_as' indicates, if for each hemisphere the opposite hemisphere
            should be included as a source or a target for the former. The possible values are
            'source', 'target' or None. If the value is None, the opposite hemisphere is not used,
            neither as source, nor as target.
        kwargs: see direction_vectors_for_hemispheres documentation.

    Returns:
        A vector field of float32 3D unit vectors over the input 3D volume.

    """
    if algorithm not in ALGORITHMS:
        raise ValueError("`algorithm` must be one of {}".format(ALGORITHMS))

    if isinstance(brain_regions, str):
        brain_regions = VoxelData.load_nrrd(brain_regions)
    else:
        if not isinstance(brain_regions, VoxelData):
            raise ValueError("`brain_regions` must be specified as a path or a VoxelData object.")
    landscape = {
        "source": np.isin(
            brain_regions.raw,  # type: ignore
            attributes_to_ids(region_map, landscape["source"]),
        ),
        "inside": np.isin(
            brain_regions.raw,  # type: ignore
            attributes_to_ids(region_map, landscape["inside"]),
        ),
        "target": np.isin(
            brain_regions.raw,  # type: ignore
            attributes_to_ids(region_map, landscape["target"]),
        ),
    }
    direction_vectors = direction_vectors_for_hemispheres(
        landscape, algorithm, hemisphere_options, **kwargs
    )

    return direction_vectors
