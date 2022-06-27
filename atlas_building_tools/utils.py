"""Generic Atlas files tools"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union

import numpy as np  # type: ignore
from atlas_commons.typing import BoolArray, NDArray, NumericArray
from scipy.ndimage.morphology import generate_binary_structure  # type: ignore
from scipy.signal import correlate  # type: ignore
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_building_tools.exceptions import AtlasBuildingToolsError

L = logging.getLogger(__name__)


def load_region_map(region_map: Union[str, dict, RegionMap]) -> RegionMap:
    """
    Load a RegionMap object specified in one of three possible ways.

    Args:
        region_map: path to hierarchy.json, dict made of such a file or a RegionMap object.

    Returns:
        A true RegionMap object.
    """
    if isinstance(region_map, str):
        region_map = RegionMap.load_json(region_map)
    elif isinstance(region_map, dict):
        region_map = RegionMap.from_dict(region_map)
    elif isinstance(region_map, RegionMap):
        return region_map
    else:
        raise TypeError(f"Cannot convert {type(region_map)} to RegionMap")
    return region_map


def query_region_mask(
    region: dict, annotation: NDArray[int], region_map: Union[str, dict, "RegionMap"]
) -> BoolArray:
    """
    Create a mask for the region defined by `query`.

    Args:
        query: dict of the form
            {
                "query": "@.*layer 1",
                "attribute": "name",
                "with_descendants": True,  # Optional, defaults to False.
            }
            Each key corresponds to an argument of the function RegionMap.find that will be called
            to create the mask.
        annotation: 3D array of region ids containing the region to mask.
        region_map: path to hierarchy.json, dict made of such a file or a RegionMap object.

    Returns:
       3D boolean array of the same shape as annotation.
    """
    region_map = load_region_map(region_map)
    ids = region_map.find(
        region["query"], region["attribute"], with_descendants=region.get("with_descendants", False)
    )

    return np.isin(annotation, list(ids))


def get_region_mask(
    acronym: str, annotation: NDArray[int], region_map: Union[str, dict, "RegionMap"]
) -> BoolArray:
    """
    Create a mask for the region defined by `acronym`.

    Args:
        acronym: the acronym of the region to mask. If it starts with @
                 the remainder is interprereted as a regexp.
        annotation: 3D array of region ids containing the region to mask.
        region_map: path to hierarchy.json, dict made of such a file or a RegionMap object.

    Returns:
       3D boolean array of the same shape as annotation.

    """
    region = {"query": acronym, "attribute": "acronym", "with_descendants": True}

    return query_region_mask(region, annotation, region_map)


def split_into_halves(
    volume: NumericArray,
    halfway_offset: int = 0,
) -> Tuple[NumericArray, NumericArray]:
    """
    Split input 3D volume into two halves along the z-axis.

    Args:
        volume: 3D numeric array.
            halfway_offset: Optional offset used for the
            splitting along the z-axis.
    Returns:
        tuple(left_volume, right_volume), the two halves of the
        input volume. Each has the same shape as `volume`.
        Voxels are zeroed for the z-values above, respectively
        below, the half of the z-dimension.
    """
    z_halfway = volume.shape[2] // 2 + halfway_offset
    left_volume = volume.copy()
    left_volume[..., z_halfway:] = 0
    right_volume = volume.copy()
    right_volume[..., :z_halfway] = 0
    return left_volume, right_volume


def is_obtuse_angle(vector_field_1: NumericArray, vector_field_2: NumericArray) -> NDArray[bool]:
    """
    Returns a mask indicating which vector pairs form an obtuse angle.

    Arguments:
        vector_field_1: 3D vector field, i.e., numeric array of shape
            (M, N, ..., 3).
        vector_field_2: 3D vector field, i.e., numeric array of shape
            (M, N, ..., 3).
    Returns:
       Binary mask of shape (M, N, ...) indicating which pairs of vectors
        form an obtuse angle.
    """
    return np.sum(vector_field_1 * vector_field_2, axis=-1) < 0


def copy_array(array: NDArray, copy=True) -> NDArray:
    """
    Returns either `array` or a deep copy of `array` depending on `copy`.

    Args:
        array: a numpy ndarray
        copy: Optional boolean. If True, returns a hard copy of `array`, otherwise
            returns `array` itself.
    Returns:
        a copy of `array` or `array` itself if `copy` is False.
    """
    return array.copy() if copy else array


def compute_boundary(v_1, v_2):
    """Compute the boundary shared by two volumes.

    The voxels of `v_1` (resp. of `v_2`) are labeled with the value 1 (resp. 8).
    We build the filter corresponding to the 6 neighbour voxels that share a face
    with a reference voxel. We apply a covolution of the filter with the labeled volume.
    In the resulting labeled volume, the `v_1` voxels with label > 8 are exactly those voxels
    that share a face with at least one voxel of `v_2`.
    (The interior voxels of `v_1` have labels bounded above by 7).

    Check https://docs.scipy.org/doc/scipy/reference/ndimage.html for the doc
    of the functions generate_binary_structure and correlate used below.

    Args:
        v_1(numpy.ndarray): boolean 3D array holding the mask of the first volume.
        v_2(numpy.ndarray): boolean 3D array holding the mask of the second volume.

    Returns:
        shared_boundary(numpy.ndarray), 3D boolean array holding the mask of the boundary shared
        by `v_1` and `v_2`. This corresponds to a subset of `v_1`.
    """

    filter_ = generate_binary_structure(3, 1).astype(int)
    full_volume = correlate(v_1 * 1 + v_2 * 8, filter_, mode="same")

    return np.logical_and(v_1, full_volume > 8)


def assert_metadata_content(metadata: dict) -> None:
    """
    Raise an error if some mandatory key is missing in `metadata`.

    Args:
        metadata: dict of the form
            {
                "region" : {
                    "name": "aibs_isocortex",
                    "query": "Isocortex",
                    "attribute": "acronym"
                }
                "layers": {
                    "names": ["layer 1", "layer 2", "layer3", "layer 4", "layer 5", "layer 6"],
                    "queries": ["@.*;L1$", "@.*;L2$", "@.;L3*$", "@.*;L4$", "@.*;L5$", "@.*;L6$"],
                    "attribute": "acronym"
                }
            }
            Queries in "query" or "queries" should be compliant with the interface of
            voxcell.RegionMap.find interface. The value of "attribute" can be "acronym" or "name".

    Raise:
        AtlasBuildingToolsError if a mandatory key is missing or if the length of
        layer names and queries are different.
    """

    if "region" not in metadata:
        raise AtlasBuildingToolsError('Missing "region" key')

    metadata_region = metadata["region"]

    missing = {"name", "query", "attribute"} - set(metadata_region.keys())
    if missing:
        err_msg = (
            'The "region" dictionary has the following mandatory keys: '
            '"name", "query" and "attribute".'
            f" Missing: {missing}."
        )
        raise AtlasBuildingToolsError(err_msg)

    if "layers" not in metadata:
        raise AtlasBuildingToolsError('Missing "layers" key')

    metadata_layers = metadata["layers"]

    missing = {"names", "queries", "attribute"} - set(metadata_layers.keys())
    if missing:
        err_msg = (
            'The "layers" dictionary has the following mandatory keys: '
            '"names", "queries" and "attribute".'
            f" Missing: {missing}."
        )
        raise AtlasBuildingToolsError(err_msg)

    if not (
        isinstance(metadata_layers["names"], list)
        and isinstance(metadata_layers["queries"], list)
        and len(metadata_layers["names"]) == len(metadata_layers["queries"])
    ):
        raise AtlasBuildingToolsError(
            'The values of "names" and "queries" must be lists of the same length.'
        )


def create_layered_volume(
    annotated_volume: NDArray[int],
    region_map: "RegionMap",
    metadata: dict,
):
    """
    Create a 3D volume whose voxels are labeled by 1-based layer indices.

    Args:
        annotated_volume: integer numpy array of shape (W, H, D) where
            W, H and D are the integer dimensions of the volume domain.
        region_map: RegionMap object used to navigate the brain regions hierarchy.
        metadata: dict, see :fun:`atlas_building_tools.utils.assert_metadata`.
            This dict contains the definitions of the layers to be built.

    Returns:
        A numpy array of the same shape as the input volume, i.e., (W, H, D). Voxels are labeled by
        the 1-based indices of the layers defined in `metadata`. Voxels out of the region defined in
        `metadata` are labeled with the 0 index.

    Raises:
        AtlasBuildingErrors if `metadata` has an incorrect format.
    """

    assert_metadata_content(metadata)

    metadata_layers = metadata["layers"]
    layers = np.zeros_like(annotated_volume, dtype=np.uint8)
    region_ids = region_map.find(
        metadata["region"]["query"],
        attr=metadata["region"]["attribute"],
        with_descendants=metadata["region"].get("with_descendants", False),
    )
    for (index, query) in enumerate(metadata_layers["queries"], 1):
        layer_ids = region_map.find(
            query,
            attr=metadata_layers["attribute"],
            with_descendants=metadata_layers.get("with_descendants", False),
        )
        layers[np.isin(annotated_volume, list(layer_ids & region_ids))] = index

    return layers


def get_layer_masks(
    annotated_volume: NDArray[int],
    region_map: "RegionMap",
    metadata: dict,
) -> Dict[str, NDArray[bool]]:
    """
    Create a 3D boolean mask of each layer in `metadata`.

    Args:
        annotated_volume: int array of shape (W, H, D) where W, H and D are integer dimensions;
            this array is the annotated volume of the brain region of interest.
        region_map: RegionMap object to navigate the brain regions hierarchy.
        metadata: dict describing the region of interest and its layers. See `app/datat/metadata`
            for examples.

    Returns: dict whose keys are the regions names from `metadata` and whose values
        are boolean masks of the corresponding regions in `annotated_volume`.
    """
    layers = create_layered_volume(annotated_volume, region_map, metadata)

    return {name: layers == i for (i, name) in enumerate(metadata["layers"]["names"], 1)}


def _merge_nrrd(source: "VoxelData", target: "VoxelData", sentinel) -> "VoxelData":
    """Merge voxels in `source` to `target`, ignoring those with `sentinel` in the last dimension.

    Note: source and target are unmodified

    If the sentinel occupies all the values in the last dimension, the voxel is ignored.
    Ex: If a voxel has the value (0, 0, 1) it would not be ignored if the sentinel is 0
    """
    if not np.allclose(source.shape, target.shape):
        raise ValueError(f"Different shapes: {source.shape} != {target.shape}")

    if not np.allclose(source.offset, target.offset):
        raise ValueError(f"Different offsets: {source.offset} != {target.offset}")

    if not np.allclose(source.voxel_dimensions, target.voxel_dimensions):
        raise ValueError(
            f"Different voxel dimensions: {source.voxel_dimensions} != "
            f"{target.voxel_dimensions}"
        )

    if source.payload_shape != target.payload_shape:
        raise ValueError(
            f"Different payload sizes: {source.payload_shape} != {target.payload_shape}"
        )

    if np.isnan(sentinel):
        mask = np.isnan(source.raw)
    else:
        mask = source.raw == sentinel

    mask = np.invert(mask.all(axis=-1))
    ret = target.raw.copy()
    ret[mask] = source.raw[mask]

    return target.with_data(ret)


def merge_nrrd(input_paths: List[str], sentinel: str) -> "VoxelData":
    """Merge multiple NRRD files together

    Args:
        input_paths: paths of nrrd files to be merged.
        Ones that come later override contents of earlier files
        They should all have the same offset, voxel size, and total shape.
        sentinel: Value for voxels that aren't to be copied (ie: 'nan', or 0)

    Returns:
        VoxelData
    """
    if len(input_paths) == 0:
        raise ValueError("Must have at least one file to merge")

    ret = VoxelData.load_nrrd(input_paths[0])

    sentinel = ret.raw.dtype.type(sentinel)

    for path in input_paths[1:]:
        source = VoxelData.load_nrrd(path)
        try:
            ret = _merge_nrrd(source, ret, sentinel)
        except ValueError as exc:
            raise ValueError(f"For file {path}: {exc}") from exc

    return ret
