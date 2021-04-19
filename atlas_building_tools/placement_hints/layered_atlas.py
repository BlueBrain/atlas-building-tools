"""Module for the computation of
voxel-to-layer distances wrt to direction vectors in a laminar brain region.

This module is used for the computation of placement hints in the mouse
isocortex and in the mouse Hippocampus CA1 region.
"""
import logging
import os
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np  # type: ignore
from cached_property import cached_property  # type: ignore
from nptyping import NDArray  # type: ignore
from tqdm import tqdm  # type: ignore

from atlas_building_tools.distances.create_watertight_mesh import create_watertight_trimesh
from atlas_building_tools.distances.distances_to_meshes import (
    distances_from_voxels_to_meshes_wrt_dir,
    fix_disordered_distances,
)
from atlas_building_tools.placement_hints.utils import centroid_outfacing_mesh, layers_volume
from atlas_building_tools.utils import get_region_mask, split_into_halves

if TYPE_CHECKING:  # pragma: no cover
    import trimesh  # type: ignore
    from voxcell import RegionMap, VoxelData  # type: ignore

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

# Constants
LEFT = 0  # left hemisphere
RIGHT = 1  # right hemisphere


class DistanceProblem(IntEnum):
    """
    Enumerate distance-related problems detected after the computation of placement hints.
    """

    NO_PROBLEM = 0
    BEFORE_INTERPOLATION = 1  # NaN distance value or excessive layer thickness
    AFTER_INTERPOLATION = 2  # Excessive layer thickness only
    # NaN distance values are assumed to have been interpolated with valid distances of nearby
    # voxels.


class LayeredAtlas:
    """
    Class holding the data of a layered atlas, i. e., an atlas with well-defined layers
    for which boundary meshes can be created.
    """

    def __init__(
        self,
        acronym: str,
        annotation: "VoxelData",
        region_map: "RegionMap",
        layer_regexps: List[str],
    ):
        """
        acronym: acronym of the atlas as written in the
            hierarchy.json file.
            Example: 'isocortex' or 'CA1', but could be another layered brain structure.
        annotation: annotated volume enclosing the whole brain atlas.
        region_map: Object to navigate the brain regions hierarchy.
        layer_regexps: list of regular expressions defining the layers in the brain hierarchy.
        """
        self.acronym = acronym
        self.annotation = annotation
        self.region_map = region_map
        self.layer_regexps = layer_regexps

    @cached_property
    def region(self) -> "VoxelData":
        """
        Accessor of the layered atlas as a VoxelData object.

        Returns:
            VoxelData instance of the layered atlas.
        """
        region_mask = get_region_mask(self.acronym, self.annotation.raw, self.region_map)
        return self.annotation.with_data(region_mask)

    @cached_property
    def volume(self) -> NDArray[int]:
        """
        Get the volume enclosed by the specified layers.

        Returns:
            layers_volume: numpy 3D array whose voxels are labelled
                by the indices of `self.layer_regexps` augmented by 1.
        """
        number_of_layers = len(self.layer_regexps)
        L.info(
            "Creating a volume for each of the %d layers of %s ...",
            number_of_layers,
            self.acronym,
        )

        return layers_volume(
            self.annotation.raw,
            self.region_map,
            layers=self.layer_regexps,
            region=self.region.raw,
        )

    def create_layer_meshes(self, layered_volume: NDArray[int]) -> List["trimesh.Trimesh"]:
        """
        Create meshes representing the upper boundary of each layer
        in the laminar region volume, referred to as `layered_volume`.

        Args:
            layered_volume: numpy 3D array whose voxels are labelled
                by the indices of `self.layer_regexps` augmented by 1.
        Returns:
            meshes: list of the layers meshes, together with the
                    mesh of the complement of the whole region.
                    Each mesh is used to define the upper boundary of
                    the corresponding layer. Meshes from the first to
                    the last layer have decreasing sizes: the first mesh encloses
                    all layers, the second mesh encloses all layers but the first
                    one, the second mesh encloses all layers but the first two,
                    and so on so forth.
                    The last mesh represents the bottom of the last layer.
                    It has the vertices of the first mesh, but its normal are
                    inverted.

        """

        layers_values = np.unique(layered_volume)
        layers_values = layers_values[layers_values > 0]
        assert len(layers_values) == len(
            self.layer_regexps
        ), "{} layer indices, {} layer strings".format(len(layers_values), len(self.layer_regexps))
        L.info(
            "Creating a watertight mesh for each of the %d layers of %s ...",
            len(layers_values),
            self.acronym,
        )
        meshes = []
        for index in tqdm(layers_values):
            mesh = create_watertight_trimesh(layered_volume >= index)
            meshes.append(mesh)

        L.info(
            "Trimming inward faces of the %d meshes of %s ...",
            len(meshes),
            self.acronym,
        )
        full_mesh_bottom = meshes[0].copy()
        # Inverting normals as we select the complement of the layered atlas
        full_mesh_bottom.invert()
        meshes.append(full_mesh_bottom)
        for i, mesh in tqdm(enumerate(meshes)):
            newmesh = centroid_outfacing_mesh(mesh)
            # This sometimes results in isolated faces which
            # cause ray intersection to fail.
            # So we trim them off by taking only the largest submesh.
            submeshes = newmesh.split(only_watertight=False)
            if len(submeshes) > 0:
                big_mesh = np.argmax([len(submesh.vertices) for submesh in submeshes])
                meshes[i] = submeshes[big_mesh]
            else:
                meshes[i] = mesh

        return meshes


def save_problematic_voxel_mask(layered_atlas: LayeredAtlas, problems: dict, output_dir: str):
    """
    Save the problematic voxel mask to file.

    The problematic volume is an array of the same shape as `layered_atlas.region`,
    i.e., (W, H, D). Its dtype is uint8.
    A voxel value equal to DistanceProblem.NO_PROBLEM indicates that no problem was detected.
    The value DistanceProblem.BEFORE_INTERPOLATION indicates that a problem was detected before
    interpolation of problematic distances by valid ones but not after.
    The value DistanceProblem.AFTER_INTERPOLATION indicates that a problem persists after
    interpolation.

    Args:
        layered_atlas: atlas for which distances computations have generated a problematic voxel
            mask.
        problems: dict returned by
            atlas_building_tools.compute_placement_hints.compute_placement_hints.
            This dictionary contains in particular 3D binary masks corresponding to voxels
            with problematic placement hints.
        output_dir: directory in which to save the problematic volume as an nrrd file.
    """
    problematic_volume_path = os.path.join(
        output_dir, layered_atlas.acronym + "_problematic_voxel_mask.nrrd"
    )
    L.info(
        "Saving problematic volume of %s to file %s ...",
        layered_atlas.acronym,
        problematic_volume_path,
    )
    voxel_mask = problems["before interpolation"]["volume"]
    problematic_volume = np.full(voxel_mask.shape, np.uint8(DistanceProblem.NO_PROBLEM.value))
    problematic_volume[voxel_mask] = np.uint8(DistanceProblem.BEFORE_INTERPOLATION.value)
    voxel_mask = problems["after interpolation"]["volume"]
    problematic_volume[voxel_mask] = np.uint8(DistanceProblem.AFTER_INTERPOLATION.value)

    layered_atlas.annotation.with_data(problematic_volume).save_nrrd(problematic_volume_path)


def _compute_dists_and_obtuse_angles(volume, layered_atlas, direction_vectors):
    layer_meshes = layered_atlas.create_layer_meshes(volume)
    # pylint: disable=fixme
    # TODO: compute max_smooth_error and use it as the value of rollback_distance
    # in the call of distances_from_voxels_to_meshes_wrt_dir()
    return distances_from_voxels_to_meshes_wrt_dir(volume, layer_meshes, direction_vectors)


def _dists_and_obtuse_angles(acronym, layered_atlas, direction_vectors, has_hemispheres=False):
    if not has_hemispheres:
        return _compute_dists_and_obtuse_angles(
            layered_atlas.volume, layered_atlas, direction_vectors
        )
    # Processing each hemisphere individually
    hemisphere_distances = []
    hemisphere_volumes = split_into_halves(layered_atlas.volume)
    hemisphere_obtuse_angles = []
    L.info(
        "Computing distances from voxels to layers meshes ...",
    )
    for hemisphere in [LEFT, RIGHT]:
        L.info(
            "Computing distances for the hemisphere %d of the %s region ...",
            hemisphere,
            acronym,
        )
        dists_to_layer_meshes, obtuse = _compute_dists_and_obtuse_angles(
            hemisphere_volumes[hemisphere], layered_atlas, direction_vectors
        )
        hemisphere_distances.append(dists_to_layer_meshes)
        hemisphere_obtuse_angles.append(obtuse)
    obtuse_angles = np.logical_or(hemisphere_obtuse_angles[LEFT], hemisphere_obtuse_angles[RIGHT])
    # Merging the distances arrays of the two hemispheres
    distances_to_layer_meshes = hemisphere_distances[LEFT]
    right_hemisphere_mask = hemisphere_volumes[RIGHT] > 0
    distances_to_layer_meshes[:, right_hemisphere_mask] = hemisphere_distances[RIGHT][
        :, right_hemisphere_mask
    ]
    return distances_to_layer_meshes, obtuse_angles


def compute_distances_to_layer_meshes(  # pylint: disable=too-many-arguments
    acronym: str,
    annotation: "VoxelData",
    region_map: "RegionMap",
    direction_vectors: NDArray[float],
    layer_regexps: List[str],
    has_hemispheres: bool = True,
    flip_direction_vectors: bool = False,
) -> Dict[str, Union[LayeredAtlas, NDArray[float], NDArray[bool]]]:
    """
    Compute distances from voxels to layers meshes wrt to direction vectors.

    Compute also the volume of voxels with problematic direction, i.e.,
    voxels for which no reliable distance information can be obtained.

    Args:
        acronym: name of the atlas region of interest, e.g., 'Isocortex'
        annotation: annotated 3D volume holding the full atlas.
        region_map: Object to navigate brain regions hierarchy.
        direction_vectors: unit vector field of shape (W, H, D, 3)
            if `annotation.raw`is of shape (W, H, D).
        layer_regexps: list of regular expressions representing
            the different layers of the region with the specified acronym.
            Layers must be ordered according to decreasing depths.
            The first layer is the upper layer, the last is the lower layer.
            The input direction vectors should flow from the lower layer to to the upper layer. If
            it is not the case, flip_direction_vectors should be set to True.
        has_hemispheres: True if the brain region of interest
            should be split in two hemispheres, False otherwise.
        flip_direction_vectors: True if the direction vectors should
            be reverted, False otherwise. This flag needs to be set to True
            depending on the algorithm used to generated orientation.nrrd.

    Returns:
        distances_info: dict with the following entries.
            layered_atlas: LayeredAtlas instance of the requested acronym.
            obtuse_angles: 3D boolean array indicating which voxels have rays
                intersecting a layer boundary with an obtuse angle. The direction vectors
                of such voxels are considered as problematic.
            distances_to_layer_meshes(numpy.ndarray): 4D float array of shape
                (number of layers + 1, W, H, D) holding the distances from
                voxel centers to the upper boundaries of layers wrt to voxel direction vectors.

    """
    if flip_direction_vectors:
        direction_vectors = -direction_vectors

    layered_atlas = LayeredAtlas(acronym, annotation, region_map, layer_regexps)
    distances_to_layer_meshes, obtuse_angles = _dists_and_obtuse_angles(
        acronym, layered_atlas, direction_vectors, has_hemispheres
    )
    L.info("Fixing disordered distances ...")
    fix_disordered_distances(distances_to_layer_meshes)

    return {
        "layered_atlas": layered_atlas,
        "distances_to_layer_meshes": distances_to_layer_meshes,
        "obtuse_angles": obtuse_angles,
    }
