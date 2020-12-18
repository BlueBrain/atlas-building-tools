'''
Utility functions for the computation of placement hints.
'''
from pathlib import Path
import logging


from typing import List, Optional, Union
from nptyping import NDArray  # type: ignore

import trimesh  # type: ignore
import numpy as np  # type: ignore

import voxcell  # type: ignore
from atlas_building_tools.utils import get_region_mask, is_obtuse_angle

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


Region = Union[str, voxcell.VoxelData, NDArray[int]]


def centroid_outfacing_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    '''
    Returns a mesh made of the faces that face away from `mesh`'s centroid.
    '''
    toward_centroid = mesh.centroid - mesh.triangles_center
    point_away_from_centroid = is_obtuse_angle(mesh.face_normals, toward_centroid)
    away_faces = mesh.faces[point_away_from_centroid]

    return trimesh.Trimesh(vertices=mesh.vertices, faces=away_faces)


def layers_volume(
    annotation: voxcell.VoxelData,
    region_map: Union[str, dict, 'voxcell.RegionMap'],
    layers: List[str],
    region: Optional[Region] = 'Isocortex',
) -> NDArray[int]:
    '''
    Labels a 3D volume using the indices of its `layers`.

    Arguments:
        annotation: whole brain annotation data.
        region_map: path to hierachy.json, dict instantiated from the latter
             or a RegionMap object.
        layers: the list of layer acronyms (or regexp)
        region: region to restrict to.

    Returns:
        A 3D volume whose labels are the indices of `layers`
        augmented by 1.
    '''
    if isinstance(region, str):
        region = get_region_mask(region, annotation, region_map)
    if isinstance(region, voxcell.VoxelData):
        region = region.raw

    result = 0
    for index, layer in enumerate(layers, 1):
        result += index * np.logical_and(
            get_region_mask(layer, annotation, region_map), region
        )
    return result


def save_placement_hints(
    distances: NDArray[float],
    output_dir: str,
    voxel_data: voxcell.VoxelData,
    layer_names: List[str],
):
    '''
    Convert distances to meshes wrt to direction vectors into placement hints
    and save these hints into a nrrd files, one for each layer.

    A placement hint of a voxel is a (lowest layer bottom)-to-layer distance wrt to
    the voxel direction vector. The last axis specifies if it is the distance
    to the layer bottom (0) or to the layer top (1).

    Args:
        distances: 4D array of shape (number-of-layers + 1, length, width, height)
            holding the signed distances from voxel centers to layer meshes wrt
            to voxel direction vectors.
        output_dir: directory in which to save the placement hints nrrd files.
        layer_names: list of layer names used to compose the placement hints file names.
    '''
    voxel_size = voxel_data.voxel_dimensions[1]  # voxel dimensions are assumed to be equal.
    # [PH]y holds, for each voxel, the distance from the bottom of the atlas region to the voxel
    # along its direction vector (non-negative value).
    y = -distances[-1]  # pylint: disable=invalid-name
    L.info('Saving placement hints [PH]y to file ...')
    placement_hints_y_path = str(Path(output_dir, '[PH]y.nrrd'))
    voxel_data.with_data(voxel_size * y).save_nrrd(placement_hints_y_path)
    L.info('Saving placement hints for each layer to file ...')
    for index, name in enumerate(layer_names):
        bottom = distances[index + 1]
        top = distances[index]
        # A placement hint array is a 1D array of size `number of layers` of float arrays of
        # shape (W, H, D, 2).
        # The description of a placement hint [PH]layer_i is quite convoluted.
        # (TODO: check if distances to boundaries could be used directly).
        # Given a voxel v, let L be the line passing through v along the direction vector of v. The
        # line L intersects the bottom of the deepest layer (e.g., layer 6 for the AIBS mouse
        # isocortex) in a voxel w. Then [PH]_layer_i.raw[..., 0] gives the (non-negative) distance
        # of w to the bottom of layer i while [PH]_layer_i.raw[..., 1] gives the (non-negative)
        # distance of w to the top of layer i.
        # Example: both arrays [PH]_layer_i.raw[...,0] and [PH]_layer_i.raw[..., 1] are constant
        # for O1 column atlases.
        # Note that [PH]_layer_i.raw[...,0] - [PH]y.raw holds the non-negative distances wrt to
        # direction vectors of all voxels to the bottom of layer i (these are non-positive values
        # for the voxels lying inside layer_i). [PH]_layer_i.raw[..., 1] - [PH]y.raw holds the
        # distances wrt to direction vectors of all voxels to the top of layer i (these are
        # non-negative values for voxels lying inside layer_i).
        placement_hints = np.stack((bottom, top), axis=-1) + y[..., np.newaxis]
        layer_placement_hints_path = str(Path(output_dir, '[PH]{}.nrrd'.format(name)))
        voxel_data.with_data(voxel_size * placement_hints).save_nrrd(
            layer_placement_hints_path
        )
