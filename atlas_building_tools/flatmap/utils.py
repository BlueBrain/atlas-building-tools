'''
Utils functions to flatten a laminar brain region.
'''
from typing import Set, TYPE_CHECKING
from nptyping import NDArray
import numpy as np
import open3d as o3d
import trimesh

from atlas_building_tools.exceptions import AtlasBuildingToolsError

if TYPE_CHECKING:
    from voxcell import RegionMap, VoxelData  # type:  ignore


def reconstruct_surface_mesh(
    volume: NDArray[bool], normals: 'VoxelData'
) -> trimesh.Trimesh:
    """
    Reconstruct a 3D surface mesh from a 3D binary image.

    The algorithm relies on open3d's implementation of Screened Poisson Surface Reconstruction, see
    http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.htm and reference
    article
    M. Kazhdan and H. Hoppe. 'Screened Poisson Surface Reconstruction.', 2013.

    Note: Voxels of the input volume need to be assigned normals.

    Args:
        volume: 3D boolean array of shape (W, H, D) representing the mask of the
            volume of interest.
        normals: VoxelData object holding a 3D unit vector field defined over
            a volume of shape (W, H, D). The input `volume` is assumed to sit in the same
            3D frame as the `normals`.

    Returns:
        surface mesh 'interpolating' the voxels of `volume`. Vertex coordinates are absolute.
        They are expressed in the 3D orthonormal frame of `normals`, i.e.,
        they incorporate the `normals` offset and voxel dimensions.


    Note: In the context of flat mapping, the input is a thin voxellized surface and the expected
        output mesh is a surface mesh with the topology of a 2D disk.
        The algorithm naturally creates a mesh with a relatively flat extent whose circular
        boundary is smooth. This property is desirable when subequently applying CGAL's authalic
        transformation. Experiments indicate that area distorsion is lower with a flat and smooth
        boundary.
    """
    points = normals.indices_to_positions(np.array(np.nonzero(volume)).T)
    normals = normals.raw[volume]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, scale=1.1
    )
    poisson_mesh = poisson_mesh.compute_vertex_normals()

    # Cleanup
    poisson_mesh = poisson_mesh.normalize_normals()
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_non_manifold_edges()

    return trimesh.Trimesh(
        vertices=np.asarray(poisson_mesh.vertices),
        faces=np.asarray(poisson_mesh.triangles),
    )


def create_layers_volume(
    annotated_volume: NDArray[int],
    region_map: 'RegionMap',
    metadata: dict,
    subregion_ids: Set[int] = None,
):
    """
    Create a 3D volume whose voxels are labeled by 1-based layer indices.

    Args:
        annotated_volume: integer numpy array of shape (W, H, D) where
            W, H and D are the integer dimensions of the volume domain.
        region_map: RegionMap object used to navigate the brain regions hierarchy.
        metadata: dict of the following form:
            {
                "layers": {
                    "names": ["layer 1", "layer 2", "layer3", "layer 4", "layer 5", "layer 6"],
                    "queries": ["@.*;L1$", "@.*;L2$", "@.;L3*$", "@.*;L4$", "@.*;L5$", "@.*;L6$"],
                    "attribute": "acronym"
                }
            }
            Queries in "queries" should be compliant with the interface of voxcell.RegionMap.find
            interface. The value of "attribute" can be "acronym" or "name".
            This object contains the definitions of the layers to be built.
        subregion_ids: (Optional) set of region identifiers used to restrict to input
            volume. The list defines a subregion of `annotated_volume` to be labeled with layer
            indices. Defaults to None, in which case the full `annotated_volume` is considered.

    Returns:
        A numpy array of the same shape as the input volume, i.e., (W, H, D). Voxels are labeled by
        the the 1-based indices of the layers defined in `metadata`. Voxels out of the (restricted)
        annotated volume are labeled with the 0 index.
    """

    if 'layers' not in metadata:
        raise AtlasBuildingToolsError('Missing "layers" key')

    metadata_layers = metadata['layers']

    missing = {'names', 'queries', 'attribute'} - set(metadata_layers.keys())
    if missing:
        err_msg = (
            'The "layers" dictionary has the following mandatory keys: '
            '"names", "queries" and "attribute".'
            f' Missing: {missing}.'
        )
        raise AtlasBuildingToolsError(err_msg)

    if not (
        isinstance(metadata_layers['names'], list)
        and isinstance(metadata_layers['queries'], list)
        and len(metadata_layers['names']) == len(metadata_layers['queries'])
    ):
        raise AtlasBuildingToolsError(
            'The values of "names" and "queries" must be lists of the same length'
        )

    layers = np.zeros_like(annotated_volume, dtype=np.uint8)
    for (index, query) in enumerate(metadata_layers['queries'], 1):
        layer_ids = region_map.find(
            query, attr=metadata_layers['attribute'], with_descendants=True
        )
        if subregion_ids is not None:
            layer_ids = layer_ids & subregion_ids
        layers[np.isin(annotated_volume, list(layer_ids))] = index

    return layers
