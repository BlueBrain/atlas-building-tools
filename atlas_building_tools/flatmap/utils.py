"""
Utils functions to flatten a laminar brain region.
"""
import logging
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from atlas_commons.typing import BoolArray  # type: ignore
from poisson_recon_pybind import Vector3dVector, create_poisson_surface

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import VoxelData  # type:  ignore

L = logging.getLogger(__name__)


def reconstruct_surface_mesh(volume: BoolArray, normals: "VoxelData") -> trimesh.Trimesh:
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
    L.info("Reconstructing the Poisson surface mesh (poisson_recon_pybind) ...")
    vertices, triangles = create_poisson_surface(Vector3dVector(points), Vector3dVector(normals))
    poisson_mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(triangles),
    )

    # Cleanup
    L.info("Cleaning up the resulting Poisson mesh ...")
    poisson_mesh.merge_vertices()
    poisson_mesh.process(validate=True)
    poisson_mesh.fill_holes()
    poisson_mesh.fix_normals()
    assert poisson_mesh.is_winding_consistent

    return poisson_mesh
