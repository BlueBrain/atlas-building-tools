'''Function computing the direction vectors of the mouse cerebellum'''
import numpy as np  # type: ignore
from voxcell import VoxelData  # type: ignore
from atlas_building_tools.direction_vectors.algorithms.blur_gradient import (
    compute_initial_field,
    RegionShading,
)

import atlas_building_tools.direction_vectors.algorithms.blur_gradient as blur_gradient


def compute_direction_vectors(annotation):
    '''
    Computes cerebellum's direction vectors as the normalized gradient of a custom scalar field.

    The output direction vector field is computed as the normalized gradient
    of a custom scalar field. This scalar field resembles a distance field in
    the neighborhood of the molecular layer.

    Afterwards, a Gaussian filter is applied and the normalized gradient of the
    blurred scalar field is returned.

    Note: For now, direction vectors are only computed for the Flocculus and Lingula subregions.
        A voxel lying outside these two regions will be assigned a 3D vector
        with np.nan coordinates.

    Arguments:
        annotation(voxcell.VoxelData): VoxelData object holding
            the annotation volumetric array of the cerebellum.

    Returns:
        VoxelData object whose numpy.ndarray is of shape (annotation.shape, 1),
        and holds a 3D unit vector field. The voxel dimensions and
        the offset coincide with those of the input.
    '''

    # Flocculus
    flocculus_weights = {
        728: -5,  # Arbor vitae
        10690: -1,  # Flocculus, granular layer
        10691: 0,  # Flocculus, purkinje layer
        10692: 1,  # Flocculus molecular layer
        0: 3,  # outside the brain
    }
    # Shading applied from the molecular layer to the outside of Flocculus.
    shading_on_flocculus_complement = [
        RegionShading([728, 10691, 10690], 10692, 1, 4, invert=True)
    ]
    flocculus = [10690, 10691, 10692]
    flocculus_field = compute_initial_field(
        annotation.raw, flocculus_weights, shading_on_flocculus_complement
    )
    flocculus_direction_vectors = blur_gradient.compute_direction_vectors(
        annotation.raw, flocculus_field, flocculus
    )

    # Lingula
    lingula_weights = {
        744: -5,  # Cerebellar commissure
        10705: -1,  # Lingula, granular layer
        10706: 0,  # Lingula, purkinje layer
        10707: 1,  # Lingula molecular layer
    }
    # Shading applied from the molecular layer to the outside of Lingula.
    shading_on_lingula_complement = [
        RegionShading([728, 744, 10705, 10706], 10707, 1, 4, invert=True)
    ]
    lingula = [10690, 10691, 10692]
    lingula_field = compute_initial_field(
        annotation.raw, lingula_weights, shading_on_lingula_complement
    )
    lingula_direction_vectors = blur_gradient.compute_direction_vectors(
        annotation.raw, lingula_field, lingula
    )

    # Assembles Flocculus and Lingula direction vectors.
    direction_vectors = flocculus_direction_vectors
    lingula_mask = np.logical_not(np.isnan(lingula_direction_vectors))
    direction_vectors[lingula_mask] = lingula_direction_vectors[lingula_mask]

    return VoxelData(direction_vectors, annotation.voxel_dimensions, annotation.offset)
