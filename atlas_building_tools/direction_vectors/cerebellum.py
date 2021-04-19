"""Function computing the direction vectors of the AIBS mouse cerebellum

The algorithm creates a scalar field with low values in surfaces where fiber tracts are incoming
and high values where fiber tracts are outgoing. The direction vectors are given by the gradient
of this scalar field.

Note: At the moment, direction vectors are generated only for the following cerebellum subregions:
    - the flocculus
    - the lingula
"""
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore

import atlas_building_tools.direction_vectors.algorithms.blur_gradient as blur_gradient
from atlas_building_tools.direction_vectors.algorithms.blur_gradient import (
    RegionShading,
    compute_initial_field,
)


def compute_direction_vectors(annotation_raw: NDArray[float]) -> NDArray[float]:
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
        annotation: integer array of shape (W, H, D) holding the annotation of the whole mouse
         brain.

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """

    # Flocculus
    flocculus_weights = {
        728: -5,  # Arbor vitae
        10690: -1,  # Flocculus, granular layer
        10691: 0,  # Flocculus, purkinje layer
        10692: 1,  # Flocculus molecular layer
        0: 3,  # outside the brain
    }
    # Shading applied from the molecular layer to the outside of flocculus.
    shading_on_flocculus_complement = [RegionShading([728, 10691, 10690], 10692, 1, 4, invert=True)]
    flocculus = [10690, 10691, 10692]
    flocculus_field = compute_initial_field(
        annotation_raw, flocculus_weights, shading_on_flocculus_complement
    )
    flocculus_direction_vectors = blur_gradient.compute_direction_vectors(
        annotation_raw, flocculus_field, flocculus
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
    lingula = [10705, 10706, 10707]
    lingula_field = compute_initial_field(
        annotation_raw, lingula_weights, shading_on_lingula_complement
    )
    lingula_direction_vectors = blur_gradient.compute_direction_vectors(
        annotation_raw, lingula_field, lingula
    )

    # Assembles flocculus and lingula direction vectors.
    direction_vectors = flocculus_direction_vectors
    lingula_mask = np.logical_not(np.isnan(lingula_direction_vectors))
    direction_vectors[lingula_mask] = lingula_direction_vectors[lingula_mask]

    return direction_vectors
