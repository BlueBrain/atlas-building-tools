"""Functions to compute the average soma radii of different brain regions."""

from typing import TYPE_CHECKING, Dict

import numpy as np
from nptyping import NDArray  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap  # type: ignore


def _compute_whole_brain_average_soma_radius(
    region_map: "RegionMap", annotation_raw: NDArray[float], soma_radii: Dict[int, str]
) -> float:
    """
    Compute the whole-brain average soma radius weighted by region volumes

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
            Instantiated with AIBS 1.json.
        annotation_raw: integer array of shape (W, H, D) holding the annotation
            of the whole mouse brain.
        soma_radii: dict whose keys are region identifiers (AIBS structure IDs) and whose values
            are the regions average soma radii.

    Returns:
        whole-brain average soma radius weighted by region volumes.

    """
    ids = (
        list(region_map.find(id_, attr="id", with_descendants=True)) for id_ in soma_radii.keys()
    )
    volumes = np.array([np.count_nonzero(np.isin(annotation_raw, id_group)) for id_group in ids])
    radii = np.array([float(radius) for radius in soma_radii.values()])
    return np.average(radii, weights=volumes)


def apply_soma_area_correction(
    region_map: "RegionMap",
    annotation_raw: NDArray[float],
    nissl: NDArray[float],
    soma_radii: Dict[int, str],
) -> None:
    """
    Apply in-place an intensity correction based on soma area estimates.

    The image intensity of the 3D Nissl stained volume is re-scaled.
    The scaling factor for a region is the inverse of the average soma area estimate
    if such an estimate is available. Otherwise this is the average
    of all available estimates weighted by the region volumes.

    The rational behing this correction is that the Nissl staining intensity,
    interpreted as a cell density, is overestimated in regions where the somata
    are large and underestimated where somata are small.

    From "A Cell Atlas for the Mouse Brain" by C. Eroe et al., 2018:
    "Our spatial density estimation assumed a constant cell body size throughout the brain, which
     is never the case in reality. In an effort to compensate for this effect, we extended our
      point-detection algorithm to further estimate the average soma size as well in each region
      (Figure 7E). This was done by fitting the image of each detected soma by a 2-dimensional
      circular Heaviside step function, until the best matching radius was found."

    Args:
        region_map: object to navigate the mouse brain regions hierarchy.
            Instantiated from AIBS 1.json.
        annotation_raw: integer array of shape (W, H, D) holding the annotation
            of the whole mouse brain.
        nissl: float array holding the intensity of Nissl stains in the whole mouse
             brain.
        dict whose keys are region identifiers (AIBS structure IDs) and whose values
            are the regions average soma radii.
    """

    mean_radius = _compute_whole_brain_average_soma_radius(region_map, annotation_raw, soma_radii)
    weights = np.full(annotation_raw.shape, np.pi * (mean_radius ** 2))
    for id_, radius in soma_radii.items():
        weights[annotation_raw == id_] = np.pi * (float(radius) ** 2)

    # In CCFv2 2011 and in the images used to estimate the soma radii, the fiber tracts are
    # assigned a unique 'root' identifier, that is 1009 (from 1.json as of 2020/06). This is
    # why we use a uniform soma area for all subregions of the fiber tracts.
    fiber_tracts_id = list(region_map.find("fiber tracts", attr="acronym"))[0]
    if fiber_tracts_id in soma_radii:
        fiber_tracts_ids = list(
            region_map.find("fiber tracts", attr="acronym", with_descendants=True)
        )
        radius_ = float(soma_radii[fiber_tracts_id])
        weights[np.isin(annotation_raw, fiber_tracts_ids)] = np.pi * (radius_ ** 2)

    nissl /= weights
