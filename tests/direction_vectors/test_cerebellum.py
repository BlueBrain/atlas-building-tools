from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap, VoxelData

from atlas_building_tools.direction_vectors import cerebellum as tested
from atlas_building_tools.exceptions import AtlasBuildingToolsError

TEST_PATH = Path(Path(__file__).parent.parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))


@pytest.fixture
def region_map():
    return RegionMap.load_json(HIERARCHY_PATH)


@pytest.fixture
def annotation():
    """Cerebellum toy annotation.

    The lingula (left) is side to side with the flocculus (right).

    Above both there is void and bellow them fiber tracts. On
    the sides other regions have been places so that void does
    not affect the gradient.
    """
    raw = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
            ],
            [
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
            ],
        ],
        dtype=np.int32,
    )

    return VoxelData(raw, (25.0, 25.0, 25.0), offset=(1.0, 2.0, 3.0))


def _check_vectors_defined_in_regions(direction_vectors, region_map, annotation, acronyms):
    assert direction_vectors.shape == annotation.raw.shape + (3,)

    # The region of interest should not have nan value
    region_mask = np.isin(
        annotation.raw, tested._acronyms_to_flattened_identifiers(region_map, acronyms)
    )
    assert not np.isnan(direction_vectors[region_mask]).any()

    # Output direction vectors should by unit vectors
    npt.assert_allclose(np.linalg.norm(direction_vectors, axis=3)[region_mask], 1.0, atol=1e-6)

    # Outside the region of interest everything should be nan
    region_mask = np.isin(
        annotation.raw, tested._acronyms_to_flattened_identifiers(region_map, acronyms), invert=True
    )
    assert np.isnan(direction_vectors[region_mask]).all()


def _check_vectors_direction_dominance(
    direction_vectors, region_map, annotation, acronyms, direction
):

    region_mask = np.isin(
        annotation.raw, tested._acronyms_to_flattened_identifiers(region_map, acronyms)
    )

    region_vectors = direction_vectors[region_mask, :]

    # First check that all vectors are in the same halfspace as the direction
    cosines = region_vectors.dot(direction)
    assert np.all(cosines > 0.0), f"Angles to direction:\n{np.rad2deg(np.arccos(cosines))}"

    # Then check that they form an angle less that 45 degrees with the direction
    mask = np.arccos(cosines) < 0.25 * np.pi
    assert mask.all(), (
        f"Less than 45: {np.sum(mask)}, More than 45: {np.sum(~mask)}\n"
        f"Angles to direction:\n{np.rad2deg(np.arccos(cosines))}"
    )


def test_compute_cerebellum_direction_vectors(region_map, annotation):

    res = tested.compute_direction_vectors(region_map, annotation)
    _check_vectors_defined_in_regions(
        res, region_map, annotation, ["FLgr", "FLpu", "FLmo"] + ["LINGgr", "LINGpu", "LINGmo"]
    )

    _check_vectors_direction_dominance(
        res, region_map, annotation, ["FLgr", "FLpu", "FLmo"], [-1.0, 0.0, 0.0]
    )

    _check_vectors_direction_dominance(
        res, region_map, annotation, ["LINGgr", "LINGpu", "LINGmo"], [-1.0, 0.0, 0.0]
    )


def test_flocculus_direction_vectors(region_map, annotation):

    res = tested._flocculus_direction_vectors(region_map, annotation)
    _check_vectors_defined_in_regions(res, region_map, annotation, ["FLgr", "FLpu", "FLmo"])


def test_lingula_direction_vectors(region_map, annotation):

    res = tested._lingula_direction_vectors(region_map, annotation)
    _check_vectors_defined_in_regions(res, region_map, annotation, ["LINGgr", "LINGpu", "LINGmo"])


def test_region_direction_vectors(region_map, annotation):

    res = tested._region_direction_vectors(
        region_map,
        annotation,
        ["FLgr", "LINGgr"],
        {"FLgr": -1, "LINGgr": -2},
        boundary_region="FLgr",
    )

    _check_vectors_defined_in_regions(res, region_map, annotation, ["FLgr", "LINGgr"])


def test_acronyms_to_flattened_identifiers(region_map):

    region_acronyms = ["FLmo"]
    identifiers = tested._acronyms_to_flattened_identifiers(region_map, region_acronyms)
    npt.assert_array_equal(identifiers, [10692])

    region_acronyms = ["FLgr", "LINGmo"]
    identifiers = tested._acronyms_to_flattened_identifiers(region_map, region_acronyms)
    npt.assert_array_equal(identifiers, [10690, 10707])

    # resolve region and add children
    region_acronyms = ["LING"]
    identifiers = tested._acronyms_to_flattened_identifiers(region_map, region_acronyms)
    npt.assert_array_equal(identifiers, [912, 10705, 10706, 10707])

    region_acronyms = ["LING", "LINGmo"]
    identifiers = tested._acronyms_to_flattened_identifiers(region_map, region_acronyms)
    npt.assert_array_equal(identifiers, [912, 10705, 10706, 10707])

    region_acronyms = ["Giorgio"]
    with pytest.raises(AtlasBuildingToolsError):
        tested._acronyms_to_flattened_identifiers(region_map, region_acronyms)

    region_acronyms = ["FL", "LING", "sct"]
    identifiers = tested._acronyms_to_flattened_identifiers(region_map, region_acronyms)
    npt.assert_array_equal(identifiers, [85, 912, 1049, 10690, 10691, 10692, 10705, 10706, 10707])


def test_build_region_weight_map(region_map):

    region_to_weight = {"FLmo": -1}
    id_to_weight = tested._build_region_weight_map(region_map, region_to_weight)
    assert id_to_weight == {10692: -1}

    # Raise if an identifier is not found
    region_to_weight = {"FLmo": -1, "Giorgio": 9001}
    with pytest.raises(AtlasBuildingToolsError):
        tested._build_region_weight_map(region_map, region_to_weight)

    # Hierarchy resolution
    region_to_weight = {"FL": -1}
    id_to_weight = tested._build_region_weight_map(region_map, region_to_weight)
    assert id_to_weight == {1049: -1, 10690: -1, 10691: -1, 10692: -1}

    # Throw if an identifier is added because of a parent region adding its children and
    # the user having additionally specified a child region
    region_to_weight = {"FL": -1, "FLmo": -2}
    with pytest.raises(AtlasBuildingToolsError):
        tested._build_region_weight_map(region_map, region_to_weight)

    # Check the special outside_of_brain keyword
    region_to_weight = {"FLmo": -1, "outside_of_brain": 3}
    id_to_weight = tested._build_region_weight_map(region_map, region_to_weight)
    assert id_to_weight == {10692: -1, 0: 3}

    # and a full example
    region_to_weight = {"cbf": -5, "FLgr": -1, "FLpu": 0, "FLmo": 1, "outside_of_brain": 3}
    id_to_weight = tested._build_region_weight_map(region_map, region_to_weight)
    assert id_to_weight == {
        960: -5,
        326: -5,
        650: -5,
        78: -5,
        850: -5,
        404: -5,
        85: -5,
        728: -5,
        410: -5,
        866: -5,
        1123: -5,
        744: -5,
        553: -5,
        490: -5,
        812: -5,
        752: -5,
        499: -5,
        373: -5,
        10690: -1,
        10691: 0,
        10692: 1,
        0: 3,
    }
