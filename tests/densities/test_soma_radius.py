'''
Unit tests for densities utils
'''
import numpy as np
from pathlib import Path
import numpy.testing as npt

from voxcell import RegionMap
import atlas_building_tools.densities.soma_radius as tested

TESTS_PATH = Path(__file__).parent.parent


def test_apply_soma_area_correction():
    soma_radii = {1089: '0.5', 315: '0.7', 1009: '0.1'}
    annotation_raw = np.zeros((9, 1, 1), dtype=int)
    annotation_raw[0:2, 0, 0] = 1089  # Hippocampal formation
    annotation_raw[2:4, 0, 0] = 315  # Isocortex
    annotation_raw[4, 0, 0] = 1009  # Fiber tracts
    annotation_raw[5, 0, 0] = 885  # tn (fiber tracts)
    annotation_raw[6, 0, 0] = 949  # von (fiber tracts)
    annotation_raw[7:9, 0, 0] = 549  # Thalamus (no fiber tracts, no known radius)
    ara_nissl = np.zeros((9, 1, 1), dtype=float)
    ara_nissl[0:2, 0, 0] = 1.0
    ara_nissl[2:4, 0, 0] = 0.5
    ara_nissl[4, 0, 0] = 0.3
    ara_nissl[5, 0, 0] = 0.4
    ara_nissl[6, 0, 0] = 0.25
    ara_nissl[7:9, 0, 0] = 0.6
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    tested.apply_soma_area_correction(region_map, annotation_raw, ara_nissl, soma_radii)
    expected = np.copy(ara_nissl)
    expected[0:2, 0, 0] = 1.0 / (np.pi * 0.25)
    expected[2:4, 0, 0] = 0.5 / (np.pi * 0.49)
    expected[4, 0, 0] = 0.3 / (np.pi * 0.01)
    expected[5, 0, 0] = 0.4 / (np.pi * 0.01)
    expected[6, 0, 0] = 0.25 / (np.pi * 0.01)
    expected[7:9, 0, 0] = 0.6 / (
        np.pi * ((2.7 / 7.0) ** 2)
    )  # divides by average soma area
    npt.assert_array_almost_equal(ara_nissl, expected)
