import numpy as np
from pathlib import Path
import numpy.testing as npt

from voxcell import RegionMap  # type: ignore
import atlas_building_tools.densities.glia_densities as tested

TESTS_PATH = Path(__file__).parent.parent


def test_compute_glia_density():
    group_ids = {
        'Purkinje layer': set({1, 2}),
        'Fiber tracts group': set({3, 6}),
    }
    annotation_raw = np.array([[[1, 10, 10, 2, 3]]], dtype=np.uint32)
    cell_density = np.array([[[0.1, 0.5, 0.75, 0.1, 1.0]]], dtype=float)
    glia_density = np.array([[[0.15, 0.1, 0.8, 0.2, 0.8]]], dtype=float)
    corrected_glia_density = tested.compute_glia_density(
        2, group_ids, annotation_raw, glia_density, cell_density, copy=False
    )
    expected = np.array([[[0.0, 0.25, 0.75, 0.0, 1.0]]], dtype=float)
    npt.assert_almost_equal(corrected_glia_density, expected, decimal=5)
    npt.assert_almost_equal(corrected_glia_density, glia_density)

    # Same constraint, but a different input glia cell density
    # and copy is activated
    glia_density = np.array([[[0.15, 0.1, 0.2, 0.8, 0.8]]], dtype=float)
    glia_density_copy = glia_density.copy()
    corrected_glia_density = tested.compute_glia_density(
        2, group_ids, annotation_raw, glia_density, cell_density
    )
    expected = np.array([[[0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 1.0]]], dtype=float)
    npt.assert_almost_equal(corrected_glia_density, expected, decimal=5)
    npt.assert_almost_equal(glia_density_copy, glia_density)


def test_compute_glia_density_large_input():
    shape = (20, 20, 20)
    annotation_raw = np.arange(8000).reshape(shape)
    cell_density = np.random.random_sample(annotation_raw.shape).reshape(shape)
    cell_density = 50000 * cell_density / np.sum(cell_density)
    glia_cell_count = 25000
    glia_density = np.random.random_sample(annotation_raw.shape).reshape(shape)
    group_ids = {
        'Purkinje layer': set({1, 2, 7, 11, 20, 25, 33, 200, 1000, 31, 16}),
        'Fiber tracts group': set({3, 6, 14, 56, 62, 88, 279, 2200, 5667, 7668}),
    }
    output_glia_density = tested.compute_glia_density(
        glia_cell_count,
        group_ids,
        annotation_raw,
        glia_density,
        cell_density,
        copy=False,
    )
    assert np.all(output_glia_density <= cell_density)
    npt.assert_almost_equal(np.sum(output_glia_density), glia_cell_count)
    print(np.sum(output_glia_density))


def test_compute_glia_densities():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    shape = (20, 20, 20)
    annotation_raw = np.arange(8000).reshape(shape)
    cell_density = np.random.random_sample(annotation_raw.shape).reshape(shape)
    cell_density = 50000 * cell_density / np.sum(cell_density)
    glia_proportions = {
        'astrocyte': '0.3',
        'oligodendrocyte': '0.5',
        'microglia': '0.2',
    }
    glia_densities = {
        glia_type: float(proportion)
        * np.random.random_sample(annotation_raw.shape).reshape(shape)
        for glia_type, proportion in glia_proportions.items()
    }
    glia_densities['glia'] = np.random.random_sample(annotation_raw.shape).reshape(
        shape
    )
    glia_cell_count = 25000
    output_glia_densities = tested.compute_glia_densities(
        region_map,
        annotation_raw,
        glia_cell_count,
        cell_density,
        glia_densities,
        glia_proportions,
        copy=True,
    )

    assert np.all(output_glia_densities['glia'] <= cell_density)

    glia_proportions['glia'] = '1.0'
    npt.assert_almost_equal(np.sum(output_glia_densities['glia']), glia_cell_count)
    for glia_type, density in output_glia_densities.items():
        assert np.all(density <= output_glia_densities['glia'])
        npt.assert_almost_equal(
            np.sum(density), float(glia_proportions[glia_type]) * glia_cell_count
        )

    npt.assert_almost_equal(
        output_glia_densities['glia'],
        output_glia_densities['oligodendrocyte']
        + output_glia_densities['astrocyte']
        + output_glia_densities['microglia'],
    )