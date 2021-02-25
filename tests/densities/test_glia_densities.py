import numpy as np
from pathlib import Path
import numpy.testing as npt

from voxcell import RegionMap
import atlas_building_tools.densities.glia_densities as tested

TESTS_PATH = Path(__file__).parent.parent


def test_compute_glia_cell_counts_per_voxel():
    group_ids = {
        'Purkinje layer': set({1, 2}),
        'Fiber tracts group': set({3, 6}),
    }
    annotation = np.array([[[1, 10, 10, 2, 3]]], dtype=np.uint32)
    cell_density = np.array([[[0.1, 0.5, 0.75, 0.1, 1.0]]], dtype=float)
    glia_density = np.array([[[0.15, 0.1, 0.8, 0.2, 0.8]]], dtype=float)
    corrected_glia_density = tested.compute_glia_cell_counts_per_voxel(
        2, group_ids, annotation, glia_density, cell_density, copy=False
    )
    expected = np.array([[[0.0, 0.25, 0.75, 0.0, 1.0]]], dtype=float)
    npt.assert_allclose(corrected_glia_density, expected, rtol=1e-2)
    npt.assert_allclose(corrected_glia_density, glia_density, rtol=1e-2)

    # Same constraint, but a different input glia cell density
    # and copy is activated
    glia_density = np.array([[[0.15, 0.1, 0.2, 0.8, 0.8]]], dtype=float)
    glia_density_copy = glia_density.copy()
    corrected_glia_density = tested.compute_glia_cell_counts_per_voxel(
        2, group_ids, annotation, glia_density, cell_density
    )
    expected = np.array([[[0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 1.0]]], dtype=float)
    npt.assert_allclose(corrected_glia_density, expected, rtol=1e-2)
    npt.assert_allclose(glia_density_copy, glia_density, rtol=1e-2)


def test_glia_cell_counts_per_voxel_input():
    shape = (20, 20, 20)
    annotation = np.arange(8000).reshape(shape)
    cell_density = np.random.random_sample(annotation.shape).reshape(shape)
    cell_density = 50000 * cell_density / np.sum(cell_density)
    glia_cell_count = 25000
    glia_density = np.random.random_sample(annotation.shape).reshape(shape)
    group_ids = {
        'Purkinje layer': set({1, 2, 7, 11, 20, 25, 33, 200, 1000, 31, 16}),
        'Fiber tracts group': set({3, 6, 14, 56, 62, 88, 279, 2200, 5667, 7668}),
    }
    output_glia_density = tested.compute_glia_cell_counts_per_voxel(
        glia_cell_count,
        group_ids,
        annotation,
        glia_density,
        cell_density,
        copy=False,
    )
    assert np.all(output_glia_density <= cell_density)
    npt.assert_allclose(np.sum(output_glia_density), glia_cell_count, rtol=1e-3)


def test_compute_glia_densities():
    region_map = RegionMap.load_json(Path(TESTS_PATH, '1.json'))
    shape = (20, 20, 20)
    annotation = np.arange(8000).reshape(shape)
    cell_density = np.random.random_sample(annotation.shape).reshape(shape)
    voxel_volume = (25 ** 3) / 1e9
    cell_density = (50000 * cell_density / np.sum(cell_density)) / voxel_volume
    glia_proportions = {
        'astrocyte': '0.3',
        'oligodendrocyte': '0.5',
        'microglia': '0.2',
    }
    glia_densities = {
        glia_type: float(proportion)
        * np.random.random_sample(annotation.shape).reshape(shape)
        for glia_type, proportion in glia_proportions.items()
    }
    glia_densities['glia'] = np.random.random_sample(annotation.shape).reshape(shape)
    glia_cell_count = 25000
    output_glia_densities = tested.compute_glia_densities(
        region_map,
        annotation,
        voxel_volume,
        glia_cell_count,
        glia_densities,
        cell_density,
        glia_proportions,
        copy=True,
    )
    assert output_glia_densities['glia'].dtype == np.float64
    assert np.all(output_glia_densities['glia'] <= cell_density)

    glia_proportions['glia'] = '1.0'
    npt.assert_allclose(
        np.sum(output_glia_densities['glia']) * voxel_volume, glia_cell_count, rtol=1e-3
    )
    for glia_type, density in output_glia_densities.items():
        assert np.all(density <= output_glia_densities['glia'])
        npt.assert_allclose(
            np.sum(density) * voxel_volume,
            float(glia_proportions[glia_type]) * glia_cell_count,
            rtol=1e-3,
        )

    npt.assert_allclose(
        output_glia_densities['glia'],
        output_glia_densities['oligodendrocyte']
        + output_glia_densities['astrocyte']
        + output_glia_densities['microglia'],
        rtol=1e-3,
    )
    for glia_type in ['astrocyte', 'oligodendrocyte', 'microglia']:
        assert output_glia_densities[glia_type].dtype == np.float64
