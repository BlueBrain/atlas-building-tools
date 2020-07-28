'''test flatmap.utils'''
import numpy as np
import numpy.testing as npt
import pytest

from voxcell import RegionMap, VoxelData

from atlas_building_tools.exceptions import AtlasBuildingToolsError
import atlas_building_tools.flatmap.utils as tested


def get_hierarchy_excerpt():
    return {
        "id": 315,
        "acronym": "Isocortex",
        "name": "Isocortex",
        "children": [
            {
                "id": 500,
                "acronym": "MO",
                "name": "Somatomotor areas",
                "children": [
                    {
                        "id": 107,
                        "acronym": "MO1",
                        "name": "Somatomotor areas, Layer 1",
                        "children": [],
                    },
                    {
                        "id": 219,
                        "acronym": "MO2/3",
                        "name": "Somatomotor areas, Layer 2/3",
                        "children": [],
                    },
                    {
                        "id": 299,
                        "acronym": "MO5",
                        "name": "Somatomotor areas, layer 5",
                        "children": [],
                    },
                ],
            }
        ],
    }


def get_metadata():
    return {
        'layers': {
            'names': ['layer_1', 'layer_2_3', 'layer_5'],
            'queries': ['@.*1$', '@.*2/3$', '@.*5$'],
            'attribute': 'acronym',
        }
    }


def test_create_layers_volume():
    metadata = get_metadata()
    region_map = RegionMap.from_dict(get_hierarchy_excerpt())
    annotated_volume = np.array(
        [[[107, 107, 107, 219, 219, 219, 299, 299, 299]]], dtype=np.uint32
    )
    expected_layers_volume = np.array([[[1, 1, 1, 2, 2, 2, 3, 3, 3]]], dtype=np.uint32)

    # No subregion_ids
    actual = tested.create_layers_volume(annotated_volume, region_map, metadata)
    npt.assert_array_equal(expected_layers_volume, actual)

    # With subregion_ids
    subregion_ids = set({219, 299})
    expected_layers_volume = np.array([[[0, 0, 0, 2, 2, 2, 3, 3, 3]]], dtype=np.uint32)
    metadata['attribute'] = 'acronym'

    actual = tested.create_layers_volume(
        annotated_volume, region_map, metadata, subregion_ids
    )
    npt.assert_array_equal(expected_layers_volume, actual)

    # Testing assertions
    with pytest.raises(AtlasBuildingToolsError):
        metadata = get_metadata()
        del metadata['layers']
        tested.create_layers_volume(annotated_volume, region_map, metadata)

    with pytest.raises(AtlasBuildingToolsError):
        metadata = get_metadata()
        del metadata['layers']['attribute']
        tested.create_layers_volume(annotated_volume, region_map, metadata)

    with pytest.raises(AtlasBuildingToolsError):
        metadata = get_metadata()
        del metadata['layers']['names']
        tested.create_layers_volume(
            annotated_volume, region_map, metadata, subregion_ids
        )

    with pytest.raises(AtlasBuildingToolsError):
        metadata = get_metadata()
        del metadata['layers']['queries']
        tested.create_layers_volume(
            annotated_volume, region_map, metadata, subregion_ids
        )


def test_reconstruct_surface_mesh():
    volume = np.zeros((10, 10, 3), dtype=bool)
    volume[:, :, 1] = True
    direction_vectors = np.full((10, 10, 3, 3), np.nan, dtype=float)
    direction_vectors[:, :, 1] = [0.0, 0.0, 1.0]
    normals = VoxelData(
        direction_vectors, offset=[5.0, 10.0, 15.0], voxel_dimensions=[1.0, 2.0, 3.0]
    )
    reconstructed_mesh = tested.reconstruct_surface_mesh(volume, normals)
    points = normals.indices_to_positions(np.array(np.nonzero(volume)).T)
    for point in points:
        deltas = np.linalg.norm(point - reconstructed_mesh.vertices, axis=1)
        assert np.min(deltas) <= 0.4
