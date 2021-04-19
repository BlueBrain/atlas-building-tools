import numpy as np
import numpy.testing as npt
import pytest as pyt
import trimesh
import voxcell

import atlas_building_tools.distances.distances_to_meshes as tested
from atlas_building_tools.exceptions import AtlasBuildingToolsError


class Test_distances_to_mesh_wrt_dir:
    def test_one_tri_boundary(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh, np.array([[0, 0, 0]]), one_mesh.face_normals
        )
        npt.assert_approx_equal(dist, np.array([np.linalg.norm([1 / 3, 1 / 3, 1 / 3])]))
        assert not wrong_side[0]

    def test_miss_boundary(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh, np.array([[0, 0, 0]]), np.array([[-1, 0, 0]])
        )
        assert dist.shape == (1,)
        assert np.isnan(dist)
        assert not wrong_side[0]

    def test_hits_vertex(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
            faces=np.array([[0, 1, 2], [1, 3, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh, np.array([[0, 0, 0]]), np.array([[0, 1, 0]])
        )
        npt.assert_array_equal(dist, [1])
        assert not wrong_side[0]

    def test_wrong_side_intersection(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh,
            np.array([[1, 1, 1], [0, 0, 0]]),
            np.array([-one_mesh.face_normals[0], one_mesh.face_normals[0]]),
        )
        npt.assert_array_equal(wrong_side, [True, False])

    def test_backward_intersection(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh, np.array([[1, 1, 1]]), one_mesh.face_normals, backward=True
        )
        npt.assert_array_almost_equal(dist, -np.linalg.norm([2 / 3, 2 / 3, 2 / 3]))
        npt.assert_array_equal(wrong_side, [False])

    def test_backward_and_wrong_side(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh,
            np.array([[1, 1, 1], [0, 0, 0]]),
            np.array([one_mesh.face_normals[0], -one_mesh.face_normals[0]]),
            backward=True,
        )
        npt.assert_array_almost_equal(
            dist,
            [
                -np.linalg.norm([2 / 3, 2 / 3, 2 / 3]),
                -np.linalg.norm([1 / 3, 1 / 3, 1 / 3]),
            ],
        )
        npt.assert_array_equal(wrong_side, [False, True])

    def test_nonunit_direction(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh, np.array([[0, 0, 0]]), 2 * one_mesh.face_normals
        )
        npt.assert_approx_equal(dist, np.array([np.linalg.norm([1 / 3, 1 / 3, 1 / 3])]))
        assert not wrong_side[0]

    def test_on_mesh_perpendicular_to_normal_vector(self):
        one_mesh = trimesh.Trimesh(
            vertices=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2]]),
        )
        dist, wrong_side = tested.distances_to_mesh_wrt_dir(
            one_mesh,
            np.array([[1 / 3, 1 / 3, 1 / 3]]),
            np.array([[np.sqrt(2), np.sqrt(2), 0]]),
        )
        npt.assert_almost_equal(dist, np.array([0]))
        assert not wrong_side[0]


class Test_distances_from_voxels_to_meshes_wrt_dir:
    def test_one_layer(self):
        test_layervol = np.array([[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]])
        bot_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 1, 0], [0, 1, 4], [1, 1, 0], [1, 1, 4]]),
            faces=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        top_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 2, 0], [0, 2, 4], [1, 2, 0], [1, 2, 4]]),
            faces=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        nanvec = [np.nan, np.nan, np.nan]
        upvec = [0, 1, 0]
        principal_axes = np.array(
            [
                [
                    [nanvec, nanvec, nanvec, nanvec],
                    [upvec, upvec, upvec, upvec],
                    [nanvec, nanvec, nanvec, nanvec],
                ]
            ]
        )

        distances, wrong_side = tested.distances_from_voxels_to_meshes_wrt_dir(
            test_layervol, [top_mesh, bot_mesh], principal_axes
        )

        dist_to_bottom_mesh = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [0.5, 0.5, 0.5, 0.5],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        dist_to_top_mesh = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [-0.5, -0.5, -0.5, -0.5],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        exp_distance = np.array([dist_to_bottom_mesh, dist_to_top_mesh])
        npt.assert_array_equal(distances, exp_distance)

        assert not np.any(wrong_side)

    def test_warns_nan(self):
        test_layervol = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [1, 1, 1, 1],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        bot_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 1, 0], [0, 1, 4], [1, 1, 0], [1, 1, 4]]),
            faces=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        top_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 2, 0], [0, 2, 4], [1, 2, 0], [1, 2, 4]]),
            faces=np.array([[0, 1, 2], [2, 1, 3]]),
        )
        nanvec = [np.nan, np.nan, np.nan]
        upvec = [0, 1, 0]
        principal_axes = np.array(
            [
                [
                    [nanvec, nanvec, nanvec, nanvec],
                    [upvec, upvec, upvec, upvec],
                    [nanvec, nanvec, nanvec, nanvec],
                ]
            ]
        )

        with pyt.warns(UserWarning, match="NaN direction vectors"):
            principal_axes[0, 1, 0] = nanvec
            tested.distances_from_voxels_to_meshes_wrt_dir(
                test_layervol, [top_mesh, bot_mesh], principal_axes
            )

    def test_other_side_mesh(self):
        """
        In the mesh optimization process, the mesh may shift enough
        at some points that voxel centers end up on the other side of
        the mesh from where they should be.
        the boundary algorithm should be robust to this, and treat such cases
        as having distance of 0 from the boundary.

        meshes roughly like this:

          .  .  .__.__.
          .  . /1  1  1
          .  1/ 1 _._/.
          . /1 _1/ 1  .
          .  .  .  .  .


        |  |  |__|__|__|
        |  |  /1 |1 |_1|
        |  |1/|1 | /|  |
        |  |/1|1_|/1|  |
        |  |  |  |  |  |

        """
        layervol = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        )

        def quad(v1, v2, v3, v4):
            """counterclockwise winding faces to make quad"""
            return [[v3, v2, v1], [v4, v3, v2]]

        top_mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [0, 1, 5],
                    [1, 1, 5],
                    [0, 1, 2],
                    [1, 1, 2],
                    [0, 3.5, 1.5],
                    [1, 3.5, 1.5],
                ]
            ),
            faces=np.concatenate([quad(0, 1, 3, 2), quad(2, 3, 5, 4)], axis=0),
        )

        bot_mesh = trimesh.Trimesh(
            vertices=np.array([[0, 2, 5], [1, 2, 5], [0, 4, 2], [1, 4, 2]]),
            faces=quad(0, 1, 3, 2),
        )

        up = [0, -1, 0]
        dup = [0, -np.sqrt(0.5), -np.sqrt(0.5)]
        nanvec = [np.nan, np.nan, np.nan]
        vectors = np.array(
            [
                [
                    [nanvec, nanvec, nanvec, nanvec, nanvec],
                    [nanvec, nanvec, dup, up, up],
                    [nanvec, dup, dup, nanvec, nanvec],
                    [nanvec, dup, up, up, nanvec],
                    [nanvec, nanvec, nanvec, nanvec, nanvec],
                ]
            ]
        )

        distances, something_wrong = tested.distances_from_voxels_to_meshes_wrt_dir(
            layervol, [top_mesh, bot_mesh], vectors
        )
        rt2 = np.sqrt(2)
        dist_to_bottom_mesh = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 1 / rt2, 0.5, 0.5],
                    [np.nan, 0.0, np.sqrt(8 / 9), np.nan, np.nan],
                    [np.nan, 0.0, 2.5, 2.5, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        dist_to_top_mesh = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, -np.sqrt(3.38), -1.5, -5 / 6],
                    [np.nan, -np.sqrt(2.42), -np.sqrt(49 / 50), np.nan, np.nan],
                    [np.nan, -1 / rt2, -1 / 6, 0.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        exp_distance = np.array([dist_to_bottom_mesh, dist_to_top_mesh])
        npt.assert_array_almost_equal(distances, exp_distance)
        assert not np.any(something_wrong)


class Test_fix_disordered_distances:
    """test the function to fix overlapped distances"""

    def test_no_overlap(self):
        distances = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])
        dist_bef = distances.copy()
        tested.fix_disordered_distances(distances)
        npt.assert_array_equal(distances, dist_bef)

    def test_overlaps(self):
        distances = np.array([[-1.0, 2.0, 0.0, 4.0, 5.0], [1.0, 3.0, 1.0, -4.0, -5.0]])
        exp_distances = np.array([[0, 2.5, 0.5, 4, 5], [0, 2.5, 0.5, -4, -5]])
        tested.fix_disordered_distances(distances)
        npt.assert_array_equal(distances, exp_distances)


class Test_report_distance_problems:
    """test the function that reports problems related to distances computations"""

    def test_report_output_format(self):
        raw = np.array(
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            ]
        )
        obtuse = np.full(raw.shape, False)
        obtuse[0][1][4] = True
        obtuse[0][3][1] = True
        voxel_data = voxcell.VoxelData(raw, voxel_dimensions=(1.0, 1.0, 1.0))
        rt2 = np.sqrt(2)
        dist_to_top_mesh = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 1 / rt2, 0.5, 0.5],
                    [np.nan, 0.0, np.sqrt(8 / 9), np.nan, np.nan],
                    [np.nan, 0.0, 2.5, 2.5, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        dist_to_bottom_mesh = np.array(
            [
                [
                    [0.0, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, -np.sqrt(3.38), -1.5, -5 / 6],
                    [np.nan, -np.sqrt(2.42), -np.sqrt(49 / 50), np.nan, np.nan],
                    [np.nan, -1 / rt2, -1 / 6, 0.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ]
        )
        distances = np.array([dist_to_top_mesh, dist_to_bottom_mesh])
        report, problematic_volume = tested.report_distance_problems(
            distances, obtuse, voxel_data, max_thicknesses=[0.5], tolerance=2.0
        )
        assert (
            report["Proportion of voxels whose rays do not intersect with the bottom mesh"] == 0.1
        )
        assert report["Proportion of voxels whose rays do not intersect with the top mesh"] == 0.2
        assert (
            report[
                "Proportion of voxels whose rays make an obtuse angle with the mesh normal at the intersection point"
            ]
            == 0.2
        )
        assert (
            report[
                "Proportion of voxels with a distance gap greater than the maximum thickness "
                "(NaN distances are ignored)"
            ]
            == 0.2
        )
        assert report["Proportion of voxels with at least one distance-related problem"] == 0.6
        expected_problematic_volume = [
            [
                [True, False, False, False, False],
                [False, False, True, False, True],
                [False, False, False, False, False],
                [False, True, True, False, False],
                [False, False, False, False, True],
            ]
        ]
        npt.assert_array_equal(problematic_volume, expected_problematic_volume)


class Test_interpolate_volume:
    def test_no_known_values(self):
        with pyt.raises(AtlasBuildingToolsError):
            tested.interpolate_volume([[[1, 2, 3, 4]]], [[[1, 0, 1, 1]]], [[[0, 0, 0, 0]]])

    def test_no_target_values(self):
        npt.assert_array_equal(
            tested.interpolate_volume([[[1, 2, 3, 4]]], [[[0, 0, 0, 0]]], [[[1, 1, 1, 0]]]),
            [],
        )

    def test_known_and_target_uses_nearest(self):
        npt.assert_array_equal(
            tested.interpolate_volume([[[1, 2, 3, 4]]], [[[0, 1, 1, 0]]], [[[1, 0, 0, 1]]]),
            [1, 4],
        )

    def test_voxel_in_known_and_target_is_unchanged(self):
        npt.assert_array_equal(
            tested.interpolate_volume([[[1, 2, 1, 1]]], [[[0, 1, 0, 0]]], [[[0, 1, 0, 0]]]),
            [2],
        )


class Test_interpolate_problematics_voxels:
    """test the function that corrects problematic voxels by interpolateing"""

    def test_no_problematic_voxels_raise(self):
        distance = np.array([[[[np.nan, 0, 1, np.nan]]]])
        with pyt.raises(AtlasBuildingToolsError):
            tested.interpolate_problematic_voxels(
                distance,
                np.array([[[0, 0, 0, 0]]]).astype(bool),
            )

    def test_nans_in_mask_corrected_with_neighbors(self):
        distance = np.array([[[[10000, -1, np.nan, -1]]]])
        tested.interpolate_problematic_voxels(
            distance,
            np.array([[[0, 1, 1, 1]]]).astype(bool),
        )
        npt.assert_array_equal(distance, np.array([[[[10000, -1, -1, -1]]]]))

    def test_invalid_non_nan_in_mask_corrected_with_neighbors(self):
        distance = np.array([[[[100, -1, 9000, -1]]], [[[100, -1, -9000, 1]]]])
        tested.interpolate_problematic_voxels(
            distance, np.array([[[0, 1, 1, 1]]]).astype(bool), max_thicknesses=[900]
        )
        npt.assert_array_equal(distance, np.array([[[[100, -1, -1, -1]]], [[[100, -1, -9000, 1]]]]))
