"""test annotations_combinator"""

import itertools
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
from PIL import Image  # type: ignore

import atlas_building_tools.cell_detection as tested

IMAGES_PATH = Path(Path(__file__).parent, "images")


def _save_float_array_as_greyscale(img: Image, filename: str):
    """
    Save a greyscale image array to png.

    Note: only used for debugging.
    """
    Image.fromarray((255 * img).astype(np.uint8), mode="L").save(str(filename), "PNG")


def test_svg_to_png():
    with TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir, "annotated_brain_slice.png"))
        tested.svg_to_png(Path(IMAGES_PATH, "image_with_strokes.svg"), output_path)
        png = np.asarray(Image.open(output_path).convert("RGB"))
        expected = np.asarray(
            Image.open(Path(IMAGES_PATH, "annotated_brain_slice.png")).convert("RGB")
        )
        npt.assert_array_equal(png, expected)

        output_path = str(Path(temp_dir, "annotated_brain_slice_without_strokes.png"))
        tested.svg_to_png(
            Path(IMAGES_PATH, "image_with_strokes.svg"),
            output_path,
            remove_strokes=True,
        )
        png = np.asarray(Image.open(output_path).convert("RGB"))
        expected = np.asarray(
            Image.open(Path(IMAGES_PATH, "annotated_brain_slice_without_strokes.png")).convert(
                "RGB"
            )
        )
        npt.assert_array_equal(png, expected)


def test_extract_color_map():
    color_map = tested.extract_color_map(Path(IMAGES_PATH, "annotated_slice.svg"))
    assert color_map == {"#bfdae3": 1, "#ffeae0": 2, "#b22aea": 3}


def test_jpg_to_greyscale():
    png = Image.open(Path(IMAGES_PATH, "mask.png"))
    greyscale = tested._jpg_to_greyscale(Path(IMAGES_PATH, "small_colored_spots.jpg"), png)
    expected = np.asarray(Image.open(Path(IMAGES_PATH, "greyscale.png")).convert("L"))
    npt.assert_array_almost_equal(greyscale, expected[:, :] / 255.0, decimal=2)


def test_remove_disk():
    jpg = np.asarray(Image.open(Path(IMAGES_PATH, "white_disks.jpg")).convert("L"))
    jpg = jpg[:, :].copy()
    radius = 75
    half_edge = 175
    for pair in itertools.product([-1, 0, 1], [-1, 0, 1]):
        tested.remove_disk(np.array(pair) * half_edge + half_edge, radius, jpg)
    assert np.all(jpg == 0)


def test_find_spots():
    jpg = np.asarray(Image.open(Path(IMAGES_PATH, "spots.jpg")).convert("L"))
    jpg = jpg[:, :].copy()
    jpg = jpg / np.max(jpg)
    spots = tested.find_spots(jpg, 25, 0.1)
    assert len(spots) == 21


def test_find_small_spots():
    jpg = np.asarray(Image.open(Path(IMAGES_PATH, "small_spots.jpg")).convert("L"))
    jpg = ~(jpg[:, :].copy())
    jpg = jpg / np.max(jpg)
    spots = tested.find_spots(jpg, 25, 0.1)
    assert len(spots) == 18


def test_compute_average_soma_radius():
    jpg_names = [Path(IMAGES_PATH, "small_spots.jpg")]
    average_small_radius = tested.compute_average_soma_radius(
        {"#ffffff": 1}, jpg_names, delta=25, max_radius=25
    )
    average_small_radius = float(average_small_radius[1])
    npt.assert_allclose(average_small_radius, 8.0, rtol=0.1)

    jpg_names = [Path(IMAGES_PATH, "big_spots.jpg")]
    average_big_radius = tested.compute_average_soma_radius(
        {"#ffffff": 1}, jpg_names, delta=25, max_radius=25
    )
    average_big_radius = float(average_big_radius[1])
    npt.assert_allclose(average_big_radius, 16.0, rtol=0.1)

    jpg_names = [Path(IMAGES_PATH, "mixed_spots.jpg")]
    average_mixed_radius = tested.compute_average_soma_radius(
        {"#ffffff": 1}, jpg_names, delta=25, max_radius=25
    )
    average_mixed_radius = float(average_mixed_radius[1])
    npt.assert_allclose(average_mixed_radius, 12.0, rtol=0.1)


def test_compute_average_soma_radius_with_color_mask():
    jpg_names = [Path(IMAGES_PATH, "mixed_spots_with_color_mask.jpg")]
    average_radii = tested.compute_average_soma_radius(
        {"#ff0000": 1, "#00ff00": 2}, jpg_names, delta=25, max_radius=25
    )
    average_red_radius = float(average_radii[1])
    npt.assert_allclose(average_red_radius, 12.8, rtol=0.1)
    average_green_radius = float(average_radii[2])
    npt.assert_allclose(average_green_radius, 12.0, rtol=0.1)


@patch("atlas_building_tools.cell_detection.FITTING_PARAMETERS", [10.0, 200.0, 0.0, 0.4])
def test_intensity_curve_fitting():
    x = np.arange(10)
    delta = 4
    curve = 200.0 * np.exp(-0.4 * (x - float(delta)) ** 2) + 10.0
    radius = tested.intensity_curve_fitting(curve, delta)
    assert radius == 1.0 / np.sqrt(0.4)

    x = np.arange(20)
    delta = 9
    curve = 100.0 * np.exp(-0.6 * (x - float(delta)) ** 2) + 5.0
    radius = tested.intensity_curve_fitting(curve, delta)
    npt.assert_array_almost_equal(radius, 1.0 / np.sqrt(0.6))

    x = np.arange(10)
    delta = 5
    curve = -100.0 * np.exp(-0.6 * (x - float(delta)) ** 2) + 5.0
    radius = tested.intensity_curve_fitting(curve, delta)
    assert np.isnan(radius)

    x = np.arange(10)
    delta = 5
    curve = 100.0 * np.exp(0.6 * (x - float(delta)) ** 2) + 5.0
    radius = tested.intensity_curve_fitting(curve, delta)
    assert np.isnan(radius)


@patch("atlas_building_tools.cell_detection.FITTING_PARAMETERS", [10.0, 200.0, 0.0, 0.4])
def test_intensity_curve_fitting_exceptions():
    curve = [-100.0, -2.0, -3.0, -4.0, -5.0]
    delta = 2
    with warnings.catch_warnings(record=True) as w:
        radius = tested.intensity_curve_fitting(curve, delta)
        assert "NaN radius is returned" in str(w[0].message)
        assert np.isnan(radius)

    x = np.arange(10)
    delta = 4
    curve = 200.0 * np.exp(-0.4 * (x - float(delta)) ** 2) + 10.0
    radius = tested.intensity_curve_fitting(curve, delta)
    assert radius == 1.0 / np.sqrt(0.4)
