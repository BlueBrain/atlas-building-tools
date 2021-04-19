"""
Cell detection module.

Functions to detect cell somata and to estimate somata diameters by processing 2D images.

Greyscale 2D images are provided as input. (Greyscale images are obtained by inverting the Nissl
stain intensity of the images provided by AIBS.) In these images, somata are represented by spots
of high pixel intensity. Somata centers are identified as local maximal pixel intensity. Somata are
incrementaly removed from an image by cutting a disk of constant radius around soma centers. This
radius is determined experimentally.

The mapping which assigns a soma center to an AIBS brain region is computed by means of companion
svg files which annotate each greyscale image.
"""
import logging
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from xml.dom import minidom  # type: ignore

import numpy as np  # type: ignore
from cairosvg import svg2png  # type: ignore
from nptyping import NDArray  # type: ignore
from PIL import Image  # type: ignore
from scipy.ndimage import filters  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from skimage import morphology  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


L = logging.getLogger(__name__)
logging.captureWarnings(True)

FITTING_PARAMETERS = [10.0, 200.0, 0.0, 0.4]  # determined experimentally


def remove_disk(coord: NDArray[int], radius: int, image: NDArray[float]) -> None:
    """
    Remove a disk of radius `radius` centered at `coord` from the input 2D `image`.

    The color of pixels lying inside the disk of radius `radius` centered at `coord`
    is set to black, i.e., set with the integer value 0.

    Args:
        coord: 1D array of shape (2,), i.e. a 2D vector (x, y) representing the disk center
            coordinates.
        radius: radius of the disk to be removed in number of pixels.
        image: 2D integer array.
    """
    bottom = np.max([coord - radius, [0, 0]], axis=0)
    top = np.min([coord + radius, np.array(image.shape) - 1], axis=0) + 1
    disk_bottom = bottom - coord + radius
    disk_top = top - coord + radius
    disk_mask = morphology.disk(radius).astype(bool)
    disk_mask = disk_mask[disk_bottom[0] : disk_top[0], disk_bottom[1] : disk_top[1]]
    image[bottom[0] : top[0], bottom[1] : top[1]][disk_mask] = 0


def find_spots(image: NDArray[float], radius: int, epsilon: float = 0.1) -> NDArray[int]:
    """
    Find all the light spots of radius <= `radius` and whose intensity exceeds `epsilon`.

    A spot is centered at a pixel which maximizes the image intensity.
    Once a spot has been detected, a disk of radius `radius` around its center is removed.
    This corresponds to a black disk hidding the former spot.

    Args:
        image: 2D float array, greyscale with all values in [0, 1].
        radius: radius of the disk removed from the image after each spot detection.
            The radius is expressed in number of pixels.
        epsilon: intensity threshold below which the image is considered spot-less, i.e.,
            fully black.

    Returns:
        An integer array of shape (N, 2) containing the 2D integer coordinates of the N detected
        spots.

    """
    spots = []
    image = image.copy()
    while True:
        argmax = np.unravel_index(np.argmax(image), image.shape)
        if image[argmax] < epsilon:
            break
        spots.append(argmax)
        remove_disk(np.array(argmax), radius, image)

    return np.array(spots)


def intensity_curve_fitting(curve: NDArray[float], delta: int) -> float:
    """
    Fit a bell-shaped curve to a function reaching its maximum at `delta`.

    Args:
        curve: 1D float array interpreted as the graph curve of an intensity function.
            The curve is assumed to reach its maximum at x = `delta`.
        delta: optimal parameter value of the graph curve.

    Returns:
        An optimal value of the parameter sigma used to defined the bell-shaped curve.
        It represents the width of the bell-shaped curve, just like the standard deviation
        represents the width of the graph curve of a Gaussian kernel.
        If the optimizatio process failed (scipy's RuntimeError) or if the returned optimal
        parameters are invalid (say, a non-positive radius), the return value is np.nan.

    """
    # pylint: disable=invalid-name
    def bell_shape(X: float, A: float, B: float, C: float, sigma: float) -> float:
        """
        Bell-shape curve generalizing the graph curve of a Gaussian kernel.
        """
        return A + B * np.exp(-sigma * (X - float(delta)) ** 2) + C * X

    try:
        optimal_parameters = curve_fit(
            bell_shape,
            np.arange(len(curve)),
            curve,
            p0=np.array(FITTING_PARAMETERS),
        )
    except RuntimeError as error:
        warnings.warn(f"Curve fitting failed: {error}.\n" " NaN radius is returned.", UserWarning)
        return np.nan

    sigma = optimal_parameters[0][3]
    if optimal_parameters[0][1] > 0.0 and sigma > 0.0:
        return 1.0 / np.sqrt(sigma)

    return np.nan


def _compute_spot_radius(spot: NDArray[int], greyscale: NDArray[float], delta: int) -> float:
    """
    Compute the radius of a light spot based on intensity curve fitting.

    Spots are assumed to be round-shaped, centered around pixels with maximal intensity.
    This function returns an estimate of a spot radius based on a fitting with a Gaussian-like
    bell-shaped curve.

    A window of radius `delta` is extracted around the spot center.
    We consider then two curves, an horizontal section (i.e., along x) and a vertical section
    (i.e., along y) of the intensity surface of the spot above the selected window.

    Args:
        spot: 1D array of shape (2,), i.e., a 2D vector (x, y) representing the spot center
            coordinates.
        greyscale: 2D float array with values in [0, 1].
        delta: radius of the interval around the spot center which is used to define
            an horizontal and a vertical intensity curve. `delta`is expressed in number of pixels.

    Returns:
        float, an estimate of the spot radius.
        Note: if the optimization process failed to return a sensible radius value, the returned
        radius is np.nan.
    """
    bottom = np.max([spot - delta, [0, 0]], axis=0)
    top = np.min([spot + delta + 1, np.array(greyscale.shape) - 1], axis=0) + 1
    window = greyscale[bottom[0] : top[0], bottom[1] : top[1]]
    centred_spot = spot - bottom
    return float(
        np.mean(
            [
                intensity_curve_fitting(window[:, centred_spot[1]], centred_spot[0]),
                intensity_curve_fitting(window[centred_spot[0], :], centred_spot[1]),
            ]
        )
    )


def _jpg_to_greyscale(
    jpg_filename: Union[str, "Path"],
    png: Image,
    gaussian_filter_stddev: float = 1.5,
    intensity_threshold: float = 0.1,
) -> NDArray[float]:
    """
    Convert a jpg image to a greyscale 2D array.

    The function applies the standard RGB-to-greyscale conversion of PIL,
    that is the ITU-R 601-2 luma transform: L = R * 299/1000 + G * 587/1000 + B * 114/1000,
    see https://pillow.readthedocs.io/en/3.1.x/reference/Image.html

    In addition:

    * The `png` companion image is used as a stencil to exclude specific parts of the image.
    * The intensity of the image is inverted so that high intensity values represent cells (white)
        whereas low values represent the background (black).
    * A Gaussian blur with standard deviation `gaussian_filter_stddev` is applied to the image.
    * Image intensity below `intensity_threshold` is zeroed.

    Args:
        jpg_filename: file name of the input RGB jpg file.
        png: companion image used to exclude part of the input.
        gaussian_filter_stddev: standard deviation of the Gaussian blur which is applied to the
            input image.
        intensity_threshold: threshold below which the intensity of the image is zeroed.

    Returns:
        A 2D greyscale image.

    """
    greyscale = np.asarray(Image.open(str(jpg_filename)).convert("L"))
    greyscale = ~greyscale.copy()
    png = np.asarray(png.convert("L"))
    greyscale[png == 0.0] = 0.0  # removes out-of-brain intensity
    greyscale = filters.gaussian_filter(greyscale.astype(float), sigma=gaussian_filter_stddev)
    greyscale = greyscale / np.max(greyscale)
    greyscale[greyscale < intensity_threshold] = 0.0

    return greyscale


def compute_average_soma_radius(
    color_map: Dict[str, int],
    jpg_filenames: Union[List[str], List["Path"]],
    delta: int = 16,
    max_radius: int = 10,
    intensity_threshold: float = 0.1,
) -> Dict[int, str]:
    """
    Compute the average soma radii of each brain region represented in the input jpg images.

    The algorithm first detects light spots based on maximal values of the image intensity.
    Spot radii are then estimated by means of curve fitting with a bell-shaped curve similar to
    a Gaussian kernel. NaN radius values returned by the curve fitting are skipped.

    The regions where the spots originate from are recognized thanks to the companion png annotated
    files and the pre-computed `color_map` which maps colors to AIBS structure IDs (integer region
    identifiers from AIBS 1.json).

    Note: The default values of
        * `delta`
        * `max_radius`
        * `intensity_threshold`
    have been found by trials and errors.

    Args:
        color_map: dict mapping hexadecimal color code (str) to AIBS structure IDs, i.e.,
             brain region identifiers. See definitions and 1.json source file here
             http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies.
        jpg_filenames: List of the file names containing brain section images with Nissl stains.
        delta: witdh of the window surrounding a spot when performing curve fitting. This width is
            expressed in number of pixels.
        max_radius: radius of the disk around a spot center that is removed once the spot has been
            detected. This radius is expressed in number of pixels.
        intensity_threshold: threshold below which the intensity of the image is zeroed.
    """
    jpg_filenames.sort()
    soma_radius_dict = defaultdict(list)
    for jpg_filename in jpg_filenames:
        L.info("Processing file %s", jpg_filename)
        png = Image.open(str(jpg_filename).replace(".jpg", ".png"))
        greyscale = _jpg_to_greyscale(jpg_filename, png, 1.5, intensity_threshold)
        L.info("Detecting spots of maximal intensity ...")
        spots = find_spots(greyscale, max_radius, epsilon=intensity_threshold)
        L.info("Computing spot radii ...")
        png = np.asarray(png)
        for spot in spots:
            radius: float = _compute_spot_radius(spot, greyscale, delta)
            if not np.isnan(radius):
                rgb = tuple(png[tuple(spot)])
                hexa_color_code = "#%02x%02x%02x" % rgb[:3]
                # The svg-to-png conversion has created unregistered shades on region boundaries.
                # These new colors without entries in the color map represent < 2 percents
                # of the colored pixels. We just skip them if they happen to be spot centers.
                if hexa_color_code in color_map:
                    structure_id = color_map[hexa_color_code]
                    soma_radius_dict[structure_id].append(radius)

    return {
        int(structure_id): repr(np.mean(radius_list))
        for structure_id, radius_list in soma_radius_dict.items()
    }


def svg_to_png(
    svg_filename: Union[str, "Path"],
    output_name: Optional[Union[str, "Path"]] = None,
    remove_strokes: bool = False,
) -> None:
    """
    Convert a svg file to a png file using cairosvg.svg2png.

    Strokes are optionally removed.

    Args:
        svg_filename: name of the file to convert (svg).
        output_name: name of the file to save (png).
        remove_strokes: Optional. If True, black are removed from the input svg file before
            conversion.
    """
    svg_str = ""
    with open(svg_filename, "r") as file_:
        svg_str = file_.read()
        if remove_strokes:
            for regexp in [
                r'stroke:[^;"]+;',
                r'stroke-opacity:[^;"]+;',
                r'stroke-width:[^;"]+;',
                r'(stroke:[^;"]+)"',
                r'(stroke-opacity:[^;"]+)";',
                r'(stroke-width:[^;"]+)"',
            ]:
                svg_str = re.sub(regexp, "", svg_str, flags=re.MULTILINE)

    if output_name is None:
        output_name = str(svg_filename).replace(".svg", ".png")

    svg2png(bytestring=bytes(svg_str, "UTF-8"), write_to=open(str(output_name), "wb"))


def extract_color_map(svg_filename: Union[str, "Path"]) -> Dict[str, int]:
    """
    Extract from svg file the map binding hexadecimal colors to AIBS structure IDs.

    Args:
        svg_filename: name of the file used for map extraction.
            Must be an AIBS svg annotation file.
            See e.g., http://help.brain-map.org/display/api/Downloading+and+Displaying+SVG.

    Returns:
        A dict whose keys are hexadecimal strings of length 6 and whose values are
        StructureGraphIDs, i.e., integer region identifiers (uint32).

    """
    color_map = {}
    doc = minidom.parse(str(svg_filename))
    regexp = re.compile(r"fill:(#([0-9a-f]{6}))")
    for path in doc.getElementsByTagName("path"):
        style = str(path.getAttribute("style"))
        structure_id = int(path.getAttribute("structure_id"))
        search = regexp.search(style)
        if search is not None:
            hexadecimal_color = search.group(1)
            color_map[hexadecimal_color] = structure_id

    return color_map
