"""test app.nrrd"""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from voxcell import VoxelData

import atlas_building_tools.app.nrrd as tested
from atlas_building_tools.exceptions import AtlasBuildingToolsError

NRRDS = Path(Path(__file__).parent) / "nrrds"


def _run_merge(tmp_path, *files):
    return CliRunner().invoke(
        tested.app,
        [
            "merge",
            "--output",
            str(tmp_path / "out.nrrd"),
            "--sentinel",
            "0",
            *[str(f) for f in files],
        ],
    )


def test_merge_ok(tmp_path):
    result = _run_merge(tmp_path, NRRDS / "1.nrrd", NRRDS / "1.nrrd")
    assert result.exit_code == 0
    assert result.output == ""


def test_merge_shape_mismatch(tmp_path):
    result = _run_merge(
        tmp_path,
        NRRDS / "1.nrrd",
        NRRDS / "1_shape.nrrd",
    )
    assert result.exit_code == -1
    assert "Different shapes" in result.output


def test_merge_voxel_dimensions_mismatch(tmp_path):
    result = _run_merge(
        tmp_path,
        NRRDS / "1.nrrd",
        NRRDS / "1_voxel_dimensions.nrrd",
    )
    assert result.exit_code == -1
    assert "Different voxel dimensions" in result.output


def test_merge_offset_mismatch(tmp_path):
    result = _run_merge(
        tmp_path,
        NRRDS / "1.nrrd",
        NRRDS / "1_offset.nrrd",
    )
    assert result.exit_code == -1
    assert "Different offsets" in result.output
