#!/usr/bin/env python3
"""Tests for the input compatibility check.  Owner: P4.

Every case is built from synthetic GeoTIFFs written with rasterio into
tmp_path and loaded through the real `load_raster`, so this exercises the
actual RasterInput contract rather than hand-built stand-ins.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import load_raster  # noqa: E402
from backend.rsio.compat import check_inputs, CompatResult  # noqa: E402


def _write_geotiff(path, array, crs=None, transform=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)


def _optical(path, width=50, height=50, crs=None, transform=None):
    array = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)
    _write_geotiff(path, array, crs=crs, transform=transform)
    return load_raster(path)


def _sar(path, width=50, height=50, crs=None, transform=None):
    array = (np.random.rand(1, height, width) * -30).astype(np.float32)
    _write_geotiff(path, array, crs=crs, transform=transform)
    return load_raster(path)


UTM43 = CRS.from_epsg(32643)
UTM44 = CRS.from_epsg(32644)


def test_single_input(tmp_path):
    r = _optical(tmp_path / "a.tif")

    result = check_inputs([r])

    assert isinstance(result, CompatResult)
    assert result.input_config == "single"
    assert result.accepted is True
    assert result.errors == []


def test_two_optical_is_bitemporal(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    result = check_inputs([a, b])

    assert result.input_config == "bitemporal"
    assert result.accepted is True
    assert result.errors == []
    assert "resample_required" not in result.warnings


def test_optical_plus_sar_is_optical_sar(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    b = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    result = check_inputs([a, b])

    assert result.input_config == "optical_sar"
    assert result.accepted is True
    assert result.errors == []


def test_two_sar_is_bitemporal_sar_change(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _sar(tmp_path / "sar1.tif", crs=UTM43, transform=t)
    b = _sar(tmp_path / "sar2.tif", crs=UTM43, transform=t)

    result = check_inputs([a, b])

    assert result.input_config == "bitemporal"
    assert result.accepted is True
    assert result.errors == []


def test_crs_mismatch_is_incompatible(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM44, transform=t)

    result = check_inputs([a, b])

    assert result.accepted is False
    assert any(e.code == "incompatible_pair" for e in result.errors)
    assert any("crs" in e.message.lower() for e in result.errors)


def test_non_overlapping_bounds_is_incompatible(tmp_path):
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=from_origin(500000, 4000000, 10, 10))
    # Far away in the same CRS: no possible overlap.
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=from_origin(900000, 4000000, 10, 10))

    result = check_inputs([a, b])

    assert result.accepted is False
    assert any(e.code == "incompatible_pair" for e in result.errors)
    assert any("overlap" in e.message.lower() for e in result.errors)


def test_size_mismatch_flags_resample_required(tmp_path):
    origin = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", width=50, height=50, crs=UTM43, transform=origin)
    b = _optical(tmp_path / "b.tif", width=60, height=60, crs=UTM43, transform=origin)

    result = check_inputs([a, b])

    assert result.accepted is True
    assert result.errors == []
    assert "resample_required" in result.warnings


def test_missing_crs_pair_is_not_rejected(tmp_path):
    def _jpeg(path):
        pixels = np.random.randint(0, 255, size=(30, 30, 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(path, format="JPEG")
        return load_raster(path)

    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")

    result = check_inputs([a, b])

    assert result.input_config == "bitemporal"
    assert result.accepted is True
    assert result.errors == []
    assert any("crs" in w.lower() for w in result.warnings)


def test_too_many_inputs(tmp_path):
    a = _optical(tmp_path / "a.tif")
    b = _optical(tmp_path / "b.tif")
    c = _optical(tmp_path / "c.tif")

    result = check_inputs([a, b, c])

    assert result.accepted is False
    assert result.input_config is None
    assert any(e.code == "too_many_inputs" for e in result.errors)


def test_no_inputs(tmp_path):
    result = check_inputs([])

    assert result.accepted is False
    assert result.input_config is None
    assert any(e.code == "no_inputs" for e in result.errors)


def test_unreadable_file(tmp_path):
    a = _optical(tmp_path / "a.tif")

    result = check_inputs([a, None])

    assert result.accepted is False
    assert result.input_config is None
    assert any(e.code == "unreadable_file" for e in result.errors)


def test_all_unreadable(tmp_path):
    result = check_inputs([None])

    assert result.accepted is False
    assert any(e.code == "unreadable_file" for e in result.errors)
