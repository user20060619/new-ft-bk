#!/usr/bin/env python3
"""Tests for the raster loader.  Owner: P4.

Synthetic rasters only -- no real satellite data needed to prove the
format-dispatch, modality guess and never-fabricate-metadata rules hold.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import load_raster, RasterInput  # noqa: E402


def _write_geotiff(path, array, crs=None, transform=None, tags=None, nodata=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array)
        if tags:
            dst.update_tags(**tags)


def test_optical_3band_uint8_with_crs(tmp_path):
    array = np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)
    transform = from_origin(500000, 4000000, 10, 10)  # 10 m pixels
    path = tmp_path / "optical.tif"
    _write_geotiff(
        path, array, crs=CRS.from_epsg(32643), transform=transform,
        tags={"TIFFTAG_DATETIME": "2026:09:20 10:30:00"},
    )

    result = load_raster(path)

    assert isinstance(result, RasterInput)
    assert result.array.shape == (3, 32, 32)
    assert result.array.dtype == np.float32
    assert result.filename == "optical.tif"
    assert result.modality == "optical"
    assert result.bands == 3
    assert result.dtype == "uint8"
    assert (result.width, result.height) == (32, 32)
    assert result.crs == "EPSG:32643"
    assert result.transform is not None
    assert result.gsd_m == pytest.approx(10.0)
    assert result.date == "2026-09-20"
    assert result.warnings == []


def test_optical_4band_uint16_no_crs(tmp_path):
    array = (np.random.rand(4, 20, 20) * 30000).astype(np.uint16)
    path = tmp_path / "multiband.tif"
    _write_geotiff(path, array)  # no crs, no transform -> identity

    result = load_raster(path)

    assert result.array.shape == (4, 20, 20)
    assert result.bands == 4
    assert result.modality == "optical"
    assert result.dtype == "uint16"
    assert result.crs is None
    assert result.transform is None
    assert result.gsd_m is None
    assert result.date is None
    assert any("geotransform" in w.lower() for w in result.warnings)


def test_sar_1band_float32(tmp_path):
    array = (np.random.rand(1, 16, 16) * -30).astype(np.float32)  # dB-like, negative
    transform = from_origin(300000, 5000000, 5, 5)
    path = tmp_path / "sar.tif"
    _write_geotiff(path, array, crs=CRS.from_epsg(32643), transform=transform)

    result = load_raster(path)

    assert result.bands == 1
    assert result.modality == "sar"
    assert result.dtype == "float32"
    assert result.gsd_m == pytest.approx(5.0)
    assert result.date is None


def test_single_band_uint_is_unknown_modality(tmp_path):
    array = np.random.randint(0, 255, size=(1, 12, 12), dtype=np.uint8)
    path = tmp_path / "single_uint.tif"
    _write_geotiff(path, array)

    result = load_raster(path)

    assert result.bands == 1
    assert result.modality == "unknown"
    assert result.dtype == "uint8"
    assert any("unknown" in w.lower() for w in result.warnings)


def test_plain_jpeg_fallback(tmp_path):
    pixels = np.random.randint(0, 255, size=(24, 24, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    path = tmp_path / "photo.jpg"
    img.save(path, format="JPEG")

    result = load_raster(path)

    assert result.array.shape == (3, 24, 24)
    assert result.array.dtype == np.float32
    assert result.modality == "optical"
    assert result.crs is None
    assert result.transform is None
    assert result.gsd_m is None
    assert result.date is None
    assert any("georeferencing" in w.lower() for w in result.warnings)


def test_modality_override(tmp_path):
    array = np.random.randint(0, 255, size=(3, 10, 10), dtype=np.uint8)
    path = tmp_path / "override.tif"
    _write_geotiff(path, array)

    result = load_raster(path, modality="sar")

    assert result.modality == "sar"


def test_invalid_modality_override_raises(tmp_path):
    array = np.random.randint(0, 255, size=(3, 10, 10), dtype=np.uint8)
    path = tmp_path / "bad.tif"
    _write_geotiff(path, array)

    with pytest.raises(ValueError):
        load_raster(path, modality="not-a-real-modality")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.bmp"
    path.write_bytes(b"not a real image")

    with pytest.raises(ValueError):
        load_raster(path)


# --- large-image guard --------------------------------------------------------

def test_large_optical_geotiff_is_downsampled(tmp_path):
    width, height = 300, 200
    max_pixels = 300 * 200 // 4  # forces roughly a 2x2 downsample
    array = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)
    transform = from_origin(500000, 4000000, 10, 10)
    path = tmp_path / "big_optical.tif"
    _write_geotiff(path, array, crs=CRS.from_epsg(32643), transform=transform)

    result = load_raster(path, max_pixels=max_pixels)

    assert result.width * result.height <= max_pixels
    assert result.width < width and result.height < height
    assert result.array.shape == (3, result.height, result.width)
    # gsd_m must scale with the downsample, not stay at the native 10 m.
    expected_gsd = 10.0 * (width / result.width)
    assert result.gsd_m == pytest.approx(expected_gsd, rel=0.05)
    assert any("downsampled" in w.lower() and "average" in w.lower() for w in result.warnings)


def test_large_sar_geotiff_uses_nearest_resampling(tmp_path):
    width, height = 300, 200
    max_pixels = 300 * 200 // 4
    array = (np.random.rand(1, height, width) * -30).astype(np.float32)
    transform = from_origin(500000, 4000000, 10, 10)
    path = tmp_path / "big_sar.tif"
    _write_geotiff(path, array, crs=CRS.from_epsg(32643), transform=transform)

    result = load_raster(path, max_pixels=max_pixels)

    assert result.width * result.height <= max_pixels
    assert result.modality == "sar"
    assert any("downsampled" in w.lower() and "nearest" in w.lower() for w in result.warnings)


def test_large_jpeg_is_downsampled_with_pil(tmp_path):
    width, height = 400, 300
    max_pixels = 400 * 300 // 4
    pixels = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    path = tmp_path / "big_photo.jpg"
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG")

    result = load_raster(path, max_pixels=max_pixels)

    assert result.width * result.height <= max_pixels
    assert result.width < width and result.height < height
    assert result.array.shape == (3, result.height, result.width)
    assert any("downsampled" in w.lower() for w in result.warnings)


def test_small_image_is_not_downsampled(tmp_path):
    array = np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)
    path = tmp_path / "small.tif"
    _write_geotiff(path, array)

    result = load_raster(path, max_pixels=64 * 64)

    assert (result.width, result.height) == (32, 32)
    assert not any("downsampled" in w.lower() for w in result.warnings)


# --- polarisation/sensor hint SAR detection -----------------------------------

def test_band_descriptions_hint_sar(tmp_path):
    array = np.random.randint(0, 5000, size=(2, 20, 20), dtype=np.uint16)
    path = tmp_path / "sentinel1_grd.tif"
    _write_geotiff(path, array)
    with rasterio.open(path, "r+") as dst:
        dst.descriptions = ("VV", "VH")

    result = load_raster(path)

    assert result.bands == 2
    assert result.modality == "sar"
    assert not any("could not confidently determine modality" in w.lower() for w in result.warnings)


def test_two_band_uint_without_hints_is_unknown_with_override_tip(tmp_path):
    array = np.random.randint(0, 5000, size=(2, 20, 20), dtype=np.uint16)
    path = tmp_path / "plain_two_band.tif"
    _write_geotiff(path, array)

    result = load_raster(path)

    assert result.bands == 2
    assert result.modality == "unknown"
    assert any('modality="sar"' in w for w in result.warnings)


def test_dataset_tag_hint_sar(tmp_path):
    array = np.random.randint(0, 5000, size=(1, 20, 20), dtype=np.uint16)
    path = tmp_path / "risat.tif"
    _write_geotiff(path, array, tags={"SENSOR": "RISAT-1"})

    result = load_raster(path)

    assert result.modality == "sar"
