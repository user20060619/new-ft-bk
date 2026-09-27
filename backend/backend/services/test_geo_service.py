#!/usr/bin/env python3
"""Tests for the honesty fixes in geo_service.py.  Owner: P4.

    python -m pytest backend/services/test_geo_service.py
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import RasterInput, load_raster  # noqa: E402
from backend.services import geo_service  # noqa: E402
from backend.services.geo_service import (  # noqa: E402
    align_images,
    calculate_area_percentage,
    calculate_vegetation_index,
    calculate_water_index,
)

UTM43 = CRS.from_epsg(32643)


def _band_constant_geotiff(path, band_values, crs=None, transform=None, dtype=np.uint16):
    height, width = 20, 20
    array = np.zeros((len(band_values), height, width), dtype=dtype)
    for i, v in enumerate(band_values):
        array[i] = v
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=len(band_values),
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)
    return load_raster(path)


def _panchromatic(path, value=100, dtype=np.uint8):
    height, width = 20, 20
    array = np.full((1, height, width), value, dtype=dtype)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=1, dtype=array.dtype,
    ) as dst:
        dst.write(array)
    return load_raster(path, modality="optical")


def _hand_built_optical(band_values_left_right, width=20, height=20) -> RasterInput:
    """4-band (B,G,R,NIR) RasterInput with each band split left/right down
    the middle -- (left_value, right_value) per band -- so a valid_mask that
    covers only one half gives a genuinely different mean than the full
    frame, without needing real SIFT alignment."""
    array = np.zeros((len(band_values_left_right), height, width), dtype=np.float32)
    half = width // 2
    for i, (left_v, right_v) in enumerate(band_values_left_right):
        array[i, :, :half] = left_v
        array[i, :, half:] = right_v
    return RasterInput(
        array=array, filename="hand_built.tif", modality="optical", bands=len(band_values_left_right),
        dtype="float32", width=width, height=height, crs=None, transform=None,
        gsd_m=None, date=None, nodata=None, warnings=[],
    )


# --- calculate_area_percentage: no silent default pixel size -----------------

def test_calculate_area_percentage_requires_explicit_pixel_size():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :5] = 255  # 25 of 100 pixels

    pct, area = calculate_area_percentage(mask)
    assert pct == pytest.approx(25.0)
    assert area is None

    pct2, area2 = calculate_area_percentage(mask, pixel_size_meters=10)
    assert pct2 == pytest.approx(25.0)
    assert area2 == pytest.approx((25 * 10 * 10) / 1_000_000)


# --- vegetation: true NDVI vs proxy -------------------------------------------

def test_true_ndvi_computed_for_4band_optical_with_nir(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    before = _band_constant_geotiff(tmp_path / "before.tif", [50, 60, 100, 200], crs=UTM43, transform=t)
    after = _band_constant_geotiff(tmp_path / "after.tif", [50, 60, 150, 150], crs=UTM43, transform=t)

    result = calculate_vegetation_index(before, after)

    assert result["is_true_index"] is True
    assert result["method"].startswith("NDVI")
    assert result["index_before"] == pytest.approx(1 / 3, abs=1e-3)
    assert result["index_after"] == pytest.approx(0.0, abs=1e-3)
    assert result["pct_change"] == pytest.approx(-100.0, abs=0.1)
    assert "area_changed_km2" in result
    assert result["area_changed_km2"] >= 0
    assert result["warnings"] == []


def test_vegetation_proxy_for_3band_rgb(tmp_path):
    before = _band_constant_geotiff(tmp_path / "before.tif", [50, 60, 100], dtype=np.uint8)
    after = _band_constant_geotiff(tmp_path / "after.tif", [50, 200, 100], dtype=np.uint8)  # boosted green

    result = calculate_vegetation_index(before, after)

    assert result["is_true_index"] is False
    assert "NDVI" not in result["method"]
    assert "vegetation_proxy" in result["method"]
    assert "vegetation_proxy_before" in result
    assert "vegetation_proxy_after" in result
    assert result["vegetation_proxy_after"] > result["vegetation_proxy_before"]
    assert any("vegetation_proxy" in w and "not NDVI" in w for w in result["warnings"])


def test_area_changed_km2_absent_without_known_gsd(tmp_path):
    before = _band_constant_geotiff(tmp_path / "before.tif", [50, 60, 100, 200])  # no crs/transform
    after = _band_constant_geotiff(tmp_path / "after.tif", [50, 60, 150, 150])

    result = calculate_vegetation_index(before, after)

    assert result["is_true_index"] is True  # still a real 4-band optical NIR case
    assert "area_changed_km2" not in result
    assert any("ground sample distance" in w.lower() for w in result["warnings"])


def test_no_true_index_for_sar_even_with_4_bands(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    before = _band_constant_geotiff(tmp_path / "before.tif", [50, 60, 100, 200], crs=UTM43, transform=t)
    after = _band_constant_geotiff(tmp_path / "after.tif", [50, 60, 150, 150], crs=UTM43, transform=t)
    before.modality = "sar"
    after.modality = "sar"

    result = calculate_vegetation_index(before, after)

    assert result["is_true_index"] is False


def test_vegetation_band_order_override(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    # NIR stored at index 0 instead of the default 3.
    before = _band_constant_geotiff(tmp_path / "before.tif", [200, 60, 100, 50], crs=UTM43, transform=t)
    after = _band_constant_geotiff(tmp_path / "after.tif", [150, 60, 150, 50], crs=UTM43, transform=t)
    custom_order = {"blue": 3, "green": 1, "red": 2, "nir": 0}

    result = calculate_vegetation_index(before, after, band_order=custom_order)

    assert result["is_true_index"] is True
    assert result["index_before"] == pytest.approx(1 / 3, abs=1e-3)


# --- water: true NDWI vs proxy -------------------------------------------------

def test_true_ndwi_computed_for_4band_optical_with_nir(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    before = _band_constant_geotiff(tmp_path / "before.tif", [30, 200, 80, 50], crs=UTM43, transform=t)
    after = _band_constant_geotiff(tmp_path / "after.tif", [30, 100, 80, 150], crs=UTM43, transform=t)

    result = calculate_water_index(before, after)

    assert result["is_true_index"] is True
    assert result["method"].startswith("NDWI")
    assert result["index_before"] == pytest.approx(0.6, abs=1e-3)
    assert result["index_after"] == pytest.approx(-0.2, abs=1e-3)
    assert "area_changed_km2" in result


def test_water_proxy_for_3band_rgb(tmp_path):
    before = _band_constant_geotiff(tmp_path / "before.tif", [30, 80, 100], dtype=np.uint8)
    after = _band_constant_geotiff(tmp_path / "after.tif", [30, 150, 60], dtype=np.uint8)

    result = calculate_water_index(before, after)

    assert result["is_true_index"] is False
    assert "NDWI" not in result["method"]
    assert "water_proxy" in result["method"]
    assert "water_proxy_before" in result
    assert "water_proxy_after" in result
    assert any("water_proxy" in w and "not NDWI" in w for w in result["warnings"])


# --- regressions: no hardcoded confidence, alignment still works -------------

def test_no_hardcoded_confidence_in_module():
    source = inspect.getsource(geo_service)
    assert "0.85" not in source


def test_align_images_still_works_with_textured_arrays():
    rng = np.random.default_rng(5)
    before = rng.integers(0, 255, size=(60, 60, 3), dtype=np.uint8)
    after = before.copy()

    aligned, valid_mask, ok, diag = align_images(before, after)

    assert aligned.shape == before.shape
    assert valid_mask.shape == before.shape[:2]
    assert ok is True
    assert diag["good_matches"] > 0
    assert diag["homography_found"] is True
    assert isinstance(diag["homography"], np.ndarray)
    assert diag["homography"].shape == (3, 3)


def test_align_images_homography_is_none_when_alignment_falls_back():
    # A flat, textureless image gives SIFT no keypoints to match -> fallback.
    before = np.full((40, 40, 3), 128, dtype=np.uint8)
    after = np.full((40, 40, 3), 128, dtype=np.uint8)

    aligned, valid_mask, ok, diag = align_images(before, after)

    assert ok is False
    assert diag["homography_found"] is False
    assert diag["homography"] is None


# --- panchromatic (1-2 band) optical: honest "not computable", never a crash -

def test_vegetation_index_panchromatic_not_computable(tmp_path):
    before = _panchromatic(tmp_path / "before.tif", value=100)
    after = _panchromatic(tmp_path / "after.tif", value=150)

    result = calculate_vegetation_index(before, after)

    assert result["is_true_index"] is False
    assert result["pct_change"] is None
    assert "not computable" in result["method"]
    assert any("colour" in w.lower() for w in result["warnings"])


def test_water_index_panchromatic_not_computable(tmp_path):
    before = _panchromatic(tmp_path / "before.tif", value=100)
    after = _panchromatic(tmp_path / "after.tif", value=150)

    result = calculate_water_index(before, after)

    assert result["is_true_index"] is False
    assert result["pct_change"] is None
    assert "not computable" in result["method"]


def test_vegetation_index_panchromatic_asymmetric_bands_still_safe(tmp_path):
    # only ONE date is panchromatic -- must still short-circuit, not crash
    # trying to index colour bands that exist on only one side.
    t = from_origin(500000, 4000000, 10, 10)
    before = _band_constant_geotiff(tmp_path / "before.tif", [50, 60, 100, 200], crs=UTM43, transform=t)
    after = _panchromatic(tmp_path / "after.tif", value=100)

    result = calculate_vegetation_index(before, after)

    assert result["pct_change"] is None
    assert "not computable" in result["method"]


# --- overlap-aware statistics: real, distinct from the native full frame -----

def test_vegetation_index_overlap_restricted_differs_from_native_full_frame():
    # before: left half NIR=200 (NDVI 0.6), right half NIR=100 (NDVI 0.333);
    # after: uniform NIR=50, red=100 (NDVI -0.333) everywhere.
    before = _hand_built_optical([(50, 50), (60, 60), (50, 50), (200, 100)])
    after = _hand_built_optical([(50, 50), (60, 60), (100, 100), (50, 50)])

    native = calculate_vegetation_index(before, after)

    valid_left_only = np.zeros((20, 20), dtype=bool)
    valid_left_only[:, :10] = True
    overlap = calculate_vegetation_index(
        before, after, target_size=(20, 20), homography=None, valid_mask=valid_left_only,
    )

    assert native["is_true_index"] is True
    assert overlap["is_true_index"] is True
    # native mixes both halves; overlap sees only the left (higher-NDVI) half.
    assert native["index_before"] == pytest.approx((0.6 + 1 / 3) / 2, abs=1e-3)
    assert overlap["index_before"] == pytest.approx(0.6, abs=1e-3)
    assert native["index_before"] != overlap["index_before"]
    assert native["pct_change"] != overlap["pct_change"]


def test_vegetation_index_zero_valid_pixels_reports_not_computable():
    before = _hand_built_optical([(50, 50), (60, 60), (50, 50), (200, 100)])
    after = _hand_built_optical([(50, 50), (60, 60), (100, 100), (50, 50)])

    empty_valid = np.zeros((20, 20), dtype=bool)
    result = calculate_vegetation_index(before, after, target_size=(20, 20), valid_mask=empty_valid)

    assert result["pct_change"] is None
    assert any("no overlapping valid area" in w for w in result["warnings"])
