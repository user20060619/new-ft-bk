#!/usr/bin/env python3
"""Tests for optical_sar.py (T8).  Owner: P4.

    python -m pytest backend/services/test_optical_sar.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import load_raster  # noqa: E402
from backend.services.landcover import LandcoverResult  # noqa: E402
from backend.services.optical_sar import fuse_masks, resample_sar_to_optical  # noqa: E402
from backend.services.sar import SarResult  # noqa: E402

UTM43 = CRS.from_epsg(32643)


def _write_geotiff(path, array, crs=None, transform=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)


def _optical(path, width=40, height=40, crs=None, transform=None):
    array = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)
    _write_geotiff(path, array, crs=crs, transform=transform)
    return load_raster(path)


def _sar_with_pattern(path, width, height, crs=None, transform=None):
    """Top half = -5.0, bottom half = -25.0 (linear intensity would be
    negative dB-looking, but we just want a spatially-known pattern to
    verify resampling preserves, not physically-real SAR values here)."""
    array = np.zeros((1, height, width), dtype=np.float32)
    array[:, : height // 2, :] = -5.0
    array[:, height // 2:, :] = -25.0
    _write_geotiff(path, array, crs=crs, transform=transform)
    return load_raster(path, modality="sar")


def _minimal_landcover_result(water_mask=None, built_up_mask=None, valid_mask=None) -> LandcoverResult:
    shape = (2, 2) if water_mask is None else water_mask.shape
    return LandcoverResult(
        water_percentage=0.0, water_method="test", water_is_true_index=False, water_threshold=None,
        vegetation_percentage=0.0, vegetation_method="test", vegetation_is_true_index=False, vegetation_threshold=None,
        built_up_percentage=0.0, built_up_method="test",
        water_mask=water_mask,
        built_up_mask=built_up_mask if built_up_mask is not None else np.zeros(shape, dtype=bool),
        valid_mask=valid_mask if valid_mask is not None else np.ones(shape, dtype=bool),
    )


def _minimal_sar_result(water_mask, built_up_mask, valid_mask=None, water_percentage=0.0) -> SarResult:
    shape = water_mask.shape
    return SarResult(
        water_percentage=water_percentage, water_method="test", water_threshold_db=-15.0, water_threshold_source="otsu",
        built_up_percentage=0.0, built_up_method="test", built_up_threshold_db=-5.0, built_up_threshold_source="percentile",
        water_mask=water_mask,
        built_up_mask=built_up_mask,
        valid_mask=valid_mask if valid_mask is not None else np.ones(shape, dtype=bool),
    )


# --- resample_sar_to_optical --------------------------------------------------

def test_resample_noop_when_grids_already_match(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    optical = _optical(tmp_path / "opt.tif", width=40, height=40, crs=UTM43, transform=t)
    sar = _sar_with_pattern(tmp_path / "sar.tif", width=40, height=40, crs=UTM43, transform=t)

    resampled, was_resampled, method = resample_sar_to_optical(sar, optical)

    assert was_resampled is False
    assert method == "n/a (already matching grid)"
    assert resampled is sar


def test_resample_crs_aware_reprojection_preserves_spatial_pattern(tmp_path):
    # optical: 40x40 @ 5m; sar: 20x20 @ 10m -- same real-world extent, coarser SAR.
    optical = _optical(tmp_path / "opt.tif", width=40, height=40, crs=UTM43,
                        transform=from_origin(500000, 4000000, 5, 5))
    sar = _sar_with_pattern(tmp_path / "sar.tif", width=20, height=20, crs=UTM43,
                             transform=from_origin(500000, 4000000, 10, 10))

    resampled, was_resampled, method = resample_sar_to_optical(sar, optical)

    assert was_resampled is True
    assert method == "reproject (CRS-aware, nearest)"
    assert (resampled.width, resampled.height) == (40, 40)
    assert resampled.gsd_m == optical.gsd_m
    assert resampled.transform == optical.transform
    assert resampled.crs == optical.crs

    # the known top/bottom pattern must survive reprojection onto the finer grid
    assert np.allclose(resampled.array[0, :15, :], -5.0)
    assert np.allclose(resampled.array[0, 25:, :], -25.0)


def test_resample_falls_back_to_resize_when_crs_missing(tmp_path):
    # optical as a plain JPEG: no CRS/transform at all.
    pixels = np.random.randint(0, 255, size=(30, 30, 3), dtype=np.uint8)
    opt_path = tmp_path / "opt.jpg"
    Image.fromarray(pixels, mode="RGB").save(opt_path, format="JPEG")
    optical = load_raster(opt_path)
    assert optical.crs is None

    sar = _sar_with_pattern(tmp_path / "sar.tif", width=15, height=15, crs=UTM43,
                             transform=from_origin(500000, 4000000, 10, 10))

    resampled, was_resampled, method = resample_sar_to_optical(sar, optical)

    assert was_resampled is True
    assert method == "resize (size-only, nearest)"
    assert (resampled.width, resampled.height) == (optical.width, optical.height)
    assert any("don't both have crs" in w.lower() for w in resampled.warnings)


# --- fuse_masks ----------------------------------------------------------------

def test_fuse_masks_computes_known_overlap_percentages(tmp_path):
    optical_water = np.array([[True, True], [False, False]])
    sar_water = np.array([[True, False], [True, False]])
    optical_built = np.array([[False, False], [True, True]])
    sar_built = np.array([[False, False], [True, False]])

    optical_result = _minimal_landcover_result(water_mask=optical_water, built_up_mask=optical_built)
    sar_result = _minimal_sar_result(water_mask=sar_water, built_up_mask=sar_built)

    result = fuse_masks(optical_result, sar_result, tmp_path)

    # water: agreement={0,0}=1px, union=3px, optical_only=1px, sar_only=1px, total=4
    assert result.water_agreement_pct == 25.0
    assert result.water_union_pct == 75.0
    assert result.water_optical_only_pct == 25.0
    assert result.water_sar_only_pct == 25.0

    # built-up: agreement={1,0}=1px, union=2px({1,0},{1,1}), optical_only={1,1}=1px, sar_only=0
    assert result.built_up_agreement_pct == 25.0
    assert result.built_up_union_pct == 50.0
    assert result.built_up_optical_only_pct == 25.0
    assert result.built_up_sar_only_pct == 0.0

    assert (tmp_path / "fused_overlay.png").exists()
    assert any(e.id == "fused_overlay" and e.modality == "fused" for e in result.evidence)


def test_fuse_masks_excludes_invalid_sar_pixels_from_denominator(tmp_path):
    optical_water = np.array([[True, True], [True, True]])
    sar_water = np.array([[True, True], [True, True]])
    valid = np.array([[True, True], [False, False]])  # only top row is valid SAR data

    optical_result = _minimal_landcover_result(water_mask=optical_water,
                                                 built_up_mask=np.zeros((2, 2), dtype=bool))
    sar_result = _minimal_sar_result(water_mask=sar_water, built_up_mask=np.zeros((2, 2), dtype=bool),
                                      valid_mask=valid)

    result = fuse_masks(optical_result, sar_result, tmp_path)

    assert result.water_agreement_pct == 100.0  # both agree on all 2 *valid* pixels
    assert any("excluded from all optical/sar fusion" in w.lower() for w in result.warnings)


def test_fuse_masks_water_none_reports_sar_only(tmp_path):
    # panchromatic optical: water_mask is None
    optical_result = _minimal_landcover_result(water_mask=None, built_up_mask=np.zeros((2, 2), dtype=bool))
    sar_water = np.array([[True, False], [False, False]])
    sar_result = _minimal_sar_result(water_mask=sar_water, built_up_mask=np.zeros((2, 2), dtype=bool),
                                      water_percentage=25.0)

    result = fuse_masks(optical_result, sar_result, tmp_path)

    assert result.water_optical_only_pct is None
    assert result.water_sar_only_pct is None
    assert result.water_agreement_pct is None
    assert result.water_union_pct == 25.0
    assert any("water fusion used sar only" in w.lower() for w in result.warnings)
