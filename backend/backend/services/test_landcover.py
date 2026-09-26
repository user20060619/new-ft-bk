#!/usr/bin/env python3
"""Tests for landcover.py (T6), on synthetic images with known regions.
Owner: P4.

    python -m pytest backend/services/test_landcover.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import load_raster  # noqa: E402
from backend.services.landcover import extract_optical  # noqa: E402

UTM43 = CRS.from_epsg(32643)

HEIGHT = WIDTH = 90  # clean multiple of 3: exact row boundaries for the thirds
THIRD = HEIGHT // 3
CELL = 3  # checkerboard cell size, small relative to the edge-density window


def _checkerboard(height: int, width: int, lo: float, hi: float, cell: int = CELL) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pattern = ((xs // cell) + (ys // cell)) % 2
    return np.where(pattern == 1, hi, lo)


def _three_region_rgb(built_up_lo: float, built_up_hi: float, dtype=np.uint8) -> np.ndarray:
    """(3,H,W) B,G,R array: rows [0:30)=water-like, [30:60)=vegetation-like,
    [60:90)=built-up (checkerboard, R=G=B so it can't leak into either colour
    proxy -- 2G-R-B and (G-R)/(G+R) are both exactly 0 whenever R=G=B)."""
    blue = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    green = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    red = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

    blue[0:THIRD, :], green[0:THIRD, :], red[0:THIRD, :] = 200, 120, 100        # water-like
    blue[THIRD:2 * THIRD, :], green[THIRD:2 * THIRD, :], red[THIRD:2 * THIRD, :] = 20, 140, 145  # vegetation-like

    checker = _checkerboard(THIRD, WIDTH, built_up_lo, built_up_hi)
    blue[2 * THIRD:, :] = checker
    green[2 * THIRD:, :] = checker
    red[2 * THIRD:, :] = checker

    return np.stack([blue, green, red]).astype(dtype)


def _three_region_4band(dtype=np.uint16, scale: float = 1.0) -> np.ndarray:
    """(4,H,W) B,G,R,NIR array using the true-index-friendly pattern: water
    has low NIR + high green; vegetation has high NIR + moderate red; the
    built-up checkerboard is flat across all four bands."""
    blue = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    green = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    red = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    nir = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

    blue[0:THIRD, :], green[0:THIRD, :], red[0:THIRD, :], nir[0:THIRD, :] = 180, 150, 50, 20  # water
    blue[THIRD:2 * THIRD, :], green[THIRD:2 * THIRD, :] = 30, 100
    red[THIRD:2 * THIRD, :], nir[THIRD:2 * THIRD, :] = 60, 200                                # vegetation

    checker = _checkerboard(THIRD, WIDTH, 40, 220)
    blue[2 * THIRD:, :] = checker
    green[2 * THIRD:, :] = checker
    red[2 * THIRD:, :] = checker
    nir[2 * THIRD:, :] = checker

    array = np.stack([blue, green, red, nir]) * scale
    return array.astype(dtype)


def _write_geotiff(path, array, crs=None, transform=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)
    return load_raster(path)


def test_rgb_only_proxy_paths_known_regions(tmp_path):
    array = _three_region_rgb(built_up_lo=40, built_up_hi=220)
    r = _write_geotiff(tmp_path / "scene.tif", array, crs=UTM43,
                        transform=from_origin(500000, 4000000, 10, 10))

    result = extract_optical(r, tmp_path)

    assert result.water_is_true_index is False
    assert "water_proxy" in result.water_method
    assert result.vegetation_is_true_index is False
    assert "vegetation_proxy" in result.vegetation_method
    assert result.built_up_is_true_index is False
    assert "rule-based" in result.built_up_method

    assert 25.0 < result.water_percentage < 40.0
    assert 25.0 < result.vegetation_percentage < 40.0
    assert result.built_up_percentage > 20.0

    assert any("vegetation_proxy" in w and "not NDVI" in w for w in result.warnings)
    assert any("water_proxy" in w and "not NDWI" in w for w in result.warnings)

    for name in ("water_mask.png", "vegetation_mask.png", "built_up_mask.png"):
        assert (tmp_path / name).exists()
    assert {e.kind for e in result.evidence} == {"mask"}
    assert all(e.modality == "optical" for e in result.evidence)
    job_id = tmp_path.name
    assert all(f"/outputs/{job_id}/" in e.url for e in result.evidence)


def test_built_up_dark_checkerboard_relies_on_texture_not_brightness(tmp_path):
    """A dark, high-contrast checkerboard (average well below the brightness
    threshold) must still register as built-up via edge density alone --
    otherwise brightness could carry every built-up assertion even with a
    completely broken texture computation."""
    array = _three_region_rgb(built_up_lo=40, built_up_hi=90)  # avg 65, both values < 100
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result = extract_optical(r, tmp_path)

    assert result.built_up_percentage > 20.0


def test_4band_true_index_paths_known_regions(tmp_path):
    array = _three_region_4band(dtype=np.uint16, scale=1.0)
    r = _write_geotiff(tmp_path / "scene.tif", array, crs=UTM43,
                        transform=from_origin(500000, 4000000, 10, 10))

    result = extract_optical(r, tmp_path)

    assert result.water_is_true_index is True
    assert result.water_method.startswith("NDWI")
    assert result.vegetation_is_true_index is True
    assert result.vegetation_method.startswith("NDVI")
    assert result.warnings == []  # no proxy fallback warnings when NIR is identifiable

    assert 25.0 < result.water_percentage < 40.0
    assert 25.0 < result.vegetation_percentage < 40.0
    assert result.built_up_percentage > 20.0


def test_vegetation_proxy_bit_depth_invariance(tmp_path):
    """The excess-green proxy is a raw linear combination, not a ratio, so
    without normalize_rgb its magnitude (and any fixed threshold) would scale
    with source bit depth.  Same relative pattern as the 8-bit RGB test,
    scaled ~257x into 16-bit range -- classification must still separate the
    regions correctly, not classify everything (or nothing) as vegetation."""
    array_8bit = _three_region_rgb(built_up_lo=40, built_up_hi=220, dtype=np.float64)
    array_16bit = (array_8bit * 257).astype(np.uint16)
    r = _write_geotiff(tmp_path / "scene16.tif", array_16bit)

    result = extract_optical(r, tmp_path)

    assert result.vegetation_is_true_index is False
    assert 25.0 < result.vegetation_percentage < 40.0
    assert 25.0 < result.water_percentage < 40.0


def test_prefix_avoids_filename_collision(tmp_path):
    array = _three_region_rgb(built_up_lo=40, built_up_hi=220)
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result_a = extract_optical(r, tmp_path, prefix="before_")
    result_b = extract_optical(r, tmp_path, prefix="after_")

    assert (tmp_path / "before_water_mask.png").exists()
    assert (tmp_path / "after_water_mask.png").exists()
    assert result_a.evidence[0].id.startswith("before_")
    assert result_b.evidence[0].id.startswith("after_")


def test_explicit_job_id_used_in_evidence_urls(tmp_path):
    array = _three_region_rgb(built_up_lo=40, built_up_hi=220)
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result = extract_optical(r, tmp_path, job_id="my-request-id")

    assert all("/outputs/my-request-id/" in e.url for e in result.evidence)
