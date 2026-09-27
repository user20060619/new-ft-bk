#!/usr/bin/env python3
"""Tests for sar.py (T7), with synthetic SAR scenes.  Owner: P4.

    python -m pytest backend/services/test_sar.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import load_raster  # noqa: E402
from backend.services.sar import (  # noqa: E402
    built_up_mask,
    extract_sar,
    lee_filter,
    to_db,
    water_mask,
)

# Multiplicative speckle: gamma(shape=8, scale=1/8) has mean 1, moderate
# spread -- tight enough that Otsu/percentile land cleanly relative to the
# tier means below, verified empirically before committing to these numbers.
_GAMMA_SHAPE = 8


def _speckled_scene(rows_split: list[int], means: list[float], seed: int, width: int = 90) -> np.ndarray:
    """Single-band linear-intensity array, `len(rows_split)` horizontal tiers
    stacked top to bottom, each with multiplicative speckle noise."""
    rng = np.random.default_rng(seed)
    height = sum(rows_split)
    array = np.zeros((height, width), dtype=np.float64)
    row = 0
    for rows, mean in zip(rows_split, means):
        speckle = rng.gamma(shape=_GAMMA_SHAPE, scale=1.0 / _GAMMA_SHAPE, size=(rows, width))
        array[row:row + rows, :] = mean * speckle
        row += rows
    return array.astype(np.float32)


def _write_geotiff(path, array_2d, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array_2d.shape[0], width=array_2d.shape[1], count=1,
        dtype=array_2d.dtype, nodata=nodata,
    ) as dst:
        dst.write(array_2d, 1)
    return load_raster(path, modality="sar")


# --- to_db --------------------------------------------------------------------

def test_to_db_intensity_explicit():
    arr = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    db, valid, info, warnings = to_db(arr, input_type="intensity")

    assert info == {"skipped": False, "input_type_used": "intensity", "assumed": False}
    np.testing.assert_allclose(db, 10 * np.log10(arr), rtol=1e-4)
    assert valid.all()
    assert not any("assumed" in w.lower() for w in warnings)


def test_to_db_intensity_assumed_by_default():
    arr = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    db, valid, info, warnings = to_db(arr)

    assert info["assumed"] is True
    assert info["input_type_used"] == "intensity"
    assert any("assumed" in w.lower() and "intensity" in w.lower() for w in warnings)


def test_to_db_amplitude():
    arr = np.array([0.1, 1.0, 10.0], dtype=np.float32)
    db, valid, info, warnings = to_db(arr, input_type="amplitude")

    assert info["input_type_used"] == "amplitude"
    np.testing.assert_allclose(db, 20 * np.log10(arr), rtol=1e-4)


def test_to_db_already_db_skips_conversion():
    arr = np.array([-25.0, -10.0, -2.0], dtype=np.float32)  # looks_like_sar_db range
    db, valid, info, warnings = to_db(arr)

    assert info["skipped"] is True
    np.testing.assert_allclose(db, arr)
    assert valid.all()
    assert any("already appears to be in db" in w.lower() for w in warnings)


def test_to_db_excludes_non_positive_without_fudging():
    # -100 sits outside the plausible dB range (SAR_DB_MIN=-60), so this array
    # doesn't accidentally look_like_sar_db and take the skip path -- it's
    # unambiguously raw intensity data with two bad values to guard against.
    arr = np.array([0.0, -100.0, 5.0], dtype=np.float32)
    db, valid, info, warnings = to_db(arr, input_type="intensity")

    assert info["skipped"] is False
    assert not valid[0] and not valid[1] and valid[2]
    assert np.isnan(db[0]) and np.isnan(db[1])
    assert db[2] == pytest.approx(10 * np.log10(5.0), rel=1e-4)
    assert any("zero or negative" in w.lower() for w in warnings)


def test_to_db_respects_preexisting_invalid_mask():
    arr = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    invalid = np.array([True, False, False])
    db, valid, info, warnings = to_db(arr, invalid_mask=invalid, input_type="intensity")

    assert not valid[0]
    assert np.isnan(db[0])
    assert valid[1] and valid[2]


def test_to_db_invalid_input_type_raises():
    with pytest.raises(ValueError):
        to_db(np.array([1.0], dtype=np.float32), input_type="bogus")


# --- lee_filter -----------------------------------------------------------------

def test_lee_filter_reduces_variance_and_preserves_ordering():
    array = _speckled_scene([36, 45, 9], [0.003, 0.3, 3.0], seed=1)
    db, valid, _, _ = to_db(array, input_type="intensity")
    filtered = lee_filter(db, valid, window=5)

    water_raw = db[0:36, :]
    water_filtered = filtered[0:36, :]
    assert np.nanvar(water_filtered) < np.nanvar(water_raw)

    water_mean = np.nanmean(filtered[0:36, :])
    other_mean = np.nanmean(filtered[36:81, :])
    built_up_mean = np.nanmean(filtered[81:, :])
    assert water_mean < other_mean < built_up_mean


# --- water_mask / built_up_mask, isolated -----------------------------------

def test_water_mask_isolated_bimodal_uses_otsu():
    rng = np.random.default_rng(10)
    water_db = rng.normal(-20, 1, size=(50, 50))
    land_db = rng.normal(-6, 1, size=(50, 50))
    db = np.concatenate([water_db, land_db], axis=0).astype(np.float32)
    valid = np.ones_like(db, dtype=bool)

    mask, threshold, source = water_mask(db, valid)

    assert source == "otsu"
    pct = mask.sum() / valid.sum() * 100
    assert 40.0 < pct < 60.0


def test_water_mask_isolated_no_water_triggers_guard():
    rng = np.random.default_rng(11)
    land1_db = rng.normal(-8, 1, size=(50, 50))
    land2_db = rng.normal(0, 1, size=(50, 50))
    db = np.concatenate([land1_db, land2_db], axis=0).astype(np.float32)
    valid = np.ones_like(db, dtype=bool)

    mask, threshold, source = water_mask(db, valid, water_max_db=-15.0)

    assert source == "guard"
    assert threshold == -15.0
    pct = mask.sum() / valid.sum() * 100
    assert pct < 5.0


def test_built_up_mask_isolated_bright_fraction():
    rng = np.random.default_rng(12)
    land_db = rng.normal(-8, 1, size=(90, 90))
    bright_db = rng.normal(3, 0.5, size=(10, 90))
    db = np.concatenate([land_db, bright_db], axis=0).astype(np.float32)
    valid = np.ones_like(db, dtype=bool)
    no_water = np.zeros_like(db, dtype=bool)

    mask, threshold, source = built_up_mask(db, valid, no_water, percentile=90.0)

    assert source == "percentile"
    pct = mask.sum() / valid.sum() * 100
    assert 5.0 < pct < 15.0


def test_built_up_mask_isolated_no_bright_targets_triggers_guard():
    rng = np.random.default_rng(13)
    db = rng.normal(-9, 1, size=(90, 90)).astype(np.float32)
    valid = np.ones_like(db, dtype=bool)
    no_water = np.zeros_like(db, dtype=bool)

    mask, threshold, source = built_up_mask(db, valid, no_water, built_up_min_db=-5.0)

    assert source == "guard"
    assert threshold == -5.0
    pct = mask.sum() / valid.sum() * 100
    assert pct < 5.0


# --- extract_sar: full pipeline scenes ------------------------------------------

def test_extract_sar_main_scene_uses_relative_thresholds(tmp_path):
    array = _speckled_scene([36, 45, 9], [0.003, 0.3, 3.0], seed=1)
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result = extract_sar(r, tmp_path)

    assert result.water_threshold_source == "otsu"
    assert result.built_up_threshold_source == "percentile"
    assert 30.0 < result.water_percentage < 50.0
    assert 5.0 < result.built_up_percentage < 15.0
    assert "rule-based" in result.water_method
    assert "rule-based" in result.built_up_method
    assert -60.0 <= result.water_threshold_db <= 15.0
    assert -60.0 <= result.built_up_threshold_db <= 15.0

    for name in ("sar_filtered.png", "water_mask.png", "built_up_mask.png"):
        assert (tmp_path / name).exists()
    assert all(e.modality == "sar" for e in result.evidence)
    job_id = tmp_path.name
    assert all(f"/outputs/{job_id}/" in e.url for e in result.evidence)


def test_extract_sar_no_water_scene_reports_near_zero_water(tmp_path):
    array = _speckled_scene([81, 9], [0.3, 3.0], seed=2)  # no dark/water tier at all
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result = extract_sar(r, tmp_path)

    assert result.water_threshold_source == "guard"
    assert result.water_threshold_db == -15.0
    assert result.water_percentage < 5.0
    assert any("water_max_db guard" in w for w in result.warnings)
    # the genuine bright tier is still correctly detected
    assert result.built_up_threshold_source == "percentile"
    assert result.built_up_percentage > 5.0


def test_extract_sar_no_built_up_scene_reports_near_zero_built_up(tmp_path):
    array = _speckled_scene([36, 54], [0.003, 0.15], seed=3)  # no bright tier at all
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result = extract_sar(r, tmp_path)

    assert result.built_up_threshold_source == "guard"
    assert result.built_up_threshold_db == -5.0
    assert result.built_up_percentage < 5.0
    assert any("built_up_min_db guard" in w for w in result.warnings)
    # the genuine water tier is still correctly detected
    assert result.water_threshold_source == "otsu"
    assert result.water_percentage > 30.0


def test_extract_sar_respects_nodata(tmp_path):
    array = _speckled_scene([36, 45, 9], [0.003, 0.3, 3.0], seed=1)
    nodata_value = -9999.0
    array_with_nodata = array.copy()
    array_with_nodata[0:10, 0:10] = nodata_value  # a block inside the water tier

    r = _write_geotiff(tmp_path / "scene.tif", array_with_nodata, nodata=nodata_value)
    assert r.nodata == nodata_value

    result = extract_sar(r, tmp_path)

    # nodata block excluded, not counted as "not water" -- percentage stays
    # close to the true ~40% water fraction rather than being diluted down
    assert 30.0 < result.water_percentage < 50.0


def test_prefix_avoids_filename_collision(tmp_path):
    array = _speckled_scene([36, 45, 9], [0.003, 0.3, 3.0], seed=1)
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result_a = extract_sar(r, tmp_path, prefix="before_")
    result_b = extract_sar(r, tmp_path, prefix="after_")

    assert (tmp_path / "before_water_mask.png").exists()
    assert (tmp_path / "after_water_mask.png").exists()
    assert result_a.evidence[0].id.startswith("before_")
    assert result_b.evidence[0].id.startswith("after_")


def test_explicit_job_id_used_in_evidence_urls(tmp_path):
    array = _speckled_scene([36, 45, 9], [0.003, 0.3, 3.0], seed=1)
    r = _write_geotiff(tmp_path / "scene.tif", array)

    result = extract_sar(r, tmp_path, job_id="my-request-id")

    assert all("/outputs/my-request-id/" in e.url for e in result.evidence)
