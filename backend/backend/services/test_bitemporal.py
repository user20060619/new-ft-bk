#!/usr/bin/env python3
"""Tests for bitemporal.py (T9): modality-agnostic per-date normalisation,
percentage-point change with a stated tolerance, mask warping, and region
detection.  Hand-built numpy arrays throughout -- these are pure-array
functions, same testing style as optical_sar.py::fuse_masks's tests.
Owner: P4.

    python -m pytest backend/services/test_bitemporal.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.bitemporal import (  # noqa: E402
    class_change,
    classify_pixels,
    date_classes_from_optical,
    date_classes_from_sar,
    draw_region_boxes,
    find_changed_regions,
    warp_and_compare_class,
    warp_mask_to_common_grid,
)
from backend.services.landcover import LandcoverResult  # noqa: E402
from backend.services.sar import SarResult  # noqa: E402

TOLERANCE = 1.0


def _landcover_result(**overrides) -> LandcoverResult:
    defaults = dict(
        water_percentage=10.0, water_method="NDWI", water_is_true_index=True, water_threshold=0.0,
        vegetation_percentage=20.0, vegetation_method="NDVI", vegetation_is_true_index=True,
        vegetation_threshold=0.2,
        built_up_percentage=30.0, built_up_method="rule-based",
        water_mask=np.zeros((10, 10), dtype=bool),
        vegetation_mask=np.zeros((10, 10), dtype=bool),
        built_up_mask=np.zeros((10, 10), dtype=bool),
        valid_mask=np.ones((10, 10), dtype=bool),
    )
    defaults.update(overrides)
    return LandcoverResult(**defaults)


def _sar_result(**overrides) -> SarResult:
    defaults = dict(
        water_percentage=15.0, water_method="Otsu", water_threshold_db=-15.0, water_threshold_source="otsu",
        built_up_percentage=25.0, built_up_method="percentile", built_up_threshold_db=-5.0,
        built_up_threshold_source="percentile",
        water_mask=np.zeros((10, 10), dtype=bool),
        built_up_mask=np.zeros((10, 10), dtype=bool),
        valid_mask=np.ones((10, 10), dtype=bool),
    )
    defaults.update(overrides)
    return SarResult(**defaults)


# --- DateClasses normalisation ---------------------------------------------

def test_date_classes_from_optical_maps_all_three_classes():
    lc = _landcover_result()
    dc = date_classes_from_optical(lc)
    assert dc.water_pct == 10.0
    assert dc.vegetation_pct == 20.0
    assert dc.built_up_pct == 30.0


def test_date_classes_from_sar_forces_vegetation_not_computable():
    sar = _sar_result()
    dc = date_classes_from_sar(sar)
    assert dc.vegetation_pct is None
    assert dc.vegetation_mask is None
    assert "not computable" in dc.vegetation_method.lower()
    assert "sar" in dc.vegetation_method.lower()
    assert dc.water_pct == 15.0
    assert dc.built_up_pct == 25.0


# --- class_change ------------------------------------------------------------

def test_class_change_increased():
    c = class_change(10.0, 20.0, TOLERANCE)
    assert c.direction == "increased"
    assert c.change_pct_points == 10.0


def test_class_change_decreased():
    c = class_change(20.0, 10.0, TOLERANCE)
    assert c.direction == "decreased"
    assert c.change_pct_points == -10.0


def test_class_change_unchanged_within_tolerance():
    c = class_change(20.0, 20.5, TOLERANCE)
    assert c.direction == "unchanged"


def test_class_change_none_when_either_input_is_none():
    assert class_change(None, 20.0, TOLERANCE).direction is None
    assert class_change(10.0, None, TOLERANCE).direction is None
    c = class_change(None, None, TOLERANCE)
    assert c.before_pct is None and c.after_pct is None and c.change_pct_points is None


# --- warp_mask_to_common_grid -------------------------------------------------

def test_warp_mask_resize_only_no_homography():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    resized = warp_mask_to_common_grid(mask, (20, 20), homography=None)
    assert resized.shape == (20, 20)
    assert resized.dtype == bool
    assert resized[4:10, 4:10].any()  # the True block scaled up


def test_warp_mask_with_homography_translates():
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:5, 0:5] = True
    # translate everything 10px right/down
    homography = np.array([[1, 0, 10], [0, 1, 10], [0, 0, 1]], dtype=np.float64)
    warped = warp_mask_to_common_grid(mask, (20, 20), homography=homography)
    assert warped[10:15, 10:15].all()
    assert not warped[0:5, 0:5].any()


# --- warp_and_compare_class: overlap vs. native-frame distinction ------------

def test_warp_and_compare_class_uses_overlap_not_native_frame():
    # before: fully "on" in native 10x10; after: fully "on" in native 10x10 too,
    # but valid_mask excludes half the common grid -- percentages must be
    # computed only over the valid half, not the full frame (both are 100%
    # either way here, so instead make the masks differ inside vs outside the
    # valid region to prove the restriction actually applies).
    before_mask = np.zeros((10, 10), dtype=bool)
    before_mask[:, :5] = True   # left half "on"
    after_mask = np.zeros((10, 10), dtype=bool)
    after_mask[:, 5:] = True    # right half "on"

    valid_left_only = np.zeros((10, 10), dtype=bool)
    valid_left_only[:, :5] = True

    result = warp_and_compare_class(before_mask, after_mask, None, (10, 10), valid_left_only, TOLERANCE)
    # over the valid (left) region: before is 100% on, after is 0% on
    assert result.change.before_pct == 100.0
    assert result.change.after_pct == 0.0
    assert result.change.direction == "decreased"


def test_warp_and_compare_class_none_when_either_mask_none():
    valid = np.ones((10, 10), dtype=bool)
    result = warp_and_compare_class(None, np.zeros((10, 10), dtype=bool), None, (10, 10), valid, TOLERANCE)
    assert result.before_mask is None and result.after_mask is None
    assert result.change.direction is None


def test_warp_and_compare_class_none_when_zero_valid_pixels():
    mask = np.ones((10, 10), dtype=bool)
    valid = np.zeros((10, 10), dtype=bool)
    result = warp_and_compare_class(mask, mask, None, (10, 10), valid, TOLERANCE)
    assert result.change.direction is None
    assert result.change.before_pct is None


# --- classify_pixels -----------------------------------------------------------

def test_classify_pixels_priority_water_beats_vegetation_beats_built_up():
    shape = (4, 4)
    water = np.zeros(shape, dtype=bool)
    water[0, 0] = True
    vegetation = np.zeros(shape, dtype=bool)
    vegetation[0, 0] = True  # overlaps water -- water should win
    vegetation[1, 1] = True
    built_up = np.zeros(shape, dtype=bool)
    built_up[1, 1] = True    # overlaps vegetation -- vegetation should win
    built_up[2, 2] = True

    labels = classify_pixels(water, vegetation, built_up)
    assert labels[0, 0] == 1  # water wins
    assert labels[1, 1] == 2  # vegetation wins over built_up
    assert labels[2, 2] == 3  # built_up alone
    assert labels[3, 3] == 0  # none


def test_classify_pixels_handles_none_water_and_vegetation():
    built_up = np.zeros((3, 3), dtype=bool)
    built_up[0, 0] = True
    labels = classify_pixels(None, None, built_up)
    assert labels[0, 0] == 3
    assert labels[1, 1] == 0


# --- find_changed_regions ------------------------------------------------------

def test_find_changed_regions_two_blobs_sorted_by_area():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 255       # small blob, area 4
    mask[10:16, 10:16] = 255  # large blob, area 36
    regions = find_changed_regions(mask, min_size_fraction=0.001)
    assert len(regions) == 2
    assert regions[0].area_px == 36
    assert regions[1].area_px == 4
    assert regions[0].bbox == (10, 10, 6, 6)


def test_find_changed_regions_filters_below_min_size():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 255  # area 4, out of 400 -> 1%, below a 2.5% cutoff
    regions = find_changed_regions(mask, min_size_fraction=0.025)
    assert regions == []


def test_find_changed_regions_min_size_scales_with_frame_size():
    """T10: min_size_fraction, not a fixed pixel count -- the identical 4px
    blob is significant enough to keep on a small frame but noise-sized on a
    much bigger one, at the same fraction."""
    fraction = 0.001  # 0.1% of the frame

    small_mask = np.zeros((20, 20), dtype=np.uint8)  # 400px frame -> cutoff = max(1, round(400*0.001)) = 1px
    small_mask[0:2, 0:2] = 255  # area 4
    assert len(find_changed_regions(small_mask, min_size_fraction=fraction)) == 1

    big_mask = np.zeros((200, 200), dtype=np.uint8)  # 40,000px frame -> cutoff = round(40000*0.001) = 40px
    big_mask[0:2, 0:2] = 255  # same area, 4px
    assert find_changed_regions(big_mask, min_size_fraction=fraction) == []


def test_find_changed_regions_area_km2_only_with_gsd():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    no_gsd = find_changed_regions(mask, min_size_fraction=0.001, gsd_m=None)
    assert no_gsd[0].area_km2 is None
    with_gsd = find_changed_regions(mask, min_size_fraction=0.001, gsd_m=10.0)
    assert with_gsd[0].area_km2 == round(4 * 10.0 * 10.0 / 1_000_000, 6)


def test_find_changed_regions_dominant_class_change_when_determinable():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    before_labels = np.full((10, 10), 2, dtype=np.uint8)   # vegetation everywhere
    after_labels = np.full((10, 10), 3, dtype=np.uint8)    # built-up everywhere
    regions = find_changed_regions(mask, min_size_fraction=0.001,
                                    before_labels=before_labels, after_labels=after_labels)
    assert regions[0].dominant_class_change == "vegetation -> built-up"


def test_find_changed_regions_dominant_class_change_none_when_labels_missing_or_equal():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    regions_no_labels = find_changed_regions(mask, min_size_fraction=0.001)
    assert regions_no_labels[0].dominant_class_change is None

    same_labels = np.full((10, 10), 1, dtype=np.uint8)
    regions_same = find_changed_regions(mask, min_size_fraction=0.001,
                                         before_labels=same_labels, after_labels=same_labels)
    assert regions_same[0].dominant_class_change is None


# --- draw_region_boxes ---------------------------------------------------------

def test_draw_region_boxes_draws_without_mutating_input():
    overlay = np.zeros((20, 20, 3), dtype=np.uint8)
    original = overlay.copy()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    regions = find_changed_regions(mask, min_size_fraction=0.001)

    boxed = draw_region_boxes(overlay, regions)
    assert np.array_equal(overlay, original)  # input untouched
    assert not np.array_equal(boxed, original)  # something was drawn
