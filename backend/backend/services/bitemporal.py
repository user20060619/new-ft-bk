"""Bitemporal (before/after) helpers shared by the change/vegetation/water
handlers: modality-agnostic per-date class normalisation, percentage-point
change with a stated tolerance, mask warping onto a common aligned grid, and
connected-component region detection.

Owner: P4.  Pure, trace-agnostic functions only -- `orchestrator.py` owns
every `ExecutionTrace` step and file write, mirroring `landcover.py`/`sar.py`/
`optical_sar.py`'s existing shape.

Honesty rules this file exists to enforce (CLAUDE.md non-negotiable rules):
  - A class that can't be computed on one or both dates (panchromatic optical,
    or vegetation on SAR) is `None`, never a fabricated `0.0`/"unchanged".
  - Before/after percentages for a change comparison are computed over the
    aligned overlap (`valid_mask`, from `geo_service.align_images`), never
    each date's independent native full frame -- two rasters can share pixel
    dimensions and still not cover the same ground.
  - Direction (increased/decreased/unchanged) uses one shared, stated
    tolerance (`fusion.explain.SIGNIFICANCE_PCT`), not an invented threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

try:                                       # package import (backend.services.bitemporal)
    from ..contract import Evidence
except ImportError:                        # bare import, e.g. a future pipeline.py-style caller
    from contract import Evidence

from .landcover import LandcoverResult
from .sar import SarResult

_DEFAULT_MIN_REGION_PX = 20

CLASS_LABELS: dict[int, str] = {0: "none", 1: "water", 2: "vegetation", 3: "built-up"}

_VEGETATION_NOT_COMPUTABLE_FROM_SAR = (
    "not computable: vegetation has no SAR equivalent (no spectral colour "
    "information); SAR supports water and built-up only"
)


@dataclass
class DateClasses:
    """One date's water/vegetation/built-up masks + percentages, normalised
    from either a `LandcoverResult` (optical) or a `SarResult` (SAR) so the
    rest of this module doesn't need to know which one produced it."""
    water_pct: float | None
    water_mask: np.ndarray | None
    water_method: str
    vegetation_pct: float | None
    vegetation_mask: np.ndarray | None
    vegetation_method: str
    built_up_pct: float
    built_up_mask: np.ndarray
    built_up_method: str
    valid_mask: np.ndarray | None
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def date_classes_from_optical(r: LandcoverResult) -> DateClasses:
    return DateClasses(
        water_pct=r.water_percentage, water_mask=r.water_mask, water_method=r.water_method,
        vegetation_pct=r.vegetation_percentage, vegetation_mask=r.vegetation_mask,
        vegetation_method=r.vegetation_method,
        built_up_pct=r.built_up_percentage, built_up_mask=r.built_up_mask,
        built_up_method=r.built_up_method,
        valid_mask=r.valid_mask, evidence=list(r.evidence), warnings=list(r.warnings),
    )


def date_classes_from_sar(r: SarResult) -> DateClasses:
    return DateClasses(
        water_pct=r.water_percentage, water_mask=r.water_mask, water_method=r.water_method,
        vegetation_pct=None, vegetation_mask=None,
        vegetation_method=_VEGETATION_NOT_COMPUTABLE_FROM_SAR,
        built_up_pct=r.built_up_percentage, built_up_mask=r.built_up_mask,
        built_up_method=r.built_up_method,
        valid_mask=r.valid_mask, evidence=list(r.evidence), warnings=list(r.warnings),
    )


@dataclass
class ClassChange:
    before_pct: float | None
    after_pct: float | None
    change_pct_points: float | None
    direction: str | None  # "increased" | "decreased" | "unchanged" | None


def class_change(before_pct: float | None, after_pct: float | None, tolerance: float) -> ClassChange:
    """Percentage-point delta + direction against a stated `tolerance`.
    `None` in either input -- not computable on one or both dates -- propagates
    to a `None` direction, never a `0.0`-point "unchanged" standing in for
    "we don't know"."""
    if before_pct is None or after_pct is None:
        return ClassChange(before_pct=None, after_pct=None, change_pct_points=None, direction=None)

    delta = round(after_pct - before_pct, 2)
    if delta > tolerance:
        direction = "increased"
    elif delta < -tolerance:
        direction = "decreased"
    else:
        direction = "unchanged"
    return ClassChange(before_pct=before_pct, after_pct=after_pct,
                        change_pct_points=delta, direction=direction)


def warp_mask_to_common_grid(mask: np.ndarray, target_size: tuple[int, int],
                              homography: np.ndarray | None = None) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask onto `target_size`, then an
    optional homography warp -- nearest, not bilinear, because a mask is
    boolean data with no valid "in-between" value to blend towards (the same
    reasoning `rsio/raster.py` already applies to SAR dB resampling)."""
    width, height = target_size
    resized = cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
    if homography is not None:
        resized = cv2.warpPerspective(resized, homography, (width, height))
    return resized > 0


@dataclass
class WarpedClassMasks:
    before_mask: np.ndarray | None  # on the common grid, in the after date's frame
    after_mask: np.ndarray | None   # on the common grid
    change: ClassChange


def warp_and_compare_class(before_mask_native: np.ndarray | None, after_mask_native: np.ndarray | None,
                            homography: np.ndarray | None, target_size: tuple[int, int],
                            valid_mask: np.ndarray, tolerance: float) -> WarpedClassMasks:
    """Warps both dates' native-resolution class masks onto the common grid
    (before through `homography`, the same transform the pixel-diff alignment
    already computed; after by resize only, since it already defines the
    destination frame) and compares them **restricted to `valid_mask`** -- the
    region where both aligned dates actually have data -- not each date's
    independent native full frame.  Both `before_pct`/`after_pct` use the same
    `count(valid_mask)` denominator, so the delta is a fair before/after
    comparison over the area that was actually compared.

    Returns an all-`None` `ClassChange` (masks also `None`) when either native
    mask is `None` (the class wasn't computable on one or both dates), or when
    `valid_mask` has zero valid pixels (near-total misalignment) -- never a
    divide-by-zero, never a fabricated percentage."""
    if before_mask_native is None or after_mask_native is None:
        return WarpedClassMasks(None, None, ClassChange(None, None, None, None))

    valid_mask = valid_mask.astype(bool)  # geo_service.align_images' valid_mask is 0/255 uint8, not 0/1
    total_valid = int(np.sum(valid_mask))
    if total_valid == 0:
        return WarpedClassMasks(None, None, ClassChange(None, None, None, None))

    before_common = warp_mask_to_common_grid(before_mask_native, target_size, homography)
    after_common = warp_mask_to_common_grid(after_mask_native, target_size, None)

    before_pct = float(np.sum(before_common & valid_mask)) / total_valid * 100
    after_pct = float(np.sum(after_common & valid_mask)) / total_valid * 100

    return WarpedClassMasks(before_common, after_common,
                             class_change(round(before_pct, 2), round(after_pct, 2), tolerance))


def classify_pixels(water: np.ndarray | None, vegetation: np.ndarray | None,
                     built_up: np.ndarray) -> np.ndarray:
    """Per-pixel integer class label, priority water > vegetation > built_up >
    none (0/1/2/3).  `water`/`vegetation` may be `None` (not computable on
    this date) -- treated as all-`False` for labelling purposes only.  This
    never feeds a reported percentage, only a changed region's qualitative
    "dominant class" string, so treating "unknown colour" as "not this class"
    narrows which label a region can get without misrepresenting any number."""
    shape = built_up.shape
    labels = np.zeros(shape, dtype=np.uint8)
    labels[built_up] = 3
    if vegetation is not None:
        labels[vegetation] = 2
    if water is not None:
        labels[water] = 1
    return labels


@dataclass
class RegionInfo:
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coordinates
    area_px: int
    area_km2: float | None
    dominant_class_change: str | None


def _dominant_class_change(region_mask: np.ndarray, before_labels: np.ndarray | None,
                            after_labels: np.ndarray | None) -> str | None:
    if before_labels is None or after_labels is None:
        return None
    before_values = before_labels[region_mask]
    after_values = after_labels[region_mask]
    if before_values.size == 0:
        return None
    before_mode = int(np.bincount(before_values).argmax())
    after_mode = int(np.bincount(after_values).argmax())
    if before_mode == after_mode:
        return None
    return f"{CLASS_LABELS[before_mode]} -> {CLASS_LABELS[after_mode]}"


def find_changed_regions(diff_mask: np.ndarray, min_size_px: int = _DEFAULT_MIN_REGION_PX,
                          gsd_m: float | None = None,
                          before_labels: np.ndarray | None = None,
                          after_labels: np.ndarray | None = None) -> list[RegionInfo]:
    """Connected components of `diff_mask` (uint8, 0/255 or bool) above
    `min_size_px`, largest first.  `area_km2` only when `gsd_m` is known --
    never a guessed pixel size.  `dominant_class_change` is the mode label
    inside the region on each date, `None` when either label array is missing
    or the two modes are equal (a real intensity change with no corresponding
    tracked-class swap)."""
    binary = (diff_mask > 0).astype(np.uint8)
    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    regions: list[RegionInfo] = []
    for label in range(1, num_labels):  # 0 is background
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < min_size_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        region_mask = labels_img == label
        area_km2 = round(area_px * gsd_m * gsd_m / 1_000_000, 6) if gsd_m is not None else None
        regions.append(RegionInfo(
            bbox=(x, y, w, h), area_px=area_px, area_km2=area_km2,
            dominant_class_change=_dominant_class_change(region_mask, before_labels, after_labels),
        ))

    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions


def draw_region_boxes(overlay_bgr: np.ndarray, regions: list[RegionInfo]) -> np.ndarray:
    """Draws a rectangle per region on a copy of `overlay_bgr`."""
    out = overlay_bgr.copy()
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out
