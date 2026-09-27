"""SAR array processing: dB conversion, Lee speckle filter, water/built-up
extraction via classical thresholds.

Owner: P4.  `extract_sar()` mirrors `landcover.py::extract_optical`'s shape
and conventions (output_dir/job_id/prefix, a plain-dataclass result with
percentages + method labels + evidence + warnings) so T8's optical-SAR
fusion service can call both side by side.

Honesty rules this file exists to enforce (CLAUDE.md non-negotiable rules):
  - Zero/negative values can't be logged; they're excluded as invalid, never
    replaced with a made-up epsilon.
  - RasterInput.nodata is excluded from thresholds and percentages.
  - Otsu and a fixed percentile are both purely relative/adaptive -- each
    will always find *some* split point, even in a scene with no water or no
    genuine bright targets at all. Both get an absolute dB guard on top, so a
    scene without a given class actually reports ~0% for it instead of a
    threshold-shaped hallucination. The guard is a *stricter* fallback in
    both directions, never a more lenient one.
  - Otsu threshold, percentile threshold and Lee filter are all classical/
    rule-based techniques; method strings say so explicitly (CLAUDE.md rule 2).

`SarResult` carries raw `np.ndarray` mask/valid_mask fields (for T8's optical/
SAR fusion) -- note for any future caller: a plain `dataclasses.asdict()` of
this type will include those raw arrays, not just the summary fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:                                       # package import (backend.services.sar)
    from ..contract import Evidence
    from ..rsio.raster import RasterInput, looks_like_sar_db
except ImportError:                        # bare import, e.g. a future pipeline.py-style caller
    from contract import Evidence
    from rsio.raster import RasterInput, looks_like_sar_db

_VALID_INPUT_TYPES = {"intensity", "amplitude"}

# Rule-of-thumb C-band backscatter guards (dB). Not measured from this
# dataset -- documented defaults a caller can override via extract_sar().
_DEFAULT_WATER_MAX_DB = -15.0
_DEFAULT_BUILT_UP_MIN_DB = -5.0
_DEFAULT_BUILT_UP_PERCENTILE = 90.0
_DEFAULT_LEE_WINDOW = 5


@dataclass
class SarResult:
    water_percentage: float
    water_method: str
    water_threshold_db: float | None
    water_threshold_source: str | None       # "otsu" | "guard" | None (no valid pixels)
    built_up_percentage: float
    built_up_method: str
    built_up_threshold_db: float | None
    built_up_threshold_source: str | None    # "percentile" | "guard" | None
    water_mask: np.ndarray | None = None
    built_up_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def to_db(array: np.ndarray, invalid_mask: np.ndarray | None = None,
          input_type: str | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    """Convert raw SAR intensity/amplitude to dB, unless it already looks
    like dB.  Returns (db_array, valid_mask, info, warnings).

    `invalid_mask` (e.g. from RasterInput.nodata) is combined with non-finite
    values first; zero/negative values found afterwards are excluded the
    same way, never replaced with a fudge value, since a log of a fabricated
    positive number would itself be a fabricated number.
    """
    warnings: list[str] = []
    invalid = (invalid_mask.copy() if invalid_mask is not None
               else np.zeros(array.shape, dtype=bool))
    invalid |= ~np.isfinite(array)

    if looks_like_sar_db(array[~invalid]):
        warnings.append(
            "Input already appears to be in dB range (negative, within a "
            "plausible SAR backscatter range); to_db skipped conversion."
        )
        result = np.where(invalid, np.nan, array).astype(np.float32)
        return result, ~invalid, {"skipped": True, "input_type_used": None, "assumed": False}, warnings

    if input_type is not None and input_type not in _VALID_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {_VALID_INPUT_TYPES}, got {input_type!r}")
    assumed = input_type is None
    resolved_type = input_type or "intensity"

    non_positive = (array <= 0) & ~invalid
    if np.any(non_positive):
        warnings.append(
            f"{int(np.sum(non_positive))} pixel(s) were zero or negative and cannot be "
            "converted to dB; excluded as invalid rather than replaced with a made-up value."
        )
    invalid = invalid | non_positive
    valid = ~invalid

    multiplier = 10.0 if resolved_type == "intensity" else 20.0
    result = np.full(array.shape, np.nan, dtype=np.float32)
    result[valid] = multiplier * np.log10(array[valid].astype(np.float64))

    if assumed:
        warnings.append(
            "input_type not specified; assumed 'intensity' (10*log10(x)). Pass "
            "input_type='amplitude' explicitly if this is amplitude data (20*log10(x))."
        )

    return result, valid, {"skipped": False, "input_type_used": resolved_type, "assumed": assumed}, warnings


def lee_filter(db_array: np.ndarray, valid_mask: np.ndarray, window: int = _DEFAULT_LEE_WINDOW) -> np.ndarray:
    """Additive-noise-model Lee speckle filter (mean_local + W*(x-mean_local),
    W = var_local/(var_local+noise_var)), applied on the dB plane -- SAR
    speckle is multiplicative in the linear domain, but a log transform makes
    it approximately additive, the usual practical justification for running
    Lee-style filtering on dB/log data rather than linear.

    Simplification, disclosed rather than hidden: `noise_var` here is the
    global variance of the valid dB pixels, a stand-in for a properly
    ENL-estimated speckle noise variance. This is the common "generic Lee
    filter" formulation, not the full multiplicative-noise-model derivation.

    Invalid (nodata) pixels inside a window are excluded from its local mean/
    variance via a per-window valid-count divisor, not treated as zero.
    """
    arr = np.where(valid_mask, db_array, 0.0).astype(np.float32)
    valid_f = valid_mask.astype(np.float32)

    count = cv2.boxFilter(valid_f, ddepth=-1, ksize=(window, window), normalize=False)
    count_safe = np.maximum(count, 1.0)  # only reachable at already-invalid centres

    sum_local = cv2.boxFilter(arr, ddepth=-1, ksize=(window, window), normalize=False)
    sum_sq_local = cv2.boxFilter(arr ** 2, ddepth=-1, ksize=(window, window), normalize=False)
    mean_local = sum_local / count_safe
    var_local = np.maximum(sum_sq_local / count_safe - mean_local ** 2, 0.0)

    valid_values = db_array[valid_mask]
    noise_var = float(np.var(valid_values)) if valid_values.size > 0 else 0.0
    if noise_var < 1e-6:
        return np.where(valid_mask, db_array, np.nan).astype(np.float32)

    weight = var_local / (var_local + noise_var)
    filtered = mean_local + weight * (arr - mean_local)
    return np.where(valid_mask, filtered, np.nan).astype(np.float32)


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on the real-valued histogram directly, so the returned
    threshold is in the same units as `values` (dB) -- no uint8 rescale/remap
    that would lose precision in the reported threshold."""
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    sum_all = np.sum(hist * bin_centers)

    weight_bg = np.cumsum(hist)
    sum_bg = np.cumsum(hist * bin_centers)
    weight_fg = total - weight_bg

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_class_variance = np.nan_to_num(between_class_variance)

    idx = int(np.argmax(between_class_variance))
    return float(bin_centers[idx])


def water_mask(filtered_db: np.ndarray, valid_mask: np.ndarray,
               water_max_db: float = _DEFAULT_WATER_MAX_DB) -> tuple[np.ndarray, float | None, str | None]:
    """Otsu threshold on low backscatter -> water mask, with an absolute
    guard: Otsu always finds *some* split point, even with no real water in
    the scene, so its raw output alone can't distinguish "there is water
    here" from "this histogram merely has two halves." threshold =
    min(otsu, water_max_db) -- if Otsu's own split is already at or below the
    guard, it's used as-is (source="otsu"); otherwise the guard wins
    (source="guard"), which only ever makes the mask *stricter*, never more
    lenient. This guard is independent of, and does not fix, the separate
    fact that Otsu's split is not guaranteed to land in the water/non-water
    gap specifically on a histogram with more than two populations -- for
    this module's own tests it does, because that's the widest gap there,
    not because Otsu guarantees it in general.
    """
    values = filtered_db[valid_mask]
    if values.size == 0:
        return np.zeros(valid_mask.shape, dtype=bool), None, None

    otsu_threshold = _otsu_threshold(values)
    if otsu_threshold <= water_max_db:
        threshold, source = otsu_threshold, "otsu"
    else:
        threshold, source = water_max_db, "guard"

    mask = (filtered_db < threshold) & valid_mask
    return mask, threshold, source


def built_up_mask(filtered_db: np.ndarray, valid_mask: np.ndarray, water: np.ndarray,
                   percentile: float = _DEFAULT_BUILT_UP_PERCENTILE,
                   built_up_min_db: float = _DEFAULT_BUILT_UP_MIN_DB
                   ) -> tuple[np.ndarray, float | None, str | None]:
    """High-backscatter percentile threshold -> built-up mask, with an
    absolute guard: a fixed percentile always flags ~(100-percentile)% of
    pixels as "high," even with no genuinely bright targets at all.
    threshold = max(percentile_value, built_up_min_db) -- if the percentile
    is already at or above the guard, it's used as-is (source="percentile");
    otherwise the guard wins (source="guard"), again only ever stricter.
    Excludes water defensively, mirroring landcover.py's non-water/
    non-vegetation restriction on built-up. Separate, still-true limitation
    the guard does not address: when the relative path *is* the one used, a
    fixed percentile still only resolves roughly the top (100-percentile)% of
    all pixels, so it under-reports a genuine built-up region much larger
    than that fraction of the scene.
    """
    values = filtered_db[valid_mask]
    if values.size == 0:
        return np.zeros(valid_mask.shape, dtype=bool), None, None

    percentile_threshold = float(np.percentile(values, percentile))
    if percentile_threshold >= built_up_min_db:
        threshold, source = percentile_threshold, "percentile"
    else:
        threshold, source = built_up_min_db, "guard"

    mask = (filtered_db > threshold) & valid_mask & ~water
    return mask, threshold, source


def _to_uint8_display(filtered_db: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """NaN-safe min-max stretch for the filtered-SAR display PNG.  Unlike
    orchestrator.py's _to_uint8_gray (fine there because that data has no
    NaN), the Lee-filtered dB plane has NaN at invalid positions -- plain
    .min()/.max() would let a single NaN poison the whole rescale."""
    if not np.any(valid_mask):
        return np.zeros(filtered_db.shape, dtype=np.uint8)
    lo = float(np.nanmin(filtered_db[valid_mask]))
    hi = float(np.nanmax(filtered_db[valid_mask]))
    if hi - lo < 1e-6:
        scaled = np.zeros(filtered_db.shape, dtype=np.float32)
    else:
        scaled = (filtered_db - lo) / (hi - lo) * 255.0
    out = np.where(valid_mask, scaled, 0.0)
    return np.nan_to_num(out, nan=0.0).astype(np.uint8)


def extract_sar(r: RasterInput, output_dir: str | Path, job_id: str | None = None,
                 prefix: str = "", input_type: str | None = None,
                 lee_window: int = _DEFAULT_LEE_WINDOW,
                 built_up_percentile: float = _DEFAULT_BUILT_UP_PERCENTILE,
                 water_max_db: float = _DEFAULT_WATER_MAX_DB,
                 built_up_min_db: float = _DEFAULT_BUILT_UP_MIN_DB,
                 band_index: int = 0) -> SarResult:
    """Water/built-up masks + percentages + method labels for one SAR image.

    `output_dir`/`job_id`/`prefix` conventions match `landcover.py::extract_optical`
    exactly, so T8 can call both into the same job folder without filename
    collisions: `output_dir` must already exist; `job_id` defaults to its
    folder name; `prefix` namespaces the three PNG filenames.
    """
    output_dir = Path(output_dir)
    job_id = job_id or output_dir.name

    raw = r.array[band_index]
    invalid_mask = ~np.isfinite(raw)
    if r.nodata is not None:
        invalid_mask = invalid_mask | (raw == r.nodata)

    db, valid_mask, db_info, db_warnings = to_db(raw, invalid_mask=invalid_mask, input_type=input_type)
    filtered = lee_filter(db, valid_mask, window=lee_window)

    water, water_threshold, water_source = water_mask(filtered, valid_mask, water_max_db=water_max_db)
    built_up, built_up_threshold, built_up_source = built_up_mask(
        filtered, valid_mask, water, percentile=built_up_percentile, built_up_min_db=built_up_min_db
    )

    total_valid = int(np.sum(valid_mask))
    water_pct = (int(np.sum(water)) / total_valid * 100) if total_valid else 0.0
    built_up_pct = (int(np.sum(built_up)) / total_valid * 100) if total_valid else 0.0

    warnings = list(db_warnings)
    if water_source == "guard":
        warnings.append(
            f"Otsu threshold for water ({_otsu_threshold(filtered[valid_mask]):.1f} dB) was "
            f"above the water_max_db guard ({water_max_db} dB); using the guard instead, so a "
            "scene with no real low-backscatter cluster doesn't report a fabricated water estimate."
        )
    if built_up_source == "guard":
        warnings.append(
            f"The {built_up_percentile}th percentile ({float(np.percentile(filtered[valid_mask], built_up_percentile)):.1f} dB) "
            f"was below the built_up_min_db guard ({built_up_min_db} dB); using the guard instead, so "
            "a scene with no genuinely bright targets doesn't report a fabricated built-up estimate."
        )

    water_method = (
        f"rule-based: Otsu threshold (low backscatter) on Lee-filtered dB (window={lee_window}), "
        f"capped at water_max_db={water_max_db} dB (C-band rule-of-thumb)"
    )
    built_up_method = (
        f"rule-based: backscatter above the {built_up_percentile}th percentile (high backscatter) "
        f"on Lee-filtered dB (window={lee_window}), floored at built_up_min_db={built_up_min_db} dB "
        "(C-band rule-of-thumb)"
    )

    files = {
        "sar_filtered": (_to_uint8_display(filtered, valid_mask), "image", "Filtered SAR (dB)"),
        "water_mask": (water.astype(np.uint8) * 255, "mask", "Water mask (SAR)"),
        "built_up_mask": (built_up.astype(np.uint8) * 255, "mask", "Built-up mask (SAR)"),
    }
    evidence: list[Evidence] = []
    for name, (image, kind, label) in files.items():
        filename = f"{prefix}{name}.png"
        cv2.imwrite(str(output_dir / filename), image)
        evidence.append(Evidence(
            id=f"{prefix}{name}", kind=kind, label=label, modality="sar",
            url=f"/outputs/{job_id}/{filename}",
        ))

    return SarResult(
        water_percentage=round(water_pct, 2),
        water_method=water_method,
        water_threshold_db=water_threshold,
        water_threshold_source=water_source,
        built_up_percentage=round(built_up_pct, 2),
        built_up_method=built_up_method,
        built_up_threshold_db=built_up_threshold,
        built_up_threshold_source=built_up_source,
        water_mask=water,
        built_up_mask=built_up,
        valid_mask=valid_mask,
        evidence=evidence,
        warnings=warnings,
    )
