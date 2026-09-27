"""Optical array processing (vegetation/water indices + change alignment).

Owner: P4.  Functions here take already-loaded `RasterInput` arrays
(backend.rsio.raster), never file paths -- the "uploads forced to .jpg, read
with cv2.imread" problem (CLAUDE.md known problems) lived in this module
before this refactor.

Honesty rules this file exists to enforce (CLAUDE.md non-negotiable rules):
  - True NDVI/NDWI only when a NIR band is identifiable (4+ band optical,
    default B,G,R,NIR order); RGB-only data gets a clearly-labelled
    "vegetation_proxy"/"water_proxy" colour index instead -- never NDVI/NDWI.
  - area_changed_km2 is reported only when both inputs agree on a real
    gsd_m; there is no default pixel size anywhere in this module.
  - No hardcoded confidence scores.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:                                       # package import (backend.services.geo_service)
    from ..rsio.raster import RasterInput
except ImportError:                        # bare import (services.geo_service, e.g. pipeline.py)
    from rsio.raster import RasterInput

# Default multispectral band order assumed when a caller doesn't specify one
# (the common Cartosat-2S/WorldView-style MX layout: Blue, Green, Red, NIR).
DEFAULT_BAND_ORDER: dict[str, int] = {"blue": 0, "green": 1, "red": 2, "nir": 3}

# True band order for plain 3-band RGB data (no NIR), verified against
# rsio/raster.py's actual loaders: PIL's "RGB" mode is R,G,B (array[0] is
# genuinely Red, not Blue -- confirmed directly against a real JPEG), and
# rasterio reads a plain 3-band file in whatever order it was written, which
# for the PNG/JPEG "public benchmark images" CLAUDE.md describes is the same
# R,G,B a photo viewer would show. This used to read {"blue":0,"green":1,
# "red":2} -- backwards for every real photo -- which made water_like_index's
# "green vs red" comparison actually compute green vs blue, flagging ~99% of
# a real city photo as water (T10). DEFAULT_BAND_ORDER (the 4-band NIR case)
# is a separate, documented assumption about a specific multispectral sensor
# family and is not affected by this fix.
RGB_PROXY_ORDER: dict[str, int] = {"red": 0, "green": 1, "blue": 2}

_NDVI_CHANGE_THRESHOLD = 0.1
_NDWI_CHANGE_THRESHOLD = 0.1
_VEGETATION_PROXY_CHANGE_THRESHOLD = 15.0  # excess-green index is on a ~[-510,510] scale
_WATER_PROXY_CHANGE_THRESHOLD = 0.1        # water proxy reuses the NDWI formula's [-1,1] scale

_PANCHROMATIC_VEG_CHANGE_WARNING = (
    "No colour bands available on at least one date (single/dual-band optical "
    "imagery); vegetation change cannot be computed as a colour proxy or NDVI."
)
_PANCHROMATIC_WATER_CHANGE_WARNING = (
    "No colour bands available on at least one date (single/dual-band optical "
    "imagery); water change cannot be computed as a colour proxy or NDWI."
)


def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    return (nir - red) / (nir + red + 1e-10)


def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """NDWI = (GREEN - NIR) / (GREEN + NIR)"""
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    return (green - nir) / (green + nir + 1e-10)


def calculate_area_percentage(mask: np.ndarray, pixel_size_meters: float | None = None
                               ) -> tuple[float, float | None]:
    """Percentage of `mask` that is non-zero, and area in km² -- but only when
    a real `pixel_size_meters` is supplied.  No default: silently assuming a
    pixel size is exactly the fabricated-number problem CLAUDE.md rules out."""
    total_pixels = mask.size
    changed_pixels = int(np.sum(mask > 0))
    changed_percentage = (changed_pixels / total_pixels) * 100 if total_pixels else 0.0
    if pixel_size_meters is None:
        return changed_percentage, None
    area_km2 = (changed_pixels * pixel_size_meters * pixel_size_meters) / 1_000_000
    return changed_percentage, area_km2


def _pct_change(before: float, after: float) -> float:
    if abs(before) > 1e-10:
        return round(((after - before) / abs(before)) * 100, 2)
    return 0.0


def _shared_gsd(before: RasterInput, after: RasterInput) -> float | None:
    """Only report a ground sample distance when both inputs genuinely agree
    on one -- guessing which of two different resolutions applies would
    itself be a fabricated number."""
    if before.gsd_m is None or after.gsd_m is None:
        return None
    if abs(before.gsd_m - after.gsd_m) > 1e-6:
        return None
    return before.gsd_m


def identify_band_order(r: RasterInput, band_order: dict[str, int] | None
                         ) -> dict[str, int] | None:
    """A NIR band is never guessed from pixel values -- only from band count
    plus a stated order (default B,G,R,NIR).  Returns None when a real NIR
    band can't be identified for this input, so the caller falls back to an
    RGB colour proxy instead of a fabricated NDVI/NDWI.  Public: reused by
    landcover.py for single-image (not just before/after pair) extraction."""
    if r.modality != "optical":
        return None
    order = band_order or DEFAULT_BAND_ORDER
    if "nir" not in order or r.bands < 4 or max(order.values()) >= r.bands:
        return None
    return order


def normalize_rgb(array: np.ndarray, order: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max stretch blue/green/red together (one combined lo/hi, not three
    independent per-band stretches, so their relative relationship survives)
    to a common ~0-255 float range.  Needed before any raw linear-combination
    colour proxy (e.g. excess-green): unlike NDVI/NDWI-style ratios, a linear
    combination's magnitude scales directly with source bit depth, and
    load_raster never rescales pixel values by dtype -- a 16-bit source would
    otherwise blow past a threshold calibrated for 8-bit-ish data."""
    blue = array[order["blue"]].astype(np.float32)
    green = array[order["green"]].astype(np.float32)
    red = array[order["red"]].astype(np.float32)
    lo = min(float(blue.min()), float(green.min()), float(red.min()))
    hi = max(float(blue.max()), float(green.max()), float(red.max()))
    if hi - lo < 1e-6:
        zeros = np.zeros(blue.shape, dtype=np.float32)
        return zeros, zeros, zeros
    scale = 255.0 / (hi - lo)
    return (blue - lo) * scale, (green - lo) * scale, (red - lo) * scale


def excess_green_index(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """(2*Green - Red - Blue) on bands normalised to a common ~0-255 scale: a
    common RGB vegetation colour proxy.  Not NDVI.  Public: reused by
    landcover.py."""
    blue, green, red = normalize_rgb(array, order)
    return (2 * green) - red - blue


def water_like_index(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """(Green-Red)/(Green+Red): reuses the NDWI formula's shape on colour
    bands instead of NIR, as a water-like proxy.  Not NDWI.  Already a ratio,
    so scale-invariant -- no normalisation needed.  Public: reused by
    landcover.py's before/after change-detection proxy (calculate_water_index),
    which needs a smooth continuous quantity, not a hard classification
    decision -- see blue_green_dominance_index for that."""
    return calculate_ndwi(array[order["green"]], array[order["red"]])


def blue_green_dominance_index(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """((max(Blue,Green) - Red) / (max(Blue,Green) + Red)) on bands normalised
    to a common ~0-255 scale: how strongly blue-or-green together dominate
    red.  Not NDWI.  Used (with a calibrated margin, not just > 0, plus a
    separate darkness condition) for water proxy *classification* in
    landcover.py -- water_like_index alone (green vs red only, no darkness
    requirement) flags almost any greenish-gray pixel in a real photo as
    water (T10).  max(blue,green), not green alone, so a genuinely
    blue-dominant water pixel (blue > green) is also caught."""
    blue, green, red = normalize_rgb(array, order)
    blue_green = np.maximum(blue, green)
    return (blue_green - red) / (blue_green + red + 1e-6)


def _warp_index_to_common_grid(index_array: np.ndarray, target_size: tuple[int, int],
                                homography: np.ndarray | None) -> np.ndarray:
    """Bilinear resize (+ optional homography warp) of a continuous index
    plane onto the common grid.  Bilinear, unlike bitemporal.py's nearest-
    neighbour mask warp: NDVI/NDWI/colour-proxy values are a continuous
    physical quantity where local averaging is the standard, correct thing to
    do -- unlike a boolean class mask or SAR's log-scale dB, which must never
    be blended."""
    width, height = target_size
    resized = cv2.resize(index_array.astype(np.float32), (width, height))
    if homography is not None:
        resized = cv2.warpPerspective(resized, homography, (width, height))
    return resized


def _index_change_stats(before_index: np.ndarray, after_index: np.ndarray,
                         gsd_m: float | None, threshold: float, *,
                         target_size: tuple[int, int] | None = None,
                         homography: np.ndarray | None = None,
                         valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """`target_size`/`homography`/`valid_mask` let a caller that has already
    aligned the two dates (SIFT/homography, same as `_change_handler`) report
    before/after values over the real overlap rather than each date's
    independent native full frame -- a plain `cv2.resize` alone doesn't
    guarantee the two planes cover the same ground.  Omitting them keeps the
    original behaviour exactly (`before_index`/`after_index` resized to
    `min(before, after)` dimensions, no valid-mask restriction) -- `valid` is
    then all-`True` over that full grid, which makes the unified formula below
    produce byte-identical numbers to the pre-existing implementation."""
    if target_size is not None:
        width, height = target_size
        ib = _warp_index_to_common_grid(before_index, (width, height), homography)
        ia = _warp_index_to_common_grid(after_index, (width, height), None)
        # geo_service.align_images' valid_mask is 0/255 uint8, not 0/1 -- normalise
        # defensively so a caller passing it straight through still counts correctly.
        valid = valid_mask.astype(bool) if valid_mask is not None else np.ones((height, width), dtype=bool)
    else:
        height = min(before_index.shape[0], after_index.shape[0])
        width = min(before_index.shape[1], after_index.shape[1])
        ib = cv2.resize(before_index.astype(np.float32), (width, height))
        ia = cv2.resize(after_index.astype(np.float32), (width, height))
        valid = np.ones((height, width), dtype=bool)

    total_valid = int(np.sum(valid))
    if total_valid == 0:
        return {"mean_before": None, "mean_after": None, "pct_change": None,
                "changed_area_pct": None}

    mean_before = float(np.mean(ib[valid]))
    mean_after = float(np.mean(ia[valid]))
    changed_count = int(np.sum((np.abs(ia - ib) > threshold) & valid))
    changed_pct = round(changed_count / total_valid * 100, 2)

    stats: dict[str, Any] = {
        "mean_before": mean_before,
        "mean_after": mean_after,
        "pct_change": _pct_change(mean_before, mean_after),
        "changed_area_pct": changed_pct,
    }
    if gsd_m is not None:
        stats["area_changed_km2"] = round(changed_count * gsd_m * gsd_m / 1_000_000, 4)
    return stats


def _finalize(result: dict[str, Any], stats: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    result["pct_change"] = stats["pct_change"]
    result["changed_area_pct"] = stats["changed_area_pct"]
    if "area_changed_km2" in stats:
        result["area_changed_km2"] = stats["area_changed_km2"]
    else:
        warnings.append(
            "Ground sample distance is unknown or differs between the two "
            "inputs; area_changed_km2 is not reported."
        )
    result["warnings"] = warnings
    return result


def calculate_vegetation_index(before: RasterInput, after: RasterInput,
                                band_order: dict[str, int] | None = None, *,
                                target_size: tuple[int, int] | None = None,
                                homography: np.ndarray | None = None,
                                valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """True NDVI when both inputs have an identifiable NIR band (4+ band
    optical, default B,G,R,NIR order); otherwise an RGB excess-green colour
    proxy ("vegetation_proxy"), clearly labelled and warned about.

    `target_size`/`homography`/`valid_mask`, when given (a caller that already
    aligned the two dates), make the before/after statistics overlap-aware --
    see `_index_change_stats`.  Omitted, behaviour is unchanged from before
    this parameter existed."""
    if before.bands < 3 or after.bands < 3:
        return {
            "is_true_index": False,
            "method": "not computable: single/dual-band (panchromatic) optical has no "
                      "colour information for a vegetation proxy or NDVI",
            "pct_change": None,
            "changed_area_pct": None,
            "warnings": [_PANCHROMATIC_VEG_CHANGE_WARNING],
        }

    order_before = identify_band_order(before, band_order)
    order_after = identify_band_order(after, band_order)
    gsd_m = _shared_gsd(before, after)
    warnings: list[str] = []
    is_true_index = bool(order_before and order_after)

    if is_true_index:
        before_index = calculate_ndvi(before.array[order_before["nir"]], before.array[order_before["red"]])
        after_index = calculate_ndvi(after.array[order_after["nir"]], after.array[order_after["red"]])
        threshold = _NDVI_CHANGE_THRESHOLD
        method = "NDVI = (NIR-Red)/(NIR+Red)"
    else:
        before_index = excess_green_index(before.array, RGB_PROXY_ORDER)
        after_index = excess_green_index(after.array, RGB_PROXY_ORDER)
        threshold = _VEGETATION_PROXY_CHANGE_THRESHOLD
        method = "vegetation_proxy: excess-green colour index (2*Green-Red-Blue)"
        warnings.append(
            "No NIR band identifiable (needs 4+ band optical imagery); reporting "
            "an RGB excess-green colour proxy as vegetation_proxy, not NDVI."
        )

    stats = _index_change_stats(before_index, after_index, gsd_m, threshold,
                                 target_size=target_size, homography=homography, valid_mask=valid_mask)

    result: dict[str, Any] = {"is_true_index": is_true_index, "method": method}
    if stats["mean_before"] is None:
        warnings.append(
            "no overlapping valid area between the two aligned images; "
            "vegetation change could not be computed"
        )
        result["pct_change"] = None
        result["changed_area_pct"] = None
        result["warnings"] = warnings
        return result

    if is_true_index:
        result["index_before"] = round(stats["mean_before"], 4)
        result["index_after"] = round(stats["mean_after"], 4)
    else:
        result["vegetation_proxy_before"] = round(stats["mean_before"], 2)
        result["vegetation_proxy_after"] = round(stats["mean_after"], 2)

    return _finalize(result, stats, warnings)


def calculate_water_index(before: RasterInput, after: RasterInput,
                           band_order: dict[str, int] | None = None, *,
                           target_size: tuple[int, int] | None = None,
                           homography: np.ndarray | None = None,
                           valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """True NDWI when both inputs have an identifiable NIR band; otherwise a
    green/red colour proxy ("water_proxy"), clearly labelled -- never called
    NDWI for RGB-only data.

    `target_size`/`homography`/`valid_mask`: see `calculate_vegetation_index`."""
    if before.bands < 3 or after.bands < 3:
        return {
            "is_true_index": False,
            "method": "not computable: single/dual-band (panchromatic) optical has no "
                      "colour information for a water proxy or NDWI",
            "pct_change": None,
            "changed_area_pct": None,
            "warnings": [_PANCHROMATIC_WATER_CHANGE_WARNING],
        }

    order_before = identify_band_order(before, band_order)
    order_after = identify_band_order(after, band_order)
    gsd_m = _shared_gsd(before, after)
    warnings: list[str] = []
    is_true_index = bool(order_before and order_after)

    if is_true_index:
        before_index = calculate_ndwi(before.array[order_before["green"]], before.array[order_before["nir"]])
        after_index = calculate_ndwi(after.array[order_after["green"]], after.array[order_after["nir"]])
        threshold = _NDWI_CHANGE_THRESHOLD
        method = "NDWI = (Green-NIR)/(Green+NIR)"
    else:
        before_index = water_like_index(before.array, RGB_PROXY_ORDER)
        after_index = water_like_index(after.array, RGB_PROXY_ORDER)
        threshold = _WATER_PROXY_CHANGE_THRESHOLD
        method = "water_proxy: green/red colour index (Green-Red)/(Green+Red)"
        warnings.append(
            "No NIR band identifiable (needs 4+ band optical imagery); reporting "
            "a green/red colour proxy as water_proxy, not NDWI."
        )

    stats = _index_change_stats(before_index, after_index, gsd_m, threshold,
                                 target_size=target_size, homography=homography, valid_mask=valid_mask)

    result: dict[str, Any] = {"is_true_index": is_true_index, "method": method}
    if stats["mean_before"] is None:
        warnings.append(
            "no overlapping valid area between the two aligned images; "
            "water change could not be computed"
        )
        result["pct_change"] = None
        result["changed_area_pct"] = None
        result["warnings"] = warnings
        return result

    if is_true_index:
        result["index_before"] = round(stats["mean_before"], 4)
        result["index_after"] = round(stats["mean_after"], 4)
    else:
        result["water_proxy_before"] = round(stats["mean_before"], 4)
        result["water_proxy_after"] = round(stats["mean_after"], 4)

    return _finalize(result, stats, warnings)


def align_images(before: np.ndarray, after: np.ndarray):
    """Register `before` onto `after`'s frame via SIFT features + homography.

    Falls back to the unaligned (resized) image if there aren't enough
    reliable feature matches, so a low-texture image pair degrades gracefully
    instead of throwing during a live demo.

    Returns `(aligned_before, valid_mask, aligned_ok, diagnostics)`, where
    `diagnostics = {"good_matches": int, "homography_found": bool, "homography":
    np.ndarray | None}` is always a real dict (never a guess) -- it's
    initialised before anything can fail, and `good_matches` is recorded even
    when it's the reason alignment falls back, so the caller can put real
    numbers in its execution trace either way. `homography` is the same 3x3
    matrix used to produce `aligned_before`, exposed so a different caller
    (e.g. warping a land-cover mask onto the same frame) reuses the exact
    transform instead of recomputing a second, potentially different one.
    """
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    height, width = gray_after.shape
    diagnostics: dict[str, Any] = {"good_matches": 0, "homography_found": False, "homography": None}

    try:
        sift = cv2.SIFT_create()
        kp_before, desc_before = sift.detectAndCompute(gray_before, None)
        kp_after, desc_after = sift.detectAndCompute(gray_after, None)

        if desc_before is None or desc_after is None:
            raise RuntimeError("Not enough visual features to align images")

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc_before, desc_after, k=2)

        good_matches = [m for m, n in matches if len(matches) and m.distance < 0.7 * n.distance]
        diagnostics["good_matches"] = len(good_matches)

        if len(good_matches) < 10:
            raise RuntimeError("Not enough reliable feature matches to align images")

        points_before = np.float32(
            [kp_before[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        points_after = np.float32(
            [kp_after[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(points_before, points_after, cv2.RANSAC, 5.0)

        if homography is None or mask is None:
            raise RuntimeError("Could not compute a reliable transformation")
        diagnostics["homography_found"] = True
        diagnostics["homography"] = homography

        aligned_before = cv2.warpPerspective(before, homography, (width, height))

        valid_mask = cv2.warpPerspective(
            np.ones((before.shape[0], before.shape[1]), dtype=np.uint8) * 255,
            homography,
            (width, height),
        )
        valid_mask = cv2.erode(valid_mask, np.ones((15, 15), np.uint8), iterations=1)

        return aligned_before, valid_mask, True, diagnostics

    except Exception:
        # Fall back to a plain resize-based comparison (previous behavior).
        return before, np.ones((height, width), dtype=np.uint8) * 255, False, diagnostics
