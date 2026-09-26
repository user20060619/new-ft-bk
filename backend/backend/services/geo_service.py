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

# rasterio/PIL band order for plain RGB data (no NIR).
_RGB_PROXY_ORDER: dict[str, int] = {"blue": 0, "green": 1, "red": 2}

_NDVI_CHANGE_THRESHOLD = 0.1
_NDWI_CHANGE_THRESHOLD = 0.1
_VEGETATION_PROXY_CHANGE_THRESHOLD = 15.0  # excess-green index is on a ~[-510,510] scale
_WATER_PROXY_CHANGE_THRESHOLD = 0.1        # water proxy reuses the NDWI formula's [-1,1] scale


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


def _identify_band_order(r: RasterInput, band_order: dict[str, int] | None
                          ) -> dict[str, int] | None:
    """A NIR band is never guessed from pixel values -- only from band count
    plus a stated order (default B,G,R,NIR).  Returns None when a real NIR
    band can't be identified for this input, so the caller falls back to an
    RGB colour proxy instead of a fabricated NDVI/NDWI."""
    if r.modality != "optical":
        return None
    order = band_order or DEFAULT_BAND_ORDER
    if "nir" not in order or r.bands < 4 or max(order.values()) >= r.bands:
        return None
    return order


def _excess_green_index(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """(2*Green - Red - Blue): a common RGB vegetation colour proxy.  Not NDVI."""
    blue = array[order["blue"]].astype(np.float32)
    green = array[order["green"]].astype(np.float32)
    red = array[order["red"]].astype(np.float32)
    return (2 * green) - red - blue


def _water_like_index(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """(Green-Red)/(Green+Red): reuses the NDWI formula's shape on colour
    bands instead of NIR, as a water-like proxy.  Not NDWI."""
    return calculate_ndwi(array[order["green"]], array[order["red"]])


def _index_change_stats(before_index: np.ndarray, after_index: np.ndarray,
                         gsd_m: float | None, threshold: float) -> dict[str, Any]:
    height = min(before_index.shape[0], after_index.shape[0])
    width = min(before_index.shape[1], after_index.shape[1])
    ib = cv2.resize(before_index.astype(np.float32), (width, height))
    ia = cv2.resize(after_index.astype(np.float32), (width, height))

    mean_before = float(np.mean(ib))
    mean_after = float(np.mean(ia))
    changed_mask = (np.abs(ia - ib) > threshold).astype(np.uint8) * 255
    changed_pct, area_km2 = calculate_area_percentage(changed_mask, gsd_m)

    stats: dict[str, Any] = {
        "mean_before": mean_before,
        "mean_after": mean_after,
        "pct_change": _pct_change(mean_before, mean_after),
        "changed_area_pct": round(changed_pct, 2),
    }
    if area_km2 is not None:
        stats["area_changed_km2"] = round(area_km2, 4)
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
                                band_order: dict[str, int] | None = None) -> dict[str, Any]:
    """True NDVI when both inputs have an identifiable NIR band (4+ band
    optical, default B,G,R,NIR order); otherwise an RGB excess-green colour
    proxy ("vegetation_proxy"), clearly labelled and warned about."""
    order_before = _identify_band_order(before, band_order)
    order_after = _identify_band_order(after, band_order)
    gsd_m = _shared_gsd(before, after)
    warnings: list[str] = []

    if order_before and order_after:
        before_index = calculate_ndvi(before.array[order_before["nir"]], before.array[order_before["red"]])
        after_index = calculate_ndvi(after.array[order_after["nir"]], after.array[order_after["red"]])
        stats = _index_change_stats(before_index, after_index, gsd_m, _NDVI_CHANGE_THRESHOLD)
        result: dict[str, Any] = {
            "is_true_index": True,
            "method": "NDVI = (NIR-Red)/(NIR+Red)",
            "index_before": round(stats["mean_before"], 4),
            "index_after": round(stats["mean_after"], 4),
        }
    else:
        before_index = _excess_green_index(before.array, _RGB_PROXY_ORDER)
        after_index = _excess_green_index(after.array, _RGB_PROXY_ORDER)
        stats = _index_change_stats(before_index, after_index, gsd_m, _VEGETATION_PROXY_CHANGE_THRESHOLD)
        warnings.append(
            "No NIR band identifiable (needs 4+ band optical imagery); reporting "
            "an RGB excess-green colour proxy as vegetation_proxy, not NDVI."
        )
        result = {
            "is_true_index": False,
            "method": "vegetation_proxy: excess-green colour index (2*Green-Red-Blue)",
            "vegetation_proxy_before": round(stats["mean_before"], 2),
            "vegetation_proxy_after": round(stats["mean_after"], 2),
        }

    return _finalize(result, stats, warnings)


def calculate_water_index(before: RasterInput, after: RasterInput,
                           band_order: dict[str, int] | None = None) -> dict[str, Any]:
    """True NDWI when both inputs have an identifiable NIR band; otherwise a
    green/red colour proxy ("water_proxy"), clearly labelled -- never called
    NDWI for RGB-only data."""
    order_before = _identify_band_order(before, band_order)
    order_after = _identify_band_order(after, band_order)
    gsd_m = _shared_gsd(before, after)
    warnings: list[str] = []

    if order_before and order_after:
        before_index = calculate_ndwi(before.array[order_before["green"]], before.array[order_before["nir"]])
        after_index = calculate_ndwi(after.array[order_after["green"]], after.array[order_after["nir"]])
        stats = _index_change_stats(before_index, after_index, gsd_m, _NDWI_CHANGE_THRESHOLD)
        result: dict[str, Any] = {
            "is_true_index": True,
            "method": "NDWI = (Green-NIR)/(Green+NIR)",
            "index_before": round(stats["mean_before"], 4),
            "index_after": round(stats["mean_after"], 4),
        }
    else:
        before_index = _water_like_index(before.array, _RGB_PROXY_ORDER)
        after_index = _water_like_index(after.array, _RGB_PROXY_ORDER)
        stats = _index_change_stats(before_index, after_index, gsd_m, _WATER_PROXY_CHANGE_THRESHOLD)
        warnings.append(
            "No NIR band identifiable (needs 4+ band optical imagery); reporting "
            "a green/red colour proxy as water_proxy, not NDWI."
        )
        result = {
            "is_true_index": False,
            "method": "water_proxy: green/red colour index (Green-Red)/(Green+Red)",
            "water_proxy_before": round(stats["mean_before"], 4),
            "water_proxy_after": round(stats["mean_after"], 4),
        }

    return _finalize(result, stats, warnings)


def align_images(before: np.ndarray, after: np.ndarray):
    """Register `before` onto `after`'s frame via SIFT features + homography.

    Falls back to the unaligned (resized) image if there aren't enough
    reliable feature matches, so a low-texture image pair degrades gracefully
    instead of throwing during a live demo.
    """
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    height, width = gray_after.shape

    try:
        sift = cv2.SIFT_create()
        kp_before, desc_before = sift.detectAndCompute(gray_before, None)
        kp_after, desc_after = sift.detectAndCompute(gray_after, None)

        if desc_before is None or desc_after is None:
            raise RuntimeError("Not enough visual features to align images")

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc_before, desc_after, k=2)

        good_matches = [m for m, n in matches if len(matches) and m.distance < 0.7 * n.distance]

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

        aligned_before = cv2.warpPerspective(before, homography, (width, height))

        valid_mask = cv2.warpPerspective(
            np.ones((before.shape[0], before.shape[1]), dtype=np.uint8) * 255,
            homography,
            (width, height),
        )
        valid_mask = cv2.erode(valid_mask, np.ones((15, 15), np.uint8), iterations=1)

        return aligned_before, valid_mask, True

    except Exception:
        # Fall back to a plain resize-based comparison (previous behavior).
        return before, np.ones((height, width), dtype=np.uint8) * 255, False
