"""Single-image optical land-cover extraction: water, vegetation, built-up.

Owner: P4.  `extract_optical()` reuses geo_service.py's true-index-vs-proxy
band-identification logic (same honesty rules: true NDVI/NDWI only with an
identifiable NIR band, otherwise a labelled colour proxy) but for ONE image,
not a before/after pair -- the percentages here are land-cover fractions of
a single scene, not a temporal change.

Built-up is inherently rule-based: there is no possible "true index"
alternative to a brightness + edge/texture heuristic, so its method label
always says so explicitly (CLAUDE.md rule 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:                                       # package import (backend.services.landcover)
    from ..contract import Evidence
    from ..rsio.raster import RasterInput
except ImportError:                        # bare import, e.g. a future pipeline.py-style caller
    from contract import Evidence
    from rsio.raster import RasterInput

from .geo_service import (
    RGB_PROXY_ORDER,
    calculate_ndvi,
    calculate_ndwi,
    excess_green_index,
    identify_band_order,
    normalize_rgb,
    water_like_index,
)

_VEGETATION_NDVI_THRESHOLD = 0.2    # standard remote-sensing "is vegetation" cut
_VEGETATION_PROXY_CLASSIFY_THRESHOLD = 15.0  # against normalize_rgb's ~0-255 scale; NOT
# the same constant as geo_service.py's _VEGETATION_PROXY_CHANGE_THRESHOLD, which is
# calibrated for a before/after delta, a different quantity from an absolute classification.
_WATER_NDWI_THRESHOLD = 0.0          # McFeeters (1996) convention: NDWI > 0 -> water
_WATER_PROXY_CLASSIFY_THRESHOLD = 0.0  # water_like_index is already a ratio, no normalisation needed

_BUILT_UP_BRIGHTNESS_THRESHOLD = 100   # out of 255, on the normalised grayscale plane
_BUILT_UP_EDGE_DENSITY_THRESHOLD = 0.15  # fraction of Canny edge pixels in a local window
_BUILT_UP_EDGE_WINDOW = 15              # box-filter window (px) the edge density is smoothed over


@dataclass
class LandcoverResult:
    water_percentage: float
    water_method: str
    water_is_true_index: bool
    vegetation_percentage: float
    vegetation_method: str
    vegetation_is_true_index: bool
    built_up_percentage: float
    built_up_method: str
    built_up_is_true_index: bool = False  # no true-index analog exists; kept for shape symmetry
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _grayscale(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """Mean of the three colour bands, normalised to a common ~0-255 scale
    via the same `normalize_rgb` used for the vegetation proxy -- one shared
    normalisation, not a second near-duplicate rescale."""
    blue, green, red = normalize_rgb(array, order)
    return ((blue + green + red) / 3.0).astype(np.uint8)


def _water_mask(r: RasterInput, band_order: dict[str, int] | None
                 ) -> tuple[np.ndarray, float, str, bool, list[str]]:
    order = identify_band_order(r, band_order)
    if order:
        index = calculate_ndwi(r.array[order["green"]], r.array[order["nir"]])
        mask = index > _WATER_NDWI_THRESHOLD
        return mask, float(np.mean(mask)) * 100, "NDWI = (Green-NIR)/(Green+NIR)", True, []

    index = water_like_index(r.array, RGB_PROXY_ORDER)
    mask = index > _WATER_PROXY_CLASSIFY_THRESHOLD
    warning = (
        "No NIR band identifiable (needs 4+ band optical imagery); reporting "
        "a green/red colour proxy as water_proxy, not NDWI."
    )
    return mask, float(np.mean(mask)) * 100, "water_proxy: green/red colour index (Green-Red)/(Green+Red)", False, [warning]


def _vegetation_mask(r: RasterInput, band_order: dict[str, int] | None
                      ) -> tuple[np.ndarray, float, str, bool, list[str]]:
    order = identify_band_order(r, band_order)
    if order:
        index = calculate_ndvi(r.array[order["nir"]], r.array[order["red"]])
        mask = index > _VEGETATION_NDVI_THRESHOLD
        return mask, float(np.mean(mask)) * 100, "NDVI = (NIR-Red)/(NIR+Red)", True, []

    index = excess_green_index(r.array, RGB_PROXY_ORDER)
    mask = index > _VEGETATION_PROXY_CLASSIFY_THRESHOLD
    warning = (
        "No NIR band identifiable (needs 4+ band optical imagery); reporting "
        "an RGB excess-green colour proxy as vegetation_proxy, not NDVI."
    )
    return (mask, float(np.mean(mask)) * 100,
            "vegetation_proxy: excess-green colour index (2*Green-Red-Blue)", False, [warning])


def _built_up_mask(r: RasterInput, water_mask: np.ndarray, vegetation_mask: np.ndarray,
                    band_order: dict[str, int] | None) -> tuple[np.ndarray, float, str]:
    order = band_order or RGB_PROXY_ORDER
    gray = _grayscale(r.array, order)

    brightness_mask = gray > _BUILT_UP_BRIGHTNESS_THRESHOLD

    edges = cv2.Canny(gray, 60, 160)
    edge_density = cv2.boxFilter(
        edges.astype(np.float32) / 255.0, ddepth=-1,
        ksize=(_BUILT_UP_EDGE_WINDOW, _BUILT_UP_EDGE_WINDOW),
    )
    texture_mask = edge_density > _BUILT_UP_EDGE_DENSITY_THRESHOLD

    candidate = ~water_mask & ~vegetation_mask
    mask = candidate & (brightness_mask | texture_mask)
    method = (
        "rule-based: brightness (>{}) or local edge/texture density (>{}) on "
        "non-water, non-vegetation pixels".format(
            _BUILT_UP_BRIGHTNESS_THRESHOLD, _BUILT_UP_EDGE_DENSITY_THRESHOLD
        )
    )
    return mask, float(np.mean(mask)) * 100, method


def extract_optical(r: RasterInput, output_dir: str | Path, job_id: str | None = None,
                     prefix: str = "", band_order: dict[str, int] | None = None) -> LandcoverResult:
    """Water/vegetation/built-up masks + percentages + method labels for one
    optical image.  `output_dir` must already exist; `job_id` defaults to its
    folder name (same derive-from-folder-name convention `_change_handler`
    uses in orchestrator.py, so evidence URLs line up with the response's
    request_id).  `prefix` namespaces the three PNG filenames -- unused by a
    single call, but required once a caller (T9) runs this twice into the
    same folder for two dates, so the second call doesn't overwrite the first.
    """
    output_dir = Path(output_dir)
    job_id = job_id or output_dir.name

    water_mask, water_pct, water_method, water_true, water_warnings = _water_mask(r, band_order)
    vegetation_mask, vegetation_pct, vegetation_method, vegetation_true, vegetation_warnings = (
        _vegetation_mask(r, band_order)
    )
    built_up_mask, built_up_pct, built_up_method = _built_up_mask(
        r, water_mask, vegetation_mask, band_order
    )

    files = {
        "water_mask": water_mask,
        "vegetation_mask": vegetation_mask,
        "built_up_mask": built_up_mask,
    }
    labels = {
        "water_mask": "Water mask",
        "vegetation_mask": "Vegetation mask",
        "built_up_mask": "Built-up mask",
    }
    evidence: list[Evidence] = []
    for name, mask in files.items():
        filename = f"{prefix}{name}.png"
        cv2.imwrite(str(output_dir / filename), (mask.astype(np.uint8) * 255))
        evidence.append(Evidence(
            id=f"{prefix}{name}", kind="mask", label=labels[name], modality="optical",
            url=f"/outputs/{job_id}/{filename}",
        ))

    return LandcoverResult(
        water_percentage=round(water_pct, 2),
        water_method=water_method,
        water_is_true_index=water_true,
        vegetation_percentage=round(vegetation_pct, 2),
        vegetation_method=vegetation_method,
        vegetation_is_true_index=vegetation_true,
        built_up_percentage=round(built_up_pct, 2),
        built_up_method=built_up_method,
        evidence=evidence,
        warnings=water_warnings + vegetation_warnings,
    )
