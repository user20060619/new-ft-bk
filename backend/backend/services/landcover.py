"""Single-image optical land-cover extraction: water, vegetation, built-up.

Owner: P4.  `extract_optical()` reuses geo_service.py's true-index-vs-proxy
band-identification logic (same honesty rules: true NDVI/NDWI only with an
identifiable NIR band, otherwise a labelled colour proxy) but for ONE image,
not a before/after pair -- the percentages here are land-cover fractions of
a single scene, not a temporal change.

Built-up is inherently rule-based: there is no possible "true index"
alternative to a brightness + edge/texture heuristic, so its method label
always says so explicitly (CLAUDE.md rule 2).

`LandcoverResult` carries raw `np.ndarray` mask fields (for T8's optical/SAR
fusion) -- note for any future caller: a plain `dataclasses.asdict()` of this
type will include those raw arrays, not just the summary fields.
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

_PANCHROMATIC_WATER_WARNING = (
    "No colour bands available (single/dual-band optical imagery); water "
    "cannot be computed as a colour proxy or NDWI. Provide a 3+ band optical "
    "image, or rely on the SAR water estimate alone."
)
_PANCHROMATIC_VEGETATION_WARNING = (
    "No colour bands available (single/dual-band optical imagery); vegetation "
    "cannot be computed as a colour proxy or NDVI. Provide a 3+ band optical image."
)


@dataclass
class LandcoverResult:
    water_percentage: float | None       # None (never 0.0) when not computable -- see panchromatic case
    water_method: str
    water_is_true_index: bool
    water_threshold: float | None
    vegetation_percentage: float | None  # None (never 0.0) when not computable
    vegetation_method: str
    vegetation_is_true_index: bool
    vegetation_threshold: float | None
    built_up_percentage: float
    built_up_method: str
    built_up_is_true_index: bool = False  # no true-index analog exists; kept for shape symmetry
    water_mask: np.ndarray | None = None
    vegetation_mask: np.ndarray | None = None
    built_up_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None  # all-True: this module has no per-pixel optical validity concept yet
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _grayscale(array: np.ndarray, order: dict[str, int]) -> np.ndarray:
    """Mean of the three colour bands, normalised to a common ~0-255 scale
    via the same `normalize_rgb` used for the vegetation proxy -- one shared
    normalisation, not a second near-duplicate rescale."""
    blue, green, red = normalize_rgb(array, order)
    return ((blue + green + red) / 3.0).astype(np.uint8)


def _panchromatic_grayscale(array: np.ndarray) -> np.ndarray:
    """Plain min-max stretch of the single band to 0-255 -- used when there
    aren't enough colour bands for normalize_rgb (which needs three)."""
    band = array[0].astype(np.float32)
    lo, hi = float(band.min()), float(band.max())
    if hi - lo < 1e-6:
        return np.zeros(band.shape, dtype=np.uint8)
    return ((band - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _water_mask(r: RasterInput, band_order: dict[str, int] | None
                 ) -> tuple[np.ndarray | None, float | None, str, bool, float | None, list[str]]:
    if r.bands < 3:
        return None, None, "not computable: single/dual-band (panchromatic) optical has no " \
            "colour information for a water proxy or NDWI", False, None, [_PANCHROMATIC_WATER_WARNING]

    order = identify_band_order(r, band_order)
    if order:
        index = calculate_ndwi(r.array[order["green"]], r.array[order["nir"]])
        mask = index > _WATER_NDWI_THRESHOLD
        return mask, float(np.mean(mask)) * 100, "NDWI = (Green-NIR)/(Green+NIR)", True, _WATER_NDWI_THRESHOLD, []

    index = water_like_index(r.array, RGB_PROXY_ORDER)
    mask = index > _WATER_PROXY_CLASSIFY_THRESHOLD
    warning = (
        "No NIR band identifiable (needs 4+ band optical imagery); reporting "
        "a green/red colour proxy as water_proxy, not NDWI."
    )
    return (mask, float(np.mean(mask)) * 100,
            "water_proxy: green/red colour index (Green-Red)/(Green+Red)", False,
            _WATER_PROXY_CLASSIFY_THRESHOLD, [warning])


def _vegetation_mask(r: RasterInput, band_order: dict[str, int] | None
                      ) -> tuple[np.ndarray | None, float | None, str, bool, float | None, list[str]]:
    if r.bands < 3:
        return None, None, "not computable: single/dual-band (panchromatic) optical has no " \
            "colour information for a vegetation proxy or NDVI", False, None, [_PANCHROMATIC_VEGETATION_WARNING]

    order = identify_band_order(r, band_order)
    if order:
        index = calculate_ndvi(r.array[order["nir"]], r.array[order["red"]])
        mask = index > _VEGETATION_NDVI_THRESHOLD
        return mask, float(np.mean(mask)) * 100, "NDVI = (NIR-Red)/(NIR+Red)", True, _VEGETATION_NDVI_THRESHOLD, []

    index = excess_green_index(r.array, RGB_PROXY_ORDER)
    mask = index > _VEGETATION_PROXY_CLASSIFY_THRESHOLD
    warning = (
        "No NIR band identifiable (needs 4+ band optical imagery); reporting "
        "an RGB excess-green colour proxy as vegetation_proxy, not NDVI."
    )
    return (mask, float(np.mean(mask)) * 100,
            "vegetation_proxy: excess-green colour index (2*Green-Red-Blue)", False,
            _VEGETATION_PROXY_CLASSIFY_THRESHOLD, [warning])


def _built_up_mask(r: RasterInput, water_mask: np.ndarray | None, vegetation_mask: np.ndarray | None,
                    band_order: dict[str, int] | None) -> tuple[np.ndarray, float, str]:
    if r.bands < 3:
        gray = _panchromatic_grayscale(r.array)
    else:
        order = band_order or RGB_PROXY_ORDER
        gray = _grayscale(r.array, order)

    brightness_mask = gray > _BUILT_UP_BRIGHTNESS_THRESHOLD

    edges = cv2.Canny(gray, 60, 160)
    edge_density = cv2.boxFilter(
        edges.astype(np.float32) / 255.0, ddepth=-1,
        ksize=(_BUILT_UP_EDGE_WINDOW, _BUILT_UP_EDGE_WINDOW),
    )
    texture_mask = edge_density > _BUILT_UP_EDGE_DENSITY_THRESHOLD

    # water_mask/vegetation_mask may be None (not computable, e.g. panchromatic
    # input) -- nothing to exclude in that case, not an error.
    water_exclusion = water_mask if water_mask is not None else np.zeros(gray.shape, dtype=bool)
    vegetation_exclusion = vegetation_mask if vegetation_mask is not None else np.zeros(gray.shape, dtype=bool)

    candidate = ~water_exclusion & ~vegetation_exclusion
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
    request_id).  `prefix` namespaces the PNG filenames -- unused by a single
    call, but required once a caller (T9) runs this twice into the same
    folder for two dates, so the second call doesn't overwrite the first.

    Water/vegetation are `None` (never `0.0`) for single/dual-band
    (panchromatic) optical imagery, which has no colour information to
    compute either from -- built-up (brightness/edge-texture) needs no
    colour information and stays fully computable regardless of band count.
    """
    output_dir = Path(output_dir)
    job_id = job_id or output_dir.name

    water_mask, water_pct, water_method, water_true, water_threshold, water_warnings = _water_mask(r, band_order)
    vegetation_mask, vegetation_pct, vegetation_method, vegetation_true, vegetation_threshold, vegetation_warnings = (
        _vegetation_mask(r, band_order)
    )
    built_up_mask, built_up_pct, built_up_method = _built_up_mask(
        r, water_mask, vegetation_mask, band_order
    )

    files: dict[str, np.ndarray] = {"built_up_mask": built_up_mask}
    if water_mask is not None:
        files["water_mask"] = water_mask
    if vegetation_mask is not None:
        files["vegetation_mask"] = vegetation_mask

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
        water_percentage=round(water_pct, 2) if water_pct is not None else None,
        water_method=water_method,
        water_is_true_index=water_true,
        water_threshold=water_threshold,
        vegetation_percentage=round(vegetation_pct, 2) if vegetation_pct is not None else None,
        vegetation_method=vegetation_method,
        vegetation_is_true_index=vegetation_true,
        vegetation_threshold=vegetation_threshold,
        built_up_percentage=round(built_up_pct, 2),
        built_up_method=built_up_method,
        water_mask=water_mask,
        vegetation_mask=vegetation_mask,
        built_up_mask=built_up_mask,
        valid_mask=np.ones((r.height, r.width), dtype=bool),
        evidence=evidence,
        warnings=water_warnings + vegetation_warnings,
    )
