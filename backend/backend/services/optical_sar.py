"""Optical + SAR fusion: agreement/union of water and built-up detections
from `landcover.py` (optical) and `sar.py` (SAR).

Owner: P4.  This is the module the CLAUDE.md "principal focus" (optical-SAR
paired analysis) is built on. Functions here are pure and trace-agnostic --
`orchestrator.py::_optical_sar_handler` owns all `ExecutionTrace` steps and
calls straight into these, mirroring `_change_handler`'s established shape.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

try:                                       # package import (backend.services.optical_sar)
    from ..contract import Evidence
    from ..rsio.raster import RasterInput
except ImportError:                        # bare import, e.g. a future pipeline.py-style caller
    from contract import Evidence
    from rsio.raster import RasterInput

from .landcover import LandcoverResult
from .sar import SarResult

_RESAMPLE_NOOP = "n/a (already matching grid)"
_RESAMPLE_REPROJECT = "reproject (CRS-aware, nearest)"
_RESAMPLE_RESIZE = "resize (size-only, nearest)"

_FALLBACK_RESAMPLE_WARNING = (
    "SAR and optical inputs don't both have CRS/geotransform metadata; "
    "resampling assumed identical extents rather than reprojecting -- "
    "alignment may be inexact if the two images don't actually cover the same area."
)
_WATER_SAR_ONLY_WARNING = (
    "Water fusion used SAR only; optical water percentage was not computable "
    "(panchromatic/single/dual-band optical input has no colour information)."
)


@dataclass
class OpticalSarResult:
    water_optical_only_pct: float | None
    water_sar_only_pct: float | None
    water_agreement_pct: float | None
    water_union_pct: float | None
    built_up_optical_only_pct: float
    built_up_sar_only_pct: float
    built_up_agreement_pct: float
    built_up_union_pct: float
    optical_result: LandcoverResult
    sar_result: SarResult
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _grids_match(sar: RasterInput, optical: RasterInput) -> bool:
    if (sar.width, sar.height) != (optical.width, optical.height):
        return False
    if sar.crs and optical.crs and sar.transform and optical.transform:
        return sar.crs == optical.crs and sar.transform == optical.transform
    return True  # no geo metadata to compare; matching pixel dims is all we can check


def resample_sar_to_optical(sar: RasterInput, optical: RasterInput) -> tuple[RasterInput, bool, str]:
    """Put SAR onto optical's exact grid.  Returns (resampled_sar, was_resampled, method).

    Prefers a real CRS-aware reprojection (rasterio.warp.reproject) whenever
    both inputs carry CRS+transform: matching pixel dimensions alone doesn't
    guarantee two rasters cover the same ground, so a plain size-based resize
    would silently misalign genuinely different-extent images. Falls back to
    size-only nearest-neighbour resize, with an explicit warning, only when
    either side lacks geo metadata to reproject against.

    Nearest-neighbour either way, matching rsio/raster.py's own established
    SAR convention: averaging dB/log-scale values doesn't equal the dB of the
    averaged power. Resampling happens on the raw, pre-to_db array -- safe
    because nearest-neighbour is pure index selection with no arithmetic
    across pixels, so it commutes with any pointwise transform (to_db
    included) applied afterward: the result is identical either order.
    """
    if _grids_match(sar, optical):
        return sar, False, _RESAMPLE_NOOP

    extra_warnings: list[str] = []

    if sar.crs and sar.transform and optical.crs and optical.transform:
        destination = np.full((sar.bands, optical.height, optical.width), np.nan, dtype=np.float32)
        reproject(
            source=sar.array,
            destination=destination,
            src_transform=Affine(*sar.transform),
            src_crs=CRS.from_string(sar.crs),
            dst_transform=Affine(*optical.transform),
            dst_crs=CRS.from_string(optical.crs),
            src_nodata=sar.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        resampled = destination
        method = _RESAMPLE_REPROJECT
    else:
        resampled = np.stack([
            cv2.resize(sar.array[b], (optical.width, optical.height), interpolation=cv2.INTER_NEAREST)
            for b in range(sar.bands)
        ])
        method = _RESAMPLE_RESIZE
        extra_warnings.append(_FALLBACK_RESAMPLE_WARNING)

    resampled_sar = dataclasses.replace(
        sar,
        array=resampled,
        width=optical.width,
        height=optical.height,
        gsd_m=optical.gsd_m,
        transform=optical.transform,
        crs=optical.crs,
        warnings=list(sar.warnings) + extra_warnings,
    )
    return resampled_sar, True, method


def _fuse_class(optical_mask: np.ndarray, sar_mask: np.ndarray, valid: np.ndarray, total_valid: int
                 ) -> tuple[float, float, float, float]:
    opt = optical_mask & valid
    s = sar_mask & valid
    agreement = opt & s
    union = opt | s
    optical_only = opt & ~s
    sar_only = s & ~opt

    def pct(mask: np.ndarray) -> float:
        return (int(np.sum(mask)) / total_valid * 100) if total_valid else 0.0

    return pct(optical_only), pct(sar_only), pct(agreement), pct(union)


def _legend_swatches() -> list[tuple[str, tuple[int, int, int]]]:
    # BGR colours (cv2 convention)
    return [
        ("Water: agreement", (255, 0, 0)),
        ("Water: optical only", (255, 255, 0)),
        ("Water: SAR only", (255, 0, 255)),
        ("Built-up: agreement", (0, 0, 255)),
        ("Built-up: optical only", (0, 165, 255)),
        ("Built-up: SAR only", (0, 255, 255)),
    ]


def _build_overlay(optical_result: LandcoverResult, sar_result: SarResult, valid: np.ndarray) -> np.ndarray:
    shape = sar_result.built_up_mask.shape
    overlay = np.full((*shape, 3), 40, dtype=np.uint8)

    legend = _legend_swatches()
    b_opt = optical_result.built_up_mask & valid
    b_sar = sar_result.built_up_mask & valid
    overlay[b_opt & ~b_sar] = legend[4][1]
    overlay[b_sar & ~b_opt] = legend[5][1]
    overlay[b_opt & b_sar] = legend[3][1]

    # optical water may be None (panchromatic input) -- paint SAR's water as
    # "SAR only" in that case; a visual simplification only, the numeric
    # fields carry the real optical-vs-SAR distinction, not this picture.
    w_sar = sar_result.water_mask & valid
    if optical_result.water_mask is not None:
        w_opt = optical_result.water_mask & valid
        overlay[w_opt & ~w_sar] = legend[1][1]
        overlay[w_sar & ~w_opt] = legend[2][1]
        overlay[w_opt & w_sar] = legend[0][1]
    else:
        overlay[w_sar] = legend[2][1]

    y = 10
    for label, color in legend:
        cv2.rectangle(overlay, (10, y), (30, y + 15), color, -1)
        cv2.putText(overlay, label, (35, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20

    return overlay


def fuse_masks(optical_result: LandcoverResult, sar_result: SarResult,
               output_dir: str | Path, job_id: str | None = None) -> OpticalSarResult:
    """Pixel-wise agreement/union of optical + SAR water and built-up masks.

    Denominator is sar_result.valid_mask (SAR's real excluded-nodata/non-
    finite/non-positive extent) -- optical has no per-pixel validity concept
    in this codebase yet, so SAR's is the meaningful constraint. Water
    degrades to "SAR only" when optical's water mask is None (panchromatic
    optical input); built-up is always computed from both, since brightness/
    edge-texture needs no colour information.
    """
    output_dir = Path(output_dir)
    job_id = job_id or output_dir.name

    valid = sar_result.valid_mask
    total_valid = int(np.sum(valid))
    warnings: list[str] = []

    excluded_pct = (1 - total_valid / valid.size) * 100 if valid.size else 0.0
    if excluded_pct > 0:
        warnings.append(
            f"{excluded_pct:.1f}% of pixels had no valid SAR data and were excluded "
            "from all optical/SAR fusion percentages."
        )

    if optical_result.water_mask is None:
        water_optical_only_pct = water_sar_only_pct = water_agreement_pct = None
        water_union_pct = sar_result.water_percentage
        warnings.append(_WATER_SAR_ONLY_WARNING)
    else:
        water_optical_only_pct, water_sar_only_pct, water_agreement_pct, water_union_pct = _fuse_class(
            optical_result.water_mask, sar_result.water_mask, valid, total_valid
        )

    built_up_optical_only_pct, built_up_sar_only_pct, built_up_agreement_pct, built_up_union_pct = _fuse_class(
        optical_result.built_up_mask, sar_result.built_up_mask, valid, total_valid
    )

    overlay = _build_overlay(optical_result, sar_result, valid)
    overlay_path = output_dir / "fused_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    evidence = list(optical_result.evidence) + list(sar_result.evidence) + [
        Evidence(id="fused_overlay", kind="overlay", label="Optical+SAR fused overlay", modality="fused",
                 url=f"/outputs/{job_id}/fused_overlay.png"),
    ]

    def r2(x: float | None) -> float | None:
        return round(x, 2) if x is not None else None

    return OpticalSarResult(
        water_optical_only_pct=r2(water_optical_only_pct),
        water_sar_only_pct=r2(water_sar_only_pct),
        water_agreement_pct=r2(water_agreement_pct),
        water_union_pct=r2(water_union_pct),
        built_up_optical_only_pct=round(built_up_optical_only_pct, 2),
        built_up_sar_only_pct=round(built_up_sar_only_pct, 2),
        built_up_agreement_pct=round(built_up_agreement_pct, 2),
        built_up_union_pct=round(built_up_union_pct, 2),
        optical_result=optical_result,
        sar_result=sar_result,
        evidence=evidence,
        warnings=warnings,
    )
