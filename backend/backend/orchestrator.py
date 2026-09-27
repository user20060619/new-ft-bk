"""Agentic orchestration: the seam between the router, the input checks and
the analysis services, and the one place that builds a unified `AnalysisResponse`.

Owner: P4.  This is the CLAUDE.md flow:

    load_raster -> check_inputs -> route(query, n_images) -> orchestration
    override -> dispatch (service registry) -> explain() where it applies
    -> unified response

`router/intent.py` and `fusion/explain.py` are wrapped here, never edited --
the router only knows its own five intents (describe, vegetation, change,
locate, water) and nothing about SAR pairing, so this file is exactly the
"orchestration override" layer that adds the optical_sar/bitemporal awareness
CLAUDE.md's agentic-orchestration requirement asks for.

Every intent in `SERVICE_REGISTRY` maps to a real handler; anything not yet
built falls through to `_unavailable_handler`, which reports `status="partial"`
and a `model_unavailable` warning rather than inventing a number.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .contract import (
    AnalysisResponse,
    Confidence,
    Evidence,
    ExecutionTrace,
    InputMetadata,
    Metadata,
    build_failed_response,
)
from .fusion.explain import SIGNIFICANCE_PCT, explain
from .router.intent import route
from .rsio.compat import check_inputs
from .rsio.raster import DEFAULT_MAX_PIXELS, RasterInput, load_raster
from .services.geo_service import align_images
from .services.landcover import extract_optical
from .services.optical_sar import fuse_masks, resample_sar_to_optical
from .services.sar import extract_sar


def _to_input_metadata(r: RasterInput) -> InputMetadata:
    return InputMetadata(
        filename=r.filename, modality=r.modality, bands=r.bands, dtype=r.dtype,
        width=r.width, height=r.height, crs=r.crs, gsd_m=r.gsd_m, date=r.date,
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _apply_overrides(input_config: str | None, router_intent: str,
                      router_reason: str = "") -> tuple[str, str | None]:
    """The router doesn't know about SAR pairing or the vqa/single-image
    distinction, so this is where those get layered on top of its five
    intents (describe, vegetation, change, locate, water, unclear).

    `unclear` wins only when the router's own abstention was a genuine
    out-of-scope rejection -- `router_reason` starting with the literal
    "out of scope: " prefix router/intent.py itself uses for that case
    (pinned by a dedicated test in test_problem_statement_queries.py, since
    this file can't edit router/intent.py to guarantee that wording stays
    stable). Every *other* abstention reason (no rule matched, ambiguous
    between two intents, needs more images) just means the router's
    five-intent vocabulary doesn't cover this on-topic request -- for
    optical_sar and bitemporal pairs, the orchestrator's own structural
    knowledge (a real optical+SAR pair; a real bitemporal pair) resolves it
    instead of abstaining. A single image with a non-out-of-scope unclear
    stays unclear -- there's no equally safe structural default for it.

    An optical+SAR pair is always routed to the optical_sar analysis
    regardless of what the query alone would have matched (or failed to
    match) -- CLAUDE.md calls this the principal focus. A bitemporal pair
    with a non-out-of-scope unclear defaults to 'change', since a bitemporal
    request is fundamentally a change question even when it's phrased with
    vocabulary the router doesn't recognise (e.g. "built-up area increased").

    A single image with a query the router placed in vegetation/water/change
    isn't really asking for a bitemporal-style trend (there's only one image),
    so it's routed to the general single-image 'vqa' intent instead; describe,
    locate and unclear are left as-is since they're already single-image-native.

    A `describe` request against a bitemporal (2-image) pair is left alone
    (describe is inherently single-image; it will caption the first file) but
    the substitution is recorded so it's visible in the trace, not silent.
    """
    if router_intent == "unclear":
        if router_reason.startswith("out of scope:"):
            return "unclear", None
        if input_config == "optical_sar":
            return "optical_sar", (
                f"router abstained ({router_reason}); forced to 'optical_sar' "
                "because the input pair is optical+SAR"
            )
        if input_config == "bitemporal":
            return "change", (
                f"router abstained ({router_reason}); defaulted to 'change' for "
                "a bitemporal (2-image) pair, since bitemporal input is "
                "fundamentally a change question even when the router's "
                "vocabulary doesn't cover the exact phrasing"
            )
        return "unclear", None  # single (or unknown input_config): unchanged, stays unclear

    if input_config == "optical_sar":
        if router_intent != "optical_sar":
            return "optical_sar", (
                f"router said '{router_intent}'; forced to 'optical_sar' because "
                "the input pair is optical+SAR"
            )
        return "optical_sar", None

    if input_config == "single" and router_intent not in ("describe", "locate"):
        return "vqa", (
            f"router said '{router_intent}'; a single image with a "
            f"non-describe/locate query is answered as 'vqa' rather than a "
            f"bitemporal-style '{router_intent}' analysis"
        )

    if input_config == "bitemporal" and router_intent == "describe":
        return "describe", (
            "'describe' requested for a bitemporal (2-image) pair; keeping the "
            "router's choice and describing the first image, rather than "
            "forcing a change/vegetation/water intent"
        )

    return router_intent, None


# --- handlers ------------------------------------------------------------

@dataclass
class HandlerResult:
    status: str
    answer: str
    computed: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    analysis_confidence: float | None = None
    confidence_basis: str = ""
    fallback: bool = False


Handler = Callable[[list["RasterInput | None"], list[str], str, ExecutionTrace, str], HandlerResult]

# Lazily-loaded VLM handle: (model, processor, caption_fn), populated on first
# use only -- importing this module must not touch torch/CUDA, so the backend
# still starts and serves every other intent on a CPU-only machine.
_VLM: tuple = ()


def _load_vlm():
    global _VLM
    if not _VLM:
        vlm_dir = Path(__file__).resolve().parents[1] / "models" / "vlm"
        if str(vlm_dir) not in sys.path:
            sys.path.insert(0, str(vlm_dir))
        from load import caption as _cap  # P1's model file: wrapped, never edited
        from load import load_model
        model, processor, _ = load_model()
        _VLM = (model, processor, _cap)
    return _VLM


def _describe(image_path: str) -> tuple[str | None, str | None]:
    """Returns (caption, error).  Never raises: `load_model()` raises
    `SystemExit` when CUDA isn't available, which -- uncaught -- would kill
    the whole API process, not just this request."""
    try:
        model, processor, cap = _load_vlm()
        return cap(model, processor, image_path)[0], None
    except (SystemExit, Exception) as e:  # noqa: BLE001 -- see docstring
        return None, f"{type(e).__name__}: {e}"


def _unavailable_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                          trace: ExecutionTrace, intent: str) -> HandlerResult:
    with trace.step("analysis", f"service_registry.{intent}", "not implemented", {}) as s:
        s.fallback = True
        s.output_summary = "no analysis model is wired up for this intent yet"
    return HandlerResult(
        status="partial",
        answer="This analysis is not available yet in this build.",
        warnings=["model_unavailable"],
        confidence_basis="no analysis model is wired up for this intent yet",
        fallback=True,
    )


def _vqa_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                  trace: ExecutionTrace, intent: str) -> HandlerResult:
    """Single-image question answering (vegetation-like/water-like %, edge
    density, brightness, ...).  Not implemented yet -- T10."""
    with trace.step("analysis", "service_registry.vqa", "not implemented", {}) as s:
        s.fallback = True
        s.output_summary = "single-image VQA analysis not implemented yet (T10)"
    return HandlerResult(
        status="partial",
        answer="Single-image question answering is not available yet in this build.",
        warnings=["model_unavailable"],
        confidence_basis="no single-image VQA model is wired up yet (T10)",
        fallback=True,
    )


def _describe_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                       trace: ExecutionTrace, intent: str) -> HandlerResult:
    image_path = paths[0]
    with trace.step("analysis", "vlm.caption", "Qwen2.5-VL-3B-Instruct (4-bit)",
                     {"image": Path(image_path).name}) as s:
        caption_text, error = _describe(image_path)
        if error is not None:
            s.fallback = True
            s.output_summary = f"unavailable: {error}"
            return HandlerResult(
                status="partial",
                answer="A caption could not be generated for this image right now.",
                warnings=["model_unavailable"],
                confidence_basis=f"caption model unavailable: {error}",
                fallback=True,
            )
        s.output_summary = "caption generated"

    exp = explain("describe", caption=caption_text)
    return HandlerResult(
        status="success",
        answer=exp.answer,
        confidence_basis="VLM caption is generated text, not a computed measurement",
        fallback=False,
    )


_CHANGE_DIFF_THRESHOLD = 30


def _to_uint8_gray(array: np.ndarray) -> np.ndarray:
    """(bands,H,W) float32 -> a single normalised uint8 plane.  SIFT and the
    absolute-difference threshold both need 8-bit intensities; this is a
    display/algorithm-input rescale, not a claim about the data, so it applies
    regardless of the source dtype or band order."""
    gray = array[:3].mean(axis=0) if array.shape[0] >= 3 else array.mean(axis=0)
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        return np.zeros(gray.shape, dtype=np.uint8)
    return ((gray - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _to_uint8_bgr_display(array: np.ndarray) -> np.ndarray:
    """(bands,H,W) float32 -> (H,W,3) uint8 for the overlay image.  Bands are
    read in file order, which is R,G,B for both rasterio and PIL sources
    here; reversed to B,G,R since cv2.imwrite always writes that order."""
    bands = array.shape[0]
    chw = array[:3] if bands >= 3 else np.repeat(array[:1], 3, axis=0)
    hwc = np.transpose(chw, (1, 2, 0))
    lo, hi = float(hwc.min()), float(hwc.max())
    if hi - lo < 1e-6:
        return np.zeros(hwc.shape, dtype=np.uint8)
    scaled = ((hwc - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return scaled[:, :, ::-1]


def _shared_gsd(before: RasterInput, after: RasterInput) -> float | None:
    """Only report a ground sample distance when both inputs genuinely agree
    on one -- guessing which of two different resolutions applies would be
    exactly the kind of fabricated number CLAUDE.md rules out."""
    if before.gsd_m is None or after.gsd_m is None:
        return None
    if abs(before.gsd_m - after.gsd_m) > 1e-6:
        return None
    return before.gsd_m


def _resample_to_common_grid(before: RasterInput, after: RasterInput
                              ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pure (trace-agnostic) resize step: both inputs to their shared
    min(width), min(height), as grayscale uint8 planes."""
    height = min(before.height, after.height)
    width = min(before.width, after.width)
    before_gray = cv2.resize(_to_uint8_gray(before.array), (width, height))
    after_gray = cv2.resize(_to_uint8_gray(after.array), (width, height))
    return before_gray, after_gray, width, height


def _diff_and_evidence(aligned_before_gray: np.ndarray, after_gray: np.ndarray,
                        valid_mask: np.ndarray, before: RasterInput, after: RasterInput,
                        width: int, height: int, job_dir: Path, job_id: str
                        ) -> tuple[dict[str, Any], list[Evidence], list[str]]:
    """Pure (trace-agnostic) diff/threshold/evidence step, given already
    resampled + aligned grayscale planes."""
    diff = cv2.absdiff(aligned_before_gray, after_gray)
    diff[valid_mask == 0] = 0

    _, mask = cv2.threshold(diff, _CHANGE_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    total_pixels = int(mask.size)
    changed_pixels = int(np.sum(mask > 0))
    changed_percentage = (changed_pixels / total_pixels) * 100 if total_pixels else 0.0

    mask_path = job_dir / "change_mask.png"
    cv2.imwrite(str(mask_path), mask)

    overlay = cv2.resize(_to_uint8_bgr_display(after.array), (width, height)).copy()
    overlay[mask > 0] = [0, 0, 255]
    overlay_path = job_dir / "change_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    computed: dict[str, Any] = {
        "pct_changed": round(changed_percentage, 2),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
    }
    gsd_m = _shared_gsd(before, after)
    if gsd_m is not None:
        computed["area_changed_km2"] = round((changed_pixels * gsd_m * gsd_m) / 1_000_000, 4)

    evidence = [
        Evidence(id="change_mask", kind="mask", label="Change mask", modality="fused",
                 url=f"/outputs/{job_id}/change_mask.png"),
        Evidence(id="change_overlay", kind="overlay", label="Change overlay", modality="fused",
                 url=f"/outputs/{job_id}/change_overlay.jpg"),
    ]

    warnings: list[str] = []
    if gsd_m is None:
        warnings.append(
            "ground sample distance is unknown or differs between the two "
            "inputs; changed area is reported as a percentage only, not km²"
        )

    return computed, evidence, warnings


def _change_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                     trace: ExecutionTrace, intent: str) -> HandlerResult:
    before, after = loaded[0], loaded[1]
    job_dir = Path(paths[0]).parent
    # Invariant: job_dir's name must equal the response's request_id -- this
    # function has no way to check that; it's enforced by run_analysis's
    # caller (main.py passes the same id as both the folder name and
    # request_id=...), not by anything here.
    job_id = job_dir.name

    try:
        with trace.step("preprocess_resample", "cv2.resize",
                         "resize both inputs to a common grid (min width/height)", {}) as s:
            before_gray, after_gray, width, height = _resample_to_common_grid(before, after)
            s.params = {"target_width": width, "target_height": height}
            s.output_summary = f"resized both inputs to {width}x{height}"

        with trace.step("preprocess_align", "geo_service.align_images",
                         "OpenCV SIFT + BFMatcher + RANSAC homography", {}) as s:
            # align_images() only needs a 3-channel array to run its own
            # internal BGR2GRAY conversion; stacking the already-gray plane
            # three times makes that conversion a no-op while reusing the
            # existing SIFT/homography code unchanged, regardless of how many
            # bands the source actually had.
            before_stack = cv2.merge([before_gray, before_gray, before_gray])
            after_stack = cv2.merge([after_gray, after_gray, after_gray])
            aligned_stack, valid_mask, aligned_ok, diag = align_images(before_stack, after_stack)
            aligned_before_gray = aligned_stack[:, :, 0]
            s.params = {"good_matches": diag["good_matches"],
                        "homography_found": diag["homography_found"]}
            s.fallback = not aligned_ok
            s.output_summary = (
                f"aligned={aligned_ok}, {diag['good_matches']} good keypoint matches, "
                f"homography_found={diag['homography_found']}"
            )

        with trace.step("analysis", "cv2.absdiff + cv2.threshold",
                         "OpenCV absolute difference threshold + morphology",
                         {"threshold": _CHANGE_DIFF_THRESHOLD}) as s:
            s.fallback = True  # classical method; the intended learned model (P2) isn't built yet
            computed, evidence, method_warnings = _diff_and_evidence(
                aligned_before_gray, after_gray, valid_mask, before, after,
                width, height, job_dir, job_id,
            )
            s.output_summary = (
                f"{computed['changed_pixels']}/{computed['total_pixels']} px changed "
                f"({computed['pct_changed']:.1f}%)"
            )
    except Exception as e:
        return HandlerResult(
            status="partial",
            answer="Change detection could not be completed for this pair.",
            warnings=["model_unavailable"],
            confidence_basis=f"classical change detection failed: {e}",
            fallback=True,
        )

    if not aligned_ok:
        method_warnings.append(
            "not enough reliable visual features to align the two images; "
            "comparison used a plain resize instead of SIFT alignment"
        )

    if "area_changed_km2" in computed:
        exp = explain("change", computed=computed)
        answer = exp.answer
    else:
        # fusion/explain.py's CHANGE/CHANGE_NONE templates require
        # area_changed_km2; when gsd_m isn't known we report honestly in
        # percentage terms instead of feeding the template a fabricated area.
        pct = computed["pct_changed"]
        if pct < SIGNIFICANCE_PCT:
            answer = (
                f"No significant change was detected between the two images. "
                f"{pct:.1f}% of pixels differ, which is within the noise expected "
                f"from co-registration and illumination differences."
            )
        else:
            answer = (
                f"{pct:.1f}% of the scene changed between the two images "
                f"({computed['changed_pixels']:,} of {computed['total_pixels']:,} pixels). "
                f"Ground sample distance is unknown, so the changed area could not be "
                f"expressed in square kilometres."
            )

    return HandlerResult(
        status="success",
        answer=answer,
        computed=computed,
        evidence=evidence,
        warnings=method_warnings,
        confidence_basis="classical OpenCV alignment + threshold method; no calibrated confidence score",
        fallback=True,
    )


def _optical_sar_water_sentence(optical_result, sar_result, fusion_result) -> str:
    if optical_result.water_percentage is None:
        return (
            f"Water: optical could not compute a water estimate "
            f"({optical_result.water_method}); using SAR alone, "
            f"{sar_result.water_percentage}% ({sar_result.water_method})."
        )
    return (
        f"Water: optical detects {optical_result.water_percentage}% "
        f"({optical_result.water_method}); SAR detects {sar_result.water_percentage}% "
        f"({sar_result.water_method}). The two agree on {fusion_result.water_agreement_pct}% "
        f"and together cover {fusion_result.water_union_pct}% (optical alone: "
        f"{fusion_result.water_optical_only_pct}%, SAR alone: {fusion_result.water_sar_only_pct}%, "
        f"consistent with SAR's ability to see through cloud and haze)."
    )


def _optical_sar_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                          trace: ExecutionTrace, intent: str) -> HandlerResult:
    a, b = loaded[0], loaded[1]
    # Select by modality, not upload position: check_inputs only guarantees
    # the SET {optical, sar}, not which file came first.
    optical, sar = (a, b) if a.modality == "optical" else (b, a)
    job_dir = Path(paths[0]).parent  # both uploads always share one job directory
    job_id = job_dir.name

    try:
        with trace.step("preprocess_resample", "optical_sar.resample_sar_to_optical",
                         "reproject SAR onto the optical grid (CRS-aware when possible)", {}) as s:
            resampled_sar, was_resampled, resample_method = resample_sar_to_optical(sar, optical)
            s.params = {
                "sar_original_size": [sar.width, sar.height],
                "target_size": [optical.width, optical.height],
                "resampled": was_resampled,
                "method": resample_method,
            }
            s.output_summary = (
                f"resampled SAR {sar.width}x{sar.height} -> {optical.width}x{optical.height} "
                f"via {resample_method}" if was_resampled else "SAR already on the optical grid"
            )

        with trace.step("optical_landcover", "landcover.extract_optical",
                         "true-index/proxy land-cover extraction", {}) as s:
            optical_result = extract_optical(optical, job_dir, job_id=job_id, prefix="optical_")
            s.params = {
                "water_method": optical_result.water_method,
                "water_is_true_index": optical_result.water_is_true_index,
                "water_threshold": optical_result.water_threshold,
                "vegetation_method": optical_result.vegetation_method,
                "vegetation_is_true_index": optical_result.vegetation_is_true_index,
                "vegetation_threshold": optical_result.vegetation_threshold,
                "built_up_method": optical_result.built_up_method,
            }
            s.output_summary = (
                f"water={optical_result.water_percentage}%, "
                f"vegetation={optical_result.vegetation_percentage}%, "
                f"built_up={optical_result.built_up_percentage}%"
            )

        with trace.step("sar_processing", "sar.extract_sar",
                         "Otsu/percentile threshold SAR extraction", {}) as s:
            sar_result = extract_sar(resampled_sar, job_dir, job_id=job_id, prefix="sar_")
            s.params = {
                "water_threshold_db": sar_result.water_threshold_db,
                "water_threshold_source": sar_result.water_threshold_source,
                "built_up_threshold_db": sar_result.built_up_threshold_db,
                "built_up_threshold_source": sar_result.built_up_threshold_source,
            }
            s.output_summary = f"water={sar_result.water_percentage}%, built_up={sar_result.built_up_percentage}%"

        with trace.step("fusion", "optical_sar.fuse_masks",
                         "pixel-wise agreement/union of optical + SAR masks", {}) as s:
            s.fallback = True  # classical set-overlap fusion, not a learned model
            fusion_result = fuse_masks(optical_result, sar_result, job_dir, job_id=job_id)
            s.params = {
                "water_agreement_pct": fusion_result.water_agreement_pct,
                "water_union_pct": fusion_result.water_union_pct,
                "built_up_agreement_pct": fusion_result.built_up_agreement_pct,
                "built_up_union_pct": fusion_result.built_up_union_pct,
            }
            s.output_summary = "computed optical/SAR agreement, union and exclusive percentages"
    except Exception as e:
        return HandlerResult(
            status="partial",
            answer="Optical-SAR fusion could not be completed for this pair.",
            warnings=["model_unavailable"],
            confidence_basis=f"optical-SAR fusion failed: {e}",
            fallback=True,
        )

    answer = (
        _optical_sar_water_sentence(optical_result, sar_result, fusion_result) + "\n" +
        f"Built-up: optical detects {optical_result.built_up_percentage}% "
        f"({optical_result.built_up_method}); SAR detects {sar_result.built_up_percentage}% "
        f"({sar_result.built_up_method}). The two agree on {fusion_result.built_up_agreement_pct}% "
        f"and together cover {fusion_result.built_up_union_pct}% (optical alone: "
        f"{fusion_result.built_up_optical_only_pct}%, SAR alone: {fusion_result.built_up_sar_only_pct}%)."
    )

    computed = {
        "water_optical_pct": optical_result.water_percentage,
        "water_sar_pct": sar_result.water_percentage,
        "water_agreement_pct": fusion_result.water_agreement_pct,
        "water_union_pct": fusion_result.water_union_pct,
        "built_up_optical_pct": optical_result.built_up_percentage,
        "built_up_sar_pct": sar_result.built_up_percentage,
        "built_up_agreement_pct": fusion_result.built_up_agreement_pct,
        "built_up_union_pct": fusion_result.built_up_union_pct,
    }

    all_warnings = list(optical_result.warnings) + list(sar_result.warnings) + list(fusion_result.warnings)

    return HandlerResult(
        status="success",
        answer=answer,
        computed=computed,
        evidence=fusion_result.evidence,
        warnings=all_warnings,
        confidence_basis="classical threshold-based extraction + set-overlap fusion; no calibrated confidence score",
        fallback=True,
    )


# vegetation/water/locate intentionally absent: they fall through to
# _unavailable_handler until a rule-compliant implementation (real GSD, real
# bands, no fabricated NDVI) is wired in.
SERVICE_REGISTRY: dict[str, Handler] = {
    "optical_sar": _optical_sar_handler,
    "describe": _describe_handler,
    "vqa": _vqa_handler,
    "change": _change_handler,
}


# --- entry point -----------------------------------------------------------

def run_analysis(paths: list[str], query: str, modalities: list[str] | None = None,
                  max_pixels: int = DEFAULT_MAX_PIXELS,
                  original_filenames: list[str] | None = None,
                  request_id: str | None = None) -> AnalysisResponse:
    """One request in, one unified AnalysisResponse out.

    `paths` are already-saved files (1 or 2), typically named `input_N.ext` on
    disk; this function knows nothing about FastAPI/UploadFile so it can be
    tested and reused directly.  `original_filenames`, when given, restores
    the caller's own filename (e.g. "Mumbai25.jpg") into `metadata.inputs[]`
    and into per-file warnings, in place of the saved-path name.

    `request_id`, when given, should be the same id the caller used to name
    the job/output folder `paths` live in -- `_change_handler` derives its
    evidence URLs from that folder name, so passing the same id here is what
    makes `response.request_id` actually match `response.evidence[].url`.
    """
    request_id = request_id or str(uuid.uuid4())
    trace = ExecutionTrace()
    extra_warnings: list[str] = []

    resolved_modalities: list[str | None] = [None] * len(paths)
    if modalities:
        if len(modalities) == len(paths):
            resolved_modalities = [m or None for m in modalities]
        else:
            extra_warnings.append(
                f"modality hint count ({len(modalities)}) did not match file "
                f"count ({len(paths)}); ignoring hints"
            )

    with trace.step("load_raster", "input.load_raster", "rasterio metadata + PIL fallback",
                     {"n_files": len(paths), "max_pixels": max_pixels}) as s:
        loaded: list[RasterInput | None] = []
        for i, (path, hint) in enumerate(zip(paths, resolved_modalities)):
            try:
                r = load_raster(path, modality=hint, max_pixels=max_pixels)
                if original_filenames and i < len(original_filenames):
                    r.filename = original_filenames[i]
                loaded.append(r)
            except Exception:
                loaded.append(None)
        ok = sum(1 for r in loaded if r is not None)
        s.output_summary = f"{ok}/{len(paths)} input(s) loaded"

    # Prefixed with each input's own filename so a warning is traceable to the
    # file it came from, e.g. "Mumbai25.jpg: No georeferencing available...".
    input_warnings = [f"{r.filename}: {w}" for r in loaded if r is not None for w in r.warnings]
    metadata = Metadata(inputs=[_to_input_metadata(r) for r in loaded if r is not None])

    with trace.step("compatibility_check", "input.compat", "CRS/bounds/size checks", {}) as s:
        compat = check_inputs(loaded)
        s.output_summary = "; ".join(compat.steps) if compat.steps else "no steps recorded"

    base_warnings = _dedupe_preserve_order(extra_warnings + input_warnings + compat.warnings)

    if not compat.accepted:
        code = compat.errors[0].code if compat.errors else "incompatible_pair"
        return build_failed_response(
            code,
            request_id=request_id,
            input_config=compat.input_config,
            execution=trace,
            metadata=metadata,
            extra_warnings=base_warnings,
        )

    with trace.step("route", "router.intent.route", "deterministic rules + semantic tier2",
                     {"query": query, "n_images": len(paths)}) as s:
        decision = route(query, n_images=len(paths))
        s.output_summary = f"{decision.intent} ({decision.confidence}) via {decision.tool or '-'}; {decision.reason}"

    with trace.step("orchestration_override", "orchestrator.apply_overrides",
                     "input_config-aware intent rules",
                     {"input_config": compat.input_config, "router_intent": decision.intent}) as s:
        final_intent, override_note = _apply_overrides(compat.input_config, decision.intent, decision.reason)
        s.output_summary = override_note or f"no override; intent stays '{final_intent}'"

    if final_intent == "unclear":
        exp = explain("unclear", clarification=decision.clarification)
        return AnalysisResponse(
            request_id=request_id,
            status="success",
            input_config=compat.input_config,
            intent="unclear",
            answer=exp.answer,
            confidence=Confidence(router=decision.confidence, analysis=None,
                                   basis="router abstained; no analysis was run"),
            execution=trace.steps,
            metadata=metadata,
            warnings=base_warnings,
        )

    handler = SERVICE_REGISTRY.get(final_intent, _unavailable_handler)
    result = handler(loaded, paths, query, trace, final_intent)

    return AnalysisResponse(
        request_id=request_id,
        status=result.status,
        input_config=compat.input_config,
        intent=final_intent,
        answer=result.answer,
        computed=result.computed,
        evidence=result.evidence,
        confidence=Confidence(router=decision.confidence, analysis=result.analysis_confidence,
                               basis=result.confidence_basis),
        execution=trace.steps,
        metadata=metadata,
        warnings=_dedupe_preserve_order(base_warnings + result.warnings),
        report_assets=[e.id for e in result.evidence],
    )
