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
from .services.bitemporal import (
    ClassChange,
    WarpedClassMasks,
    classify_pixels,
    date_classes_from_optical,
    date_classes_from_sar,
    draw_region_boxes,
    find_changed_regions,
    warp_and_compare_class,
)
from .services.geo_service import align_images, calculate_vegetation_index, calculate_water_index
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


_LOW_OVERLAP_WARNING_PCT = 50.0
_CLASS_NAMES = ("water", "vegetation", "built_up")
_NOT_COMPUTABLE_WARPED = WarpedClassMasks(None, None, ClassChange(None, None, None, None))
_BUILT_UP_QUERY_HINTS = ("built-up", "built up", "builtup", "urban", "buildings", "construction")


def _align_grayscale(before_gray: np.ndarray, after_gray: np.ndarray):
    """Wraps geo_service.align_images for a grayscale (or single-plane)
    before/after pair -- stacks each to 3 channels (a no-op for align_images's
    own internal BGR2GRAY conversion) so its SIFT/homography code runs
    unchanged regardless of source band count.  Returns exactly what
    align_images returns: (aligned_stack, valid_mask, aligned_ok, diag)."""
    before_stack = cv2.merge([before_gray, before_gray, before_gray])
    after_stack = cv2.merge([after_gray, after_gray, after_gray])
    return align_images(before_stack, after_stack)


def _evidence_modality(modality: str) -> str:
    return modality if modality in ("optical", "sar") else "none"


def _diff_and_stats(aligned_before_gray: np.ndarray, after_gray: np.ndarray,
                     valid_mask: np.ndarray, before: RasterInput, after: RasterInput,
                     job_dir: Path) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Pure (trace-agnostic) diff/threshold step, given already resampled +
    aligned grayscale planes.  Writes change_mask.png; the overlay is built
    later (in the `regions` step) so region boxes can be drawn onto it before
    it's saved once, rather than saving it twice."""
    diff = cv2.absdiff(aligned_before_gray, after_gray)
    diff[valid_mask == 0] = 0

    _, mask = cv2.threshold(diff, _CHANGE_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    total_pixels = int(mask.size)
    changed_pixels = int(np.sum(mask > 0))
    changed_percentage = (changed_pixels / total_pixels) * 100 if total_pixels else 0.0

    cv2.imwrite(str(job_dir / "change_mask.png"), mask)

    computed: dict[str, Any] = {
        "pct_changed": round(changed_percentage, 2),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
    }
    gsd_m = _shared_gsd(before, after)
    if gsd_m is not None:
        computed["area_changed_km2"] = round((changed_pixels * gsd_m * gsd_m) / 1_000_000, 4)

    warnings: list[str] = []
    if gsd_m is None:
        warnings.append(
            "ground sample distance is unknown or differs between the two "
            "inputs; changed area is reported as a percentage only, not km²"
        )
    return mask, computed, warnings


def _class_change_clause(name: str, c: ClassChange) -> str:
    label = name.replace("_", "-")
    if c.direction is None:
        return f"{label.capitalize()} change could not be computed for this pair."
    if c.direction == "unchanged":
        return (
            f"{label.capitalize()} area is essentially unchanged "
            f"({c.change_pct_points:+.1f} percentage points, within the "
            f"{SIGNIFICANCE_PCT:.1f}pp tolerance)."
        )
    return (
        f"{label.capitalize()} area {c.direction} by {abs(c.change_pct_points):.1f} "
        f"percentage points ({c.before_pct:.1f}% -> {c.after_pct:.1f}% of the compared overlap)."
    )


def _region_summary(regions: list[dict[str, Any]], total_region_count: int) -> str:
    if not regions:
        return "No individual change regions above the minimum size were detected."
    parts = []
    for r in regions[:2]:
        x, y, w, h = r["bbox"]
        area_str = f"{r['area_km2']:.4f} km²" if r["area_km2"] is not None else f"{r['area_px']} px"
        cls_str = f", {r['dominant_class_change']}" if r["dominant_class_change"] else ""
        parts.append(f"bbox ({x},{y},{w},{h}), {area_str}{cls_str}")
    count_str = (f"{total_region_count} change region(s)" if total_region_count == len(regions)
                 else f"{total_region_count} change region(s) (top {len(regions)} shown)")
    return f"{count_str} detected above the minimum size; largest: " + "; ".join(parts) + "."


def _mentions_built_up(query: str) -> bool:
    q = query.lower()
    return any(hint in q for hint in _BUILT_UP_QUERY_HINTS)


def _build_change_answer(query: str, computed: dict[str, Any]) -> str:
    if computed["pct_changed"] < SIGNIFICANCE_PCT:
        pixel_sentence = (
            f"No significant change was detected between the two images. "
            f"{computed['pct_changed']:.1f}% of pixels differ, which is within the noise "
            f"expected from co-registration and illumination differences."
        )
    else:
        pixel_sentence = (
            f"{computed['pct_changed']:.1f}% of the scene changed between the two images "
            f"({computed['changed_pixels']:,} of {computed['total_pixels']:,} pixels)"
            + (f", about {computed['area_changed_km2']:.2f} square kilometres."
               if "area_changed_km2" in computed else
               " (ground sample distance is unknown, so area could not be expressed in km²).")
        )

    changes = {name: ClassChange(computed[f"{name}_before_pct"], computed[f"{name}_after_pct"],
                                  computed[f"{name}_change_pct_points"], computed[f"{name}_direction"])
               for name in _CLASS_NAMES}
    region_sentence = _region_summary(computed["regions"], computed["total_region_count"])

    def _magnitude(name: str) -> float:
        pts = changes[name].change_pct_points
        return abs(pts) if pts is not None else -1.0

    if _mentions_built_up(query):
        lead = _class_change_clause("built_up", changes["built_up"])
        runner_up = max(("vegetation", "water"), key=_magnitude)
        return " ".join([lead, pixel_sentence, _class_change_clause(runner_up, changes[runner_up]), region_sentence])

    ranked = sorted(_CLASS_NAMES, key=_magnitude, reverse=True)
    class_sentences = [_class_change_clause(name, changes[name]) for name in ranked]
    return " ".join([pixel_sentence, *class_sentences, region_sentence])


def _change_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                     trace: ExecutionTrace, intent: str) -> HandlerResult:
    before, after = loaded[0], loaded[1]
    job_dir = Path(paths[0]).parent
    # Invariant: job_dir's name must equal the response's request_id -- this
    # function has no way to check that; it's enforced by run_analysis's
    # caller (main.py passes the same id as both the folder name and
    # request_id=...), not by anything here.
    job_id = job_dir.name
    warnings: list[str] = []
    same_modality = before.modality == after.modality and before.modality in ("optical", "sar")

    try:
        with trace.step("preprocess_resample", "cv2.resize",
                         "resize both inputs to a common grid (min width/height)", {}) as s:
            before_gray, after_gray, width, height = _resample_to_common_grid(before, after)
            s.params = {"target_width": width, "target_height": height}
            s.output_summary = f"resized both inputs to {width}x{height}"

        with trace.step("preprocess_align", "geo_service.align_images",
                         "OpenCV SIFT + BFMatcher + RANSAC homography", {}) as s:
            aligned_stack, valid_mask, aligned_ok, diag = _align_grayscale(before_gray, after_gray)
            valid_mask = valid_mask.astype(bool)  # align_images' valid_mask is 0/255 uint8, not 0/1
            aligned_before_gray = aligned_stack[:, :, 0]
            homography = diag.get("homography")
            s.params = {"good_matches": diag["good_matches"],
                        "homography_found": diag["homography_found"]}
            s.fallback = not aligned_ok
            s.output_summary = (
                f"aligned={aligned_ok}, {diag['good_matches']} good keypoint matches, "
                f"homography_found={diag['homography_found']}"
            )
        if not aligned_ok:
            warnings.append(
                "not enough reliable visual features to align the two images; "
                "comparison used a plain resize instead of SIFT alignment"
            )

        landcover_tool = {"optical": "landcover.extract_optical", "sar": "sar.extract_sar"}.get(
            before.modality, "n/a")
        before_classes = after_classes = None

        with trace.step("landcover_before", landcover_tool,
                         "per-date water/vegetation/built-up extraction", {}) as s:
            if same_modality:
                if before.modality == "optical":
                    lc = extract_optical(before, job_dir, job_id=job_id, prefix="before_")
                    before_classes = date_classes_from_optical(lc)
                else:
                    sr = extract_sar(before, job_dir, job_id=job_id, prefix="before_")
                    before_classes = date_classes_from_sar(sr)
                s.output_summary = (f"water={before_classes.water_pct}, "
                                     f"vegetation={before_classes.vegetation_pct}, "
                                     f"built_up={before_classes.built_up_pct}")
            else:
                s.output_summary = (f"skipped: modalities are '{before.modality}'+'{after.modality}', "
                                     "not both optical or both SAR")

        with trace.step("landcover_after", landcover_tool,
                         "per-date water/vegetation/built-up extraction", {}) as s:
            if same_modality:
                if after.modality == "optical":
                    lc = extract_optical(after, job_dir, job_id=job_id, prefix="after_")
                    after_classes = date_classes_from_optical(lc)
                else:
                    sr = extract_sar(after, job_dir, job_id=job_id, prefix="after_")
                    after_classes = date_classes_from_sar(sr)
                s.output_summary = (f"water={after_classes.water_pct}, "
                                     f"vegetation={after_classes.vegetation_pct}, "
                                     f"built_up={after_classes.built_up_pct}")
            else:
                s.output_summary = (f"skipped: modalities are '{before.modality}'+'{after.modality}', "
                                     "not both optical or both SAR")

        if not same_modality:
            warnings.append(
                "per-class land-cover change requires both dates to share a known, matching "
                f"modality (both optical or both SAR); this pair is '{before.modality}'+"
                f"'{after.modality}', so only the pixel-level change was computed."
            )

        with trace.step("analysis", "cv2.absdiff + cv2.threshold",
                         "OpenCV absolute difference threshold + morphology",
                         {"threshold": _CHANGE_DIFF_THRESHOLD}) as s:
            s.fallback = True  # classical method; the intended learned model (P2) isn't built yet
            mask, computed, diff_warnings = _diff_and_stats(
                aligned_before_gray, after_gray, valid_mask, before, after, job_dir,
            )
            warnings.extend(diff_warnings)
            s.output_summary = (
                f"{computed['changed_pixels']}/{computed['total_pixels']} px changed "
                f"({computed['pct_changed']:.1f}%)"
            )

        with trace.step("per_class_change", "bitemporal.warp_and_compare_class",
                         "percentage-point change per class, over the aligned overlap", {}) as s:
            total_valid = int(np.sum(valid_mask))
            overlap_pct = round(total_valid / valid_mask.size * 100, 2) if valid_mask.size else 0.0
            s.params = {"overlap_pct": overlap_pct}
            if overlap_pct < _LOW_OVERLAP_WARNING_PCT:
                warnings.append(
                    f"Only {overlap_pct:.1f}% of the frame overlaps between the two aligned "
                    "images; per-class change is computed over that overlap only."
                )

            warped: dict[str, WarpedClassMasks] = {}
            if same_modality:
                for cls in _CLASS_NAMES:
                    warped[cls] = warp_and_compare_class(
                        getattr(before_classes, f"{cls}_mask"), getattr(after_classes, f"{cls}_mask"),
                        homography, (width, height), valid_mask, SIGNIFICANCE_PCT,
                    )
                s.output_summary = "; ".join(f"{cls}={warped[cls].change.direction}" for cls in _CLASS_NAMES)
            else:
                warped = {cls: _NOT_COMPUTABLE_WARPED for cls in _CLASS_NAMES}
                s.output_summary = "skipped: no per-class land-cover available for this modality pairing"

        with trace.step("regions", "bitemporal.find_changed_regions",
                         "connected components of the pixel-diff mask, labelled by dominant class change",
                         {}) as s:
            before_labels = after_labels = None
            if same_modality and warped["built_up"].before_mask is not None:
                before_labels = classify_pixels(
                    warped["water"].before_mask, warped["vegetation"].before_mask, warped["built_up"].before_mask)
                after_labels = classify_pixels(
                    warped["water"].after_mask, warped["vegetation"].after_mask, warped["built_up"].after_mask)

            gsd_m = _shared_gsd(before, after)
            regions = find_changed_regions(mask, gsd_m=gsd_m, before_labels=before_labels, after_labels=after_labels)

            base_overlay = cv2.resize(_to_uint8_bgr_display(after.array), (width, height)).copy()
            base_overlay[mask > 0] = [0, 0, 255]
            overlay = draw_region_boxes(base_overlay, regions)
            cv2.imwrite(str(job_dir / "change_overlay.jpg"), overlay)

            s.params = {"region_count": len(regions)}
            s.output_summary = f"{len(regions)} region(s) of change found"

        before_img = cv2.resize(_to_uint8_bgr_display(before.array), (width, height))
        after_img = cv2.resize(_to_uint8_bgr_display(after.array), (width, height))
        cv2.imwrite(str(job_dir / "before.jpg"), before_img)
        cv2.imwrite(str(job_dir / "after.jpg"), after_img)
    except Exception as e:
        return HandlerResult(
            status="partial",
            answer="Change detection could not be completed for this pair.",
            warnings=["model_unavailable"],
            confidence_basis=f"classical change detection failed: {e}",
            fallback=True,
        )

    evidence: list[Evidence] = [
        Evidence(id="before", kind="image", label="Before", modality=_evidence_modality(before.modality),
                 url=f"/outputs/{job_id}/before.jpg"),
        Evidence(id="after", kind="image", label="After", modality=_evidence_modality(after.modality),
                 url=f"/outputs/{job_id}/after.jpg"),
        Evidence(id="change_mask", kind="mask", label="Change mask", modality="fused",
                 url=f"/outputs/{job_id}/change_mask.png"),
        Evidence(id="change_overlay", kind="overlay", label="Change overlay", modality="fused",
                 url=f"/outputs/{job_id}/change_overlay.jpg"),
    ]
    if same_modality:
        evidence.extend(before_classes.evidence)
        evidence.extend(after_classes.evidence)
        warnings.extend(before_classes.warnings)
        warnings.extend(after_classes.warnings)
    for cls in _CLASS_NAMES:
        wc = warped[cls]
        if wc.before_mask is None or wc.after_mask is None:
            continue
        diff_mask_cls = (wc.before_mask ^ wc.after_mask) & valid_mask
        filename = f"{cls}_change_mask.png"
        cv2.imwrite(str(job_dir / filename), diff_mask_cls.astype(np.uint8) * 255)
        evidence.append(Evidence(
            id=f"{cls}_change_mask", kind="mask", label=f"{cls.replace('_', ' ').title()} change mask",
            modality="fused", url=f"/outputs/{job_id}/{filename}",
        ))

    computed["overlap_pct"] = overlap_pct
    for cls in _CLASS_NAMES:
        c = warped[cls].change
        computed[f"{cls}_before_pct"] = c.before_pct
        computed[f"{cls}_after_pct"] = c.after_pct
        computed[f"{cls}_change_pct_points"] = c.change_pct_points
        computed[f"{cls}_direction"] = c.direction
    computed["total_region_count"] = len(regions)
    computed["regions"] = [
        {"bbox": list(r.bbox), "area_px": r.area_px, "area_km2": r.area_km2,
         "dominant_class_change": r.dominant_class_change}
        for r in regions[:10]
    ]

    answer = _build_change_answer(query, computed)

    return HandlerResult(
        status="success",
        answer=answer,
        computed=computed,
        evidence=evidence,
        warnings=warnings,
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


def _index_and_class_answer(intent_label: str, index_result: dict[str, Any], change: ClassChange) -> str:
    """Hand-written answer for the bitemporal vegetation/water handlers --
    never delegated to fusion.explain.explain("vegetation"/"water", ...): its
    templates hardcode "Mean NDVI"/"Mean NDWI" wording (wrong for a colour
    proxy -- the honesty violation CLAUDE.md rule 3 exists to prevent) and
    require area_changed_km2 unconditionally in the increase/decrease
    branches, raising inside explain() when gsd is unknown (the same failure
    mode _change_handler already routes around for its own template).  States
    both the continuous index trend and the discrete land-cover-area trend
    (`change`) rather than forcing them to agree."""
    if index_result.get("pct_change") is None:
        return f"{intent_label.capitalize()} change could not be computed: {index_result['method']}."

    is_true = index_result["is_true_index"]
    class_sentence = _class_change_clause(intent_label, change)

    if not is_true:
        # T10: a colour proxy's raw value is an uncalibrated linear combination
        # with no natural "percent change" meaning (unlike a real index) --
        # reporting its relative change as the headline risks reading as a
        # calibrated measurement (CLAUDE.md rule 3's underlying concern). Lead
        # with the land-cover AREA percentage-point change instead; the
        # proxy's own before/after means stay in `computed` only
        # (vegetation_proxy_before/after or water_proxy_before/after).
        ref = "NDVI" if intent_label == "vegetation" else "NDWI"
        return (f"{class_sentence} (based on a {intent_label}-like colour proxy, not {ref} -- "
                f"see the raw proxy values in the response data.)")

    # True index (NDVI/NDWI): a real physical quantity, so its relative
    # change is a meaningful, honest headline number.
    label = "NDVI" if intent_label == "vegetation" else "NDWI"
    pct = index_result["pct_change"]

    if pct > SIGNIFICANCE_PCT:
        verb = f"increased by {pct:.1f}%"
    elif pct < -SIGNIFICANCE_PCT:
        verb = f"decreased by {abs(pct):.1f}%"
    else:
        verb = f"is essentially unchanged ({pct:+.1f}%)"

    area_clause = (f", affecting approximately {index_result['area_changed_km2']:.2f} square kilometres"
                   if "area_changed_km2" in index_result else "")
    index_sentence = (
        f"{intent_label.capitalize()} {verb} (mean {label} from "
        f"{index_result['index_before']:.3f} to {index_result['index_after']:.3f}{area_clause})."
    )
    return f"{index_sentence} {class_sentence}"


def _vegetation_or_water_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                                  trace: ExecutionTrace, class_name: str) -> HandlerResult:
    """Shared body for `_vegetation_handler`/`_water_handler` (bitemporal
    only -- `_apply_overrides` guarantees these only ever fire for a
    bitemporal pair). Requires both dates `modality == "optical"`; any other
    combination (SAR+SAR, or anything involving "unknown") short-circuits to
    a clean not-computable result rather than building a parallel SAR-based
    path here -- `bitemporal.date_classes_from_sar`/`class_change` are
    already available if that's wanted later."""
    before, after = loaded[0], loaded[1]
    job_dir = Path(paths[0]).parent
    job_id = job_dir.name
    index_fn = calculate_vegetation_index if class_name == "vegetation" else calculate_water_index

    if before.modality != "optical" or after.modality != "optical":
        reason = (f"{class_name} change requires optical imagery on both dates; this pair is "
                  f"'{before.modality}'+'{after.modality}'")
        with trace.step("analysis", f"geo_service.calculate_{class_name}_index", "not computable", {}) as s:
            s.fallback = True
            s.output_summary = f"not computable: {reason}"
        return HandlerResult(
            status="success",
            answer=f"{class_name.capitalize()} change could not be computed: {reason}.",
            computed={"pct_change": None},
            warnings=[reason],
            confidence_basis=reason,
            fallback=True,
        )

    try:
        with trace.step("preprocess_resample", "cv2.resize",
                         "resize both inputs to a common grid (min width/height)", {}) as s:
            before_gray, after_gray, width, height = _resample_to_common_grid(before, after)
            s.params = {"target_width": width, "target_height": height}
            s.output_summary = f"resized both inputs to {width}x{height}"

        with trace.step("preprocess_align", "geo_service.align_images",
                         "OpenCV SIFT + BFMatcher + RANSAC homography", {}) as s:
            _, valid_mask, aligned_ok, diag = _align_grayscale(before_gray, after_gray)
            valid_mask = valid_mask.astype(bool)  # align_images' valid_mask is 0/255 uint8, not 0/1
            homography = diag.get("homography")
            s.params = {"good_matches": diag["good_matches"], "homography_found": diag["homography_found"]}
            s.fallback = not aligned_ok
            s.output_summary = f"aligned={aligned_ok}, {diag['good_matches']} good keypoint matches"

        with trace.step("landcover_before", "landcover.extract_optical",
                         "per-date water/vegetation/built-up extraction", {}) as s:
            before_classes = date_classes_from_optical(
                extract_optical(before, job_dir, job_id=job_id, prefix="before_"))
            s.output_summary = f"{class_name}={getattr(before_classes, f'{class_name}_pct')}"

        with trace.step("landcover_after", "landcover.extract_optical",
                         "per-date water/vegetation/built-up extraction", {}) as s:
            after_classes = date_classes_from_optical(
                extract_optical(after, job_dir, job_id=job_id, prefix="after_"))
            s.output_summary = f"{class_name}={getattr(after_classes, f'{class_name}_pct')}"

        with trace.step("analysis", f"geo_service.calculate_{class_name}_index",
                         "NDVI/NDWI (true) or colour proxy, before/after comparison over the "
                         "aligned overlap", {}) as s:
            index_result = index_fn(before, after, target_size=(width, height),
                                     homography=homography, valid_mask=valid_mask)
            s.fallback = not index_result["is_true_index"]
            s.params = {"is_true_index": index_result["is_true_index"], "method": index_result["method"]}
            s.output_summary = f"pct_change={index_result.get('pct_change')}"

        with trace.step("class_change", "bitemporal.warp_and_compare_class",
                         f"{class_name} area percentage-point change, over the aligned overlap", {}) as s:
            total_valid = int(np.sum(valid_mask))
            overlap_pct = round(total_valid / valid_mask.size * 100, 2) if valid_mask.size else 0.0
            s.params = {"overlap_pct": overlap_pct}
            if overlap_pct < _LOW_OVERLAP_WARNING_PCT:
                s.output_summary = f"low overlap ({overlap_pct:.1f}%); "
            warped = warp_and_compare_class(
                getattr(before_classes, f"{class_name}_mask"), getattr(after_classes, f"{class_name}_mask"),
                homography, (width, height), valid_mask, SIGNIFICANCE_PCT,
            )
            s.output_summary = (s.output_summary or "") + f"direction={warped.change.direction}"
    except Exception as e:
        return HandlerResult(
            status="partial",
            answer=f"{class_name.capitalize()} change analysis could not be completed for this pair.",
            warnings=["model_unavailable"],
            confidence_basis=f"{class_name} index/class analysis failed: {e}",
            fallback=True,
        )

    warnings = list(index_result.get("warnings", []))
    if overlap_pct < _LOW_OVERLAP_WARNING_PCT:
        warnings.append(
            f"Only {overlap_pct:.1f}% of the frame overlaps between the two aligned images; "
            f"{class_name} change is computed over that overlap only."
        )
    warnings.extend(before_classes.warnings)
    warnings.extend(after_classes.warnings)

    evidence = list(before_classes.evidence) + list(after_classes.evidence)
    if warped.before_mask is not None and warped.after_mask is not None:
        diff_mask_cls = (warped.before_mask ^ warped.after_mask) & valid_mask
        filename = f"{class_name}_change_mask.png"
        cv2.imwrite(str(job_dir / filename), diff_mask_cls.astype(np.uint8) * 255)
        evidence.append(Evidence(
            id=f"{class_name}_change_mask", kind="mask", label=f"{class_name.capitalize()} change mask",
            modality="fused", url=f"/outputs/{job_id}/{filename}",
        ))

    computed = {k: v for k, v in index_result.items() if k != "warnings"}
    computed["overlap_pct"] = overlap_pct
    computed[f"{class_name}_before_pct"] = warped.change.before_pct
    computed[f"{class_name}_after_pct"] = warped.change.after_pct
    computed[f"{class_name}_change_pct_points"] = warped.change.change_pct_points
    computed[f"{class_name}_direction"] = warped.change.direction

    answer = _index_and_class_answer(class_name, index_result, warped.change)

    return HandlerResult(
        status="success",
        answer=answer,
        computed=computed,
        evidence=evidence,
        warnings=warnings,
        confidence_basis=(
            f"classical {'NDVI/NDWI' if index_result['is_true_index'] else 'colour-proxy'} index + "
            "land-cover threshold comparison; no calibrated confidence score"
        ),
        fallback=not index_result["is_true_index"],
    )


def _vegetation_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                         trace: ExecutionTrace, intent: str) -> HandlerResult:
    return _vegetation_or_water_handler(loaded, paths, query, trace, "vegetation")


def _water_handler(loaded: list["RasterInput | None"], paths: list[str], query: str,
                    trace: ExecutionTrace, intent: str) -> HandlerResult:
    return _vegetation_or_water_handler(loaded, paths, query, trace, "water")


# locate intentionally absent: it falls through to _unavailable_handler until
# a rule-compliant implementation is wired in.
SERVICE_REGISTRY: dict[str, Handler] = {
    "optical_sar": _optical_sar_handler,
    "describe": _describe_handler,
    "vqa": _vqa_handler,
    "change": _change_handler,
    "vegetation": _vegetation_handler,
    "water": _water_handler,
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

    # T10: the router's own confidence is meaningless once its "unclear"
    # verdict has been entirely discarded in favour of a structural default
    # (optical_sar / bitemporal) -- show null rather than a misleading ~0.0,
    # and explain why in `basis`. Every other override (a *confident* router
    # intent redirected to a different handler, e.g. vegetation -> vqa for a
    # single image) keeps the router's real score, since it's still accurate.
    router_overridden_from_unclear = decision.intent == "unclear" and override_note is not None
    if router_overridden_from_unclear:
        router_confidence = None
        # override_note (from _apply_overrides) already reads "router abstained
        # (<reason>); defaulted to '<intent>' for ..." -- self-contained, no
        # need to re-prefix it.
        confidence_basis = f"{override_note}. {result.confidence_basis}".strip()
    else:
        router_confidence = decision.confidence
        confidence_basis = result.confidence_basis

    return AnalysisResponse(
        request_id=request_id,
        status=result.status,
        input_config=compat.input_config,
        intent=final_intent,
        answer=result.answer,
        computed=result.computed,
        evidence=result.evidence,
        confidence=Confidence(router=router_confidence, analysis=result.analysis_confidence,
                               basis=confidence_basis),
        execution=trace.steps,
        metadata=metadata,
        warnings=_dedupe_preserve_order(base_warnings + result.warnings),
        report_assets=[e.id for e in result.evidence],
    )
