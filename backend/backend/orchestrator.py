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

from .contract import (
    AnalysisResponse,
    Confidence,
    Evidence,
    ExecutionTrace,
    InputMetadata,
    Metadata,
    build_failed_response,
)
from .fusion.explain import explain
from .router.intent import route
from .rsio.compat import check_inputs
from .rsio.raster import DEFAULT_MAX_PIXELS, RasterInput, load_raster


def _to_input_metadata(r: RasterInput) -> InputMetadata:
    return InputMetadata(
        filename=r.filename, modality=r.modality, bands=r.bands, dtype=r.dtype,
        width=r.width, height=r.height, crs=r.crs, gsd_m=r.gsd_m, date=r.date,
    )


def _apply_overrides(input_config: str | None, router_intent: str) -> tuple[str, str | None]:
    """The router doesn't know about SAR pairing or the vqa/single-image
    distinction, so this is where those get layered on top of its five
    intents (describe, vegetation, change, locate, water, unclear).

    `unclear` always wins first: if the router explicitly abstained, no
    override forces a different intent onto that abstention -- not even for
    an optical+SAR pair, which would otherwise look like the strongest case
    for a forced override.

    An optical+SAR pair is always routed to the optical_sar analysis
    regardless of what the query alone would have matched -- CLAUDE.md calls
    this the principal focus.

    A single image with a query the router placed in vegetation/water/change
    isn't really asking for a bitemporal-style trend (there's only one image),
    so it's routed to the general single-image 'vqa' intent instead; describe,
    locate and unclear are left as-is since they're already single-image-native.

    A `describe` request against a bitemporal (2-image) pair is left alone
    (describe is inherently single-image; it will caption the first file) but
    the substitution is recorded so it's visible in the trace, not silent.
    """
    if router_intent == "unclear":
        return "unclear", None

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


# vegetation/water/change/locate/optical_sar intentionally absent: they fall
# through to _unavailable_handler until a rule-compliant implementation (real
# GSD, real bands, no fabricated NDVI) is wired in.
SERVICE_REGISTRY: dict[str, Handler] = {
    "describe": _describe_handler,
    "vqa": _vqa_handler,
}


# --- entry point -----------------------------------------------------------

def run_analysis(paths: list[str], query: str, modalities: list[str] | None = None,
                  max_pixels: int = DEFAULT_MAX_PIXELS) -> AnalysisResponse:
    """One request in, one unified AnalysisResponse out.

    `paths` are already-saved files (1 or 2); this function knows nothing
    about FastAPI/UploadFile so it can be tested and reused directly.
    """
    request_id = str(uuid.uuid4())
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
        for path, hint in zip(paths, resolved_modalities):
            try:
                loaded.append(load_raster(path, modality=hint, max_pixels=max_pixels))
            except Exception:
                loaded.append(None)
        ok = sum(1 for r in loaded if r is not None)
        s.output_summary = f"{ok}/{len(paths)} input(s) loaded"

    input_warnings = [w for r in loaded if r is not None for w in r.warnings]
    metadata = Metadata(inputs=[_to_input_metadata(r) for r in loaded if r is not None])

    with trace.step("compatibility_check", "input.compat", "CRS/bounds/size checks", {}) as s:
        compat = check_inputs(loaded)
        s.output_summary = "; ".join(compat.steps) if compat.steps else "no steps recorded"

    if not compat.accepted:
        code = compat.errors[0].code if compat.errors else "incompatible_pair"
        return build_failed_response(
            code,
            request_id=request_id,
            input_config=compat.input_config,
            execution=trace,
            metadata=metadata,
            extra_warnings=extra_warnings + input_warnings + compat.warnings,
        )

    with trace.step("route", "router.intent.route", "deterministic rules + semantic tier2",
                     {"query": query, "n_images": len(paths)}) as s:
        decision = route(query, n_images=len(paths))
        s.output_summary = f"{decision.intent} ({decision.confidence}) via {decision.tool or '-'}; {decision.reason}"

    with trace.step("orchestration_override", "orchestrator.apply_overrides",
                     "input_config-aware intent rules",
                     {"input_config": compat.input_config, "router_intent": decision.intent}) as s:
        final_intent, override_note = _apply_overrides(compat.input_config, decision.intent)
        s.output_summary = override_note or f"no override; intent stays '{final_intent}'"

    base_warnings = extra_warnings + input_warnings + compat.warnings

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
        warnings=base_warnings + result.warnings,
        report_assets=[e.id for e in result.evidence],
    )
