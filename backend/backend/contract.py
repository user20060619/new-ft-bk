"""Unified response contract (CLAUDE.md) as pydantic models, plus the
execution-trace helper that fills its `execution` list.

Owner: P4.  Every analysis endpoint should return an `AnalysisResponse`
(directly, or via `build_failed_response` for errors) so `/analyze`, `/vqa`
and `/query` stop returning three different shapes.

`model_config = ConfigDict(extra="forbid")` on every model here is
deliberate: this file exists to BE the contract, so a caller that sets a
field the contract doesn't define should fail loudly, not silently ship an
inconsistent response.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

InputConfig = Literal["single", "bitemporal", "optical_sar"]
# CLAUDE.md's example lists "vqa | describe | change | vegetation | water |
# optical_sar | unclear" but omits "locate", even though router/intent.py has
# a real LOCATE intent and evidence.kind already has "boxes" for it -- added
# here so a "where are the buildings" query doesn't fail response validation.
Intent = Literal["vqa", "describe", "change", "vegetation", "water", "locate", "optical_sar", "unclear"]
Status = Literal["success", "partial", "failed"]
EvidenceKind = Literal["image", "mask", "overlay", "boxes", "heatmap"]
EvidenceModality = Literal["optical", "sar", "fused", "none"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    label: str
    modality: EvidenceModality
    url: str


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # T10: null when the orchestrator has replaced a genuinely-unclear router
    # verdict with a structural default (e.g. bitemporal -> "change") -- the
    # router's own ~0.0 score would otherwise read as "0% confident in this
    # answer" when really its opinion was discarded entirely. Every other
    # path still sets a real float.
    router: float | None = Field(default=0.0, ge=0.0, le=1.0)
    analysis: float | None = Field(default=None, ge=0.0, le=1.0)
    basis: str
    # T15: additive structured fields alongside `basis` -- `basis` stays the
    # single prose string the downloaded report renders unchanged; these let
    # the live UI show the same facts as separate bullet points instead of
    # parsing that prose. All null except when run_analysis actually
    # overrode the router's intent (router_suggested_intent/score,
    # override_reason) or ran a handler (method_basis).
    router_suggested_intent: Intent | None = None
    router_suggested_score: float | None = Field(default=None, ge=0.0, le=1.0)
    override_reason: str | None = None
    method_basis: str | None = None


class ExecutionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    name: str
    tool: str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    ms: int = Field(ge=0)
    fallback: bool = False


class InputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    modality: str
    bands: int
    dtype: str
    width: int
    height: int
    crs: str | None = None
    gsd_m: float | None = None
    date: str | None = None
    # T12: [west, south, east, north] in EPSG:4326, from the real CRS +
    # geotransform (rsio/raster.py); null when either is missing -- never a
    # guessed location.
    bounds: list[float] | None = None


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[InputMetadata] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Status
    input_config: InputConfig | None = None
    intent: Intent
    answer: str
    # T14: optional structured facts (same values `answer` narrates as prose)
    # for the frontend/report to render as a bulleted list; empty for
    # handlers that don't build one, which fall back to `answer`.
    answer_points: list[str] = Field(default_factory=list)
    computed: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence
    execution: list[ExecutionStep] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)
    warnings: list[str] = Field(default_factory=list)
    report_assets: list[str] = Field(default_factory=list)


# --- execution trace ---------------------------------------------------------
# `with trace.step(...) as s:` times the block and records it as one
# ExecutionStep, whatever happens inside -- including an exception, so a step
# that blew up still shows up in the audit trail instead of vanishing.

class _StepHandle:
    def __init__(self, trace: "ExecutionTrace", name: str, tool: str, method: str,
                 params: dict[str, Any]):
        self._trace = trace
        self.name = name
        self.tool = tool
        self.method = method
        self.params = params
        self.output_summary = ""
        self.fallback = False
        self._start = 0.0

    def __enter__(self) -> "_StepHandle":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = int((time.perf_counter() - self._start) * 1000)
        summary = self.output_summary
        if exc_type is not None and not summary:
            summary = f"error: {exc_type.__name__}: {exc}"

        self._trace.steps.append(ExecutionStep(
            step=len(self._trace.steps) + 1,
            name=self.name,
            tool=self.tool,
            method=self.method,
            params=self.params,
            output_summary=summary,
            ms=elapsed_ms,
            fallback=self.fallback,
        ))
        return False  # never suppress the exception


class ExecutionTrace:
    """Accumulates ExecutionStep entries in call order."""

    def __init__(self) -> None:
        self.steps: list[ExecutionStep] = []

    def step(self, name: str, tool: str, method: str,
              params: dict[str, Any] | None = None) -> _StepHandle:
        return _StepHandle(self, name, tool, method, params or {})


# --- failure helper ----------------------------------------------------------

_KNOWN_ERROR_MESSAGES: dict[str, str] = {
    "incompatible_pair": "The provided inputs could not be compared (mismatched CRS or no overlapping extent).",
    "missing_band": "A required spectral band is missing from the input.",
    "unsupported_format": "The uploaded file format is not supported.",
    "model_unavailable": "The required model is not available right now.",
    "unreadable_file": "One of the uploaded files could not be read.",
    "no_inputs": "At least one input image is required.",
    "too_many_inputs": "Too many input images were supplied.",
}


def build_failed_response(
    code: str,
    message: str | None = None,
    *,
    request_id: str | None = None,
    input_config: InputConfig | None = None,
    intent: Intent = "unclear",
    execution: ExecutionTrace | list[ExecutionStep] | None = None,
    metadata: Metadata | None = None,
    extra_warnings: list[str] | None = None,
) -> AnalysisResponse:
    """Build a contract-shaped failure response for a machine-readable `code`.

    Per CLAUDE.md, errors reuse the success shape with `status="failed"` and
    the code placed in `warnings` -- there is no separate error channel.
    `extra_warnings` (e.g. per-input notices collected before the failure)
    are appended after `code`, which always stays first.
    """
    steps = execution.steps if isinstance(execution, ExecutionTrace) else (execution or [])

    return AnalysisResponse(
        request_id=request_id or str(uuid.uuid4()),
        status="failed",
        input_config=input_config,
        intent=intent,
        answer=message or _KNOWN_ERROR_MESSAGES.get(code, f"Request failed: {code}"),
        computed={},
        evidence=[],
        confidence=Confidence(router=0.0, analysis=None, basis=f"failed before analysis: {code}"),
        execution=steps,
        metadata=metadata or Metadata(),
        warnings=[code, *(extra_warnings or [])],
        report_assets=[],
    )
