#!/usr/bin/env python3
"""Tests for the unified response contract.  Owner: P4.

    python -m pytest backend/test_contract.py
"""
import sys
import time
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (  # noqa: E402
    AnalysisResponse,
    Confidence,
    Evidence,
    ExecutionStep,
    ExecutionTrace,
    InputMetadata,
    Metadata,
    build_failed_response,
)


def _minimal_response(**overrides):
    fields = dict(
        status="success",
        input_config="single",
        intent="describe",
        answer="A residential area with regularly spaced buildings.",
        confidence=Confidence(router=0.8, analysis=None, basis="router score"),
    )
    fields.update(overrides)
    return AnalysisResponse(**fields)


def test_minimal_response_has_defaults():
    r = _minimal_response()

    assert uuid.UUID(r.request_id)  # a real uuid was generated
    assert r.computed == {}
    assert r.evidence == []
    assert r.execution == []
    assert r.metadata.inputs == []
    assert r.warnings == []
    assert r.report_assets == []


def test_full_response_round_trips_through_json():
    r = _minimal_response(
        status="success",
        input_config="optical_sar",
        intent="water",
        computed={"pct_change": -12.5},
        evidence=[Evidence(id="e1", kind="mask", label="SAR water mask",
                            modality="sar", url="/outputs/x/mask.png")],
        confidence=Confidence(router=0.9, analysis=0.7, basis="pixel overlap ratio"),
        execution=[ExecutionStep(step=1, name="compatibility_check", tool="input.compat",
                                  method="rasterio metadata", params={}, ms=12,
                                  output_summary="2 inputs accepted", fallback=False)],
        metadata=Metadata(inputs=[InputMetadata(filename="a.tif", modality="sar", bands=1,
                                                 dtype="float32", width=512, height=512,
                                                 crs="EPSG:32643", gsd_m=10.0, date=None)]),
        warnings=["resample_required"],
        report_assets=["e1"],
    )

    payload = r.model_dump()
    restored = AnalysisResponse(**payload)

    assert restored == r
    assert set(payload) == {
        "request_id", "status", "input_config", "intent", "answer", "answer_points",
        "computed", "evidence", "confidence", "execution", "metadata", "warnings", "report_assets",
    }


def test_rejects_unknown_status():
    with pytest.raises(ValidationError):
        _minimal_response(status="ok")


def test_rejects_unknown_evidence_kind():
    with pytest.raises(ValidationError):
        Evidence(id="e1", kind="video", label="x", modality="optical", url="/x.png")


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        _minimal_response(unexpected_field=True)


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        Confidence(router=1.5, basis="bad")
    with pytest.raises(ValidationError):
        Confidence(router=0.5, analysis=-0.1, basis="bad")


def test_confidence_structured_override_fields_default_null():
    c = Confidence(router=0.8, basis="router score")
    assert c.router_suggested_intent is None
    assert c.router_suggested_score is None
    assert c.override_reason is None
    assert c.method_basis is None


def test_confidence_structured_override_fields_round_trip():
    c = Confidence(
        router=None, basis="router suggested 'unclear' (0.1), overridden to 'change' because x",
        router_suggested_intent="unclear", router_suggested_score=0.1,
        override_reason="router abstained; defaulted to 'change'",
        method_basis="classical OpenCV alignment + threshold method; no calibrated confidence score",
    )
    restored = Confidence(**c.model_dump())
    assert restored == c


def test_input_config_allows_none_for_rejected_requests():
    r = _minimal_response(input_config=None, status="failed", intent="unclear")
    assert r.input_config is None


def test_intent_allows_locate():
    """router/intent.py has a real LOCATE intent and evidence.kind has 'boxes'
    for it; CLAUDE.md's example schema omitted it, which would otherwise make
    a 'where are the buildings' response fail contract validation."""
    r = _minimal_response(intent="locate")
    assert r.intent == "locate"


# --- ExecutionTrace ----------------------------------------------------------

def test_trace_step_records_timing_and_fields():
    trace = ExecutionTrace()

    with trace.step("compatibility_check", "input.compat", "rasterio metadata",
                     {"n": 2}) as s:
        time.sleep(0.001)
        s.output_summary = "2 inputs accepted: optical + SAR"

    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.step == 1
    assert step.name == "compatibility_check"
    assert step.tool == "input.compat"
    assert step.method == "rasterio metadata"
    assert step.params == {"n": 2}
    assert step.output_summary == "2 inputs accepted: optical + SAR"
    assert step.fallback is False
    assert step.ms >= 1


def test_trace_step_defaults_params_and_fallback():
    trace = ExecutionTrace()

    with trace.step("describe", "vlm.caption", "Qwen2.5-VL-3B") as s:
        s.output_summary = "caption generated"
        s.fallback = True

    step = trace.steps[0]
    assert step.params == {}
    assert step.fallback is True


def test_trace_steps_increment_across_calls():
    trace = ExecutionTrace()

    with trace.step("a", "t1", "m1"):
        pass
    with trace.step("b", "t2", "m2"):
        pass
    with trace.step("c", "t3", "m3"):
        pass

    assert [s.step for s in trace.steps] == [1, 2, 3]
    assert [s.name for s in trace.steps] == ["a", "b", "c"]


def test_trace_records_step_and_reraises_on_exception():
    trace = ExecutionTrace()

    with pytest.raises(ValueError, match="boom"):
        with trace.step("ndvi", "geo_service.ndvi", "band math") as s:
            raise ValueError("boom")

    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert "boom" in step.output_summary
    assert "ValueError" in step.output_summary
    assert step.fallback is False


def test_trace_keeps_manual_summary_when_exception_occurs():
    trace = ExecutionTrace()

    with pytest.raises(RuntimeError):
        with trace.step("ndvi", "geo_service.ndvi", "band math") as s:
            s.output_summary = "partial result before crash"
            raise RuntimeError("nope")

    assert trace.steps[0].output_summary == "partial result before crash"


# --- build_failed_response ----------------------------------------------------

def test_build_failed_response_known_code():
    r = build_failed_response("incompatible_pair")

    assert r.status == "failed"
    assert r.warnings == ["incompatible_pair"]
    assert r.answer  # a real default message, not empty
    assert r.confidence.analysis is None
    assert uuid.UUID(r.request_id)
    assert r.evidence == []
    assert r.computed == {}


def test_build_failed_response_unknown_code_gets_generic_message():
    r = build_failed_response("some_new_code_nobody_documented")

    assert r.warnings == ["some_new_code_nobody_documented"]
    assert "some_new_code_nobody_documented" in r.answer


def test_build_failed_response_custom_message_overrides_default():
    r = build_failed_response("missing_band", "The NIR band is required for NDVI.")
    assert r.answer == "The NIR band is required for NDVI."


def test_build_failed_response_carries_execution_trace():
    trace = ExecutionTrace()
    with trace.step("compatibility_check", "input.compat", "rasterio metadata") as s:
        s.output_summary = "rejected: CRS mismatch"

    r = build_failed_response("incompatible_pair", execution=trace, input_config=None)

    assert len(r.execution) == 1
    assert r.execution[0].name == "compatibility_check"


def test_build_failed_response_extra_warnings_appended_after_code():
    r = build_failed_response("incompatible_pair", extra_warnings=["resample_required", "no CRS"])
    assert r.warnings == ["incompatible_pair", "resample_required", "no CRS"]


def test_build_failed_response_accepts_explicit_request_id_and_metadata():
    fixed_id = str(uuid.uuid4())
    meta = Metadata(inputs=[InputMetadata(filename="a.jpg", modality="optical", bands=3,
                                           dtype="uint8", width=10, height=10)])

    r = build_failed_response("unsupported_format", request_id=fixed_id, metadata=meta)

    assert r.request_id == fixed_id
    assert r.metadata.inputs[0].filename == "a.jpg"
