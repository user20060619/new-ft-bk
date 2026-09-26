#!/usr/bin/env python3
"""Tests for the orchestrator's agentic flow.  Owner: P4.

    python -m pytest backend/test_orchestrator.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.orchestrator import run_analysis  # noqa: E402

UTM43 = CRS.from_epsg(32643)


def _write_geotiff(path, array, crs=None, transform=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)


def _optical(path, width=40, height=40, crs=None, transform=None):
    array = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)
    _write_geotiff(path, array, crs=crs, transform=transform)
    return str(path)


def _sar(path, width=40, height=40, crs=None, transform=None):
    array = (np.random.rand(1, height, width) * -30).astype(np.float32)
    _write_geotiff(path, array, crs=crs, transform=transform)
    return str(path)


def _step_names(response):
    return [s.name for s in response.execution]


def test_single_image_describe_query(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "What is visible in this image?")

    assert r.input_config == "single"
    assert r.intent == "describe"
    # No GPU on this machine -> the real VLM cannot load; must degrade
    # honestly rather than invent a caption.
    assert r.status == "partial"
    assert "model_unavailable" in r.warnings
    assert _step_names(r) == ["load_raster", "compatibility_check", "route",
                              "orchestration_override", "analysis"]


def test_unclear_query_short_circuits_before_dispatch(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "what will this look like in 2030")

    assert r.intent == "unclear"
    assert r.status == "success"
    assert r.answer  # a real clarification message, not empty
    assert r.confidence.analysis is None
    # No handler runs for an abstention.
    assert _step_names(r) == ["load_raster", "compatibility_check", "route",
                              "orchestration_override"]


def test_two_optical_change_query_is_not_yet_built(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "What changed between these two images?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    assert r.status == "partial"
    assert "model_unavailable" in r.warnings
    assert r.computed == {}  # never fabricate a number for an unbuilt service


def test_optical_sar_pair_forces_optical_sar_intent_regardless_of_query(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    sar = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    r = run_analysis([opt, sar], "has vegetation decreased here")

    assert r.input_config == "optical_sar"
    assert r.intent == "optical_sar"  # overridden away from whatever the router said
    assert r.status == "partial"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "forced to 'optical_sar'" in override_step.output_summary


def test_unclear_takes_precedence_over_optical_sar_override(tmp_path):
    """An out-of-scope/abstained query must stay 'unclear' even over an
    optical+SAR pair -- the strongest case for the override -- rather than
    being forced into an intent the router explicitly declined to guess."""
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    sar = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    r = run_analysis([opt, sar], "what will this look like in 2030")

    assert r.input_config == "optical_sar"
    assert r.intent == "unclear"
    assert r.status == "success"
    assert r.answer
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "no override" in override_step.output_summary.lower()


def test_single_image_non_describe_query_becomes_vqa(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "how much vegetation is visible here")

    assert r.input_config == "single"
    assert r.intent == "vqa"
    assert r.status == "partial"
    assert "model_unavailable" in r.warnings
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "vegetation" in override_step.output_summary  # original router intent recorded
    assert _step_names(r) == ["load_raster", "compatibility_check", "route",
                              "orchestration_override", "analysis"]


def test_single_image_locate_query_is_not_overridden_to_vqa(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "Where are the buildings?")

    assert r.input_config == "single"
    assert r.intent == "locate"  # locate is single-image-native; no override


def test_bitemporal_describe_is_kept_and_noted(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "What is visible in this image?")

    assert r.input_config == "bitemporal"
    assert r.intent == "describe"  # not forced into change/vegetation/water
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "bitemporal" in override_step.output_summary.lower()


def test_unreadable_file_fails_with_structured_error(tmp_path):
    missing = tmp_path / "missing.tif"  # never written

    r = run_analysis([str(missing)], "describe this")

    assert r.status == "failed"
    assert r.warnings[0] == "unreadable_file"
    assert r.input_config is None


def test_too_many_files_fails_with_structured_error(tmp_path):
    paths = [_optical(tmp_path / f"{n}.tif") for n in "abc"]

    r = run_analysis(paths, "describe this")

    assert r.status == "failed"
    assert r.warnings[0] == "too_many_inputs"


def test_no_files_fails_with_structured_error():
    r = run_analysis([], "describe this")

    assert r.status == "failed"
    assert r.warnings[0] == "no_inputs"


def test_crs_mismatch_fails_with_input_warnings_preserved(tmp_path):
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=from_origin(500000, 4000000, 10, 10))
    b = _optical(tmp_path / "b.tif", crs=CRS.from_epsg(32644), transform=from_origin(500000, 4000000, 10, 10))

    r = run_analysis([a, b], "what changed")

    assert r.status == "failed"
    assert r.warnings[0] == "incompatible_pair"
    assert len(r.metadata.inputs) == 2  # both loaded fine; only pairing failed


def test_modality_hint_count_mismatch_is_warned_and_ignored(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "describe this", modalities=["optical", "sar"])

    assert any("modality hint count" in w for w in r.warnings)
    assert r.metadata.inputs[0].modality == "optical"  # auto-detected, hint ignored


def test_modality_hint_applied_when_count_matches(tmp_path):
    # A 3-band image would auto-detect as optical; force it to "sar" to prove
    # the per-file hint actually reaches load_raster.
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "describe this", modalities=["sar"])

    assert r.metadata.inputs[0].modality == "sar"
