#!/usr/bin/env python3
"""Tests for the orchestrator's agentic flow.  Owner: P4.

    python -m pytest backend/test_orchestrator.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.orchestrator as orchestrator_module  # noqa: E402
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


def _before_after_arrays(bands, height, width, seed):
    """Identical images except for one modified block, so change detection
    has real texture to find SIFT keypoints in and a real change to find."""
    rng = np.random.default_rng(seed)
    before = rng.integers(0, 255, size=(bands, height, width), dtype=np.uint8)
    after = before.copy()
    after[:, height // 4: 3 * height // 4, width // 4: 3 * width // 4] = (
        255 - before[:, height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    )
    return before, after


def _uniform_optical(path, band_values, width=40, height=40, crs=None, transform=None, dtype=np.uint8):
    array = np.zeros((len(band_values), height, width), dtype=dtype)
    for i, v in enumerate(band_values):
        array[i] = v
    _write_geotiff(path, array, crs=crs, transform=transform)
    return str(path)


def _unknown_modality(path, width=40, height=40, crs=None, transform=None):
    """1-band uint8, mid-range values -- not float, no SAR hint, not in a
    plausible dB range -> auto-detects as modality='unknown'."""
    array = np.random.randint(50, 150, size=(1, height, width)).astype(np.uint8)
    _write_geotiff(path, array, crs=crs, transform=transform)
    return str(path)


def _built_up_increase_before_after(width=60, height=60, seed=11):
    """Dim, textured background (identical on both dates, for SIFT texture)
    plus a bright/dark R=G=B checkerboard block injected only in `after` --
    high brightness + high edge density (built-up) but no colour signal (can't
    leak into the water/vegetation colour proxies), so built-up should show a
    clear increase while water/vegetation stay near flat."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 80, size=(3, height, width), dtype=np.uint8)
    before = base.copy()
    after = base.copy()
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    ys, xs = np.meshgrid(np.arange(y1 - y0), np.arange(x1 - x0), indexing="ij")
    checker = ((xs // 3 + ys // 3) % 2 == 0)
    block = np.where(checker, 230, 30).astype(np.uint8)
    for band in range(3):
        after[band, y0:y1, x0:x1] = block
    return before, after


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


def test_change_detection_no_real_change(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    rng = np.random.default_rng(9)
    array = rng.integers(0, 255, size=(3, 60, 60), dtype=np.uint8)
    a = tmp_path / "a.tif"
    _write_geotiff(a, array, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, array.copy(), crs=UTM43, transform=t)  # byte-identical

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    assert r.status == "success"
    assert r.computed["pct_changed"] < 1.0
    assert "no significant change" in r.answer.lower()


def test_change_detection_finds_real_change_with_known_gsd(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)  # 10 m pixels -> gsd_m known
    before, after = _before_after_arrays(3, 60, 60, seed=1)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    assert r.status == "success"
    assert r.computed["pct_changed"] > 1.0
    assert "area_changed_km2" in r.computed
    assert r.computed["area_changed_km2"] > 0

    # evidence: before/after images, mask + overlay, plus per-date/per-class
    # masks (both dates are 3-band optical -> same_modality -> landcover runs)
    assert {"mask", "overlay", "image"} <= {e.kind for e in r.evidence}
    # T18: r.evidence also carries an input_N_preview per loaded image now --
    # deliberately excluded from report_assets (the report's separate "Input"
    # section uses them directly; report_assets stays the handler's own set).
    non_preview_ids = {e.id for e in r.evidence if not e.id.endswith("_preview")}
    assert set(r.report_assets) == non_preview_ids
    job_dir = a.parent
    assert (job_dir / "change_mask.png").exists()
    assert (job_dir / "change_overlay.jpg").exists()
    assert (job_dir / "before.jpg").exists()
    assert (job_dir / "after.jpg").exists()

    # T12: alignment/difference/heatmap evidence, ported back from the
    # pre-T5 geo_service.py::generate_visualizations
    for filename in ("alignment.jpg", "difference.jpg", "heatmap.jpg"):
        path = job_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0
    evidence_by_id = {e.id: e for e in r.evidence}
    assert evidence_by_id["alignment"].kind == "overlay"
    assert evidence_by_id["difference"].kind == "image"
    assert evidence_by_id["heatmap"].kind == "heatmap"

    # T9: per-class change + regions, computed over the aligned overlap
    assert 0 <= r.computed["overlap_pct"] <= 100
    for cls in ("water", "vegetation", "built_up"):
        assert f"{cls}_direction" in r.computed
    assert isinstance(r.computed["regions"], list)

    # T9: full fixed step sequence for the change intent
    assert _step_names(r) == [
        "load_raster", "compatibility_check", "route", "orchestration_override",
        "preprocess_resample", "preprocess_align", "landcover_before", "landcover_after",
        "analysis", "per_class_change", "regions",
    ]

    resample_step = next(s for s in r.execution if s.name == "preprocess_resample")
    assert resample_step.params["target_width"] == 60
    assert resample_step.params["target_height"] == 60

    align_step = next(s for s in r.execution if s.name == "preprocess_align")
    assert "SIFT" in align_step.method
    assert "geo_service" in align_step.tool
    # Random-noise fixtures don't reliably give SIFT enough coherent structure
    # to clear the 10-good-match threshold even when most of the image is
    # byte-identical; the clean-alignment success path is covered separately
    # in test_geo_service.py with a self-identical (guaranteed-match) fixture.
    assert isinstance(align_step.fallback, bool)
    assert isinstance(align_step.params["good_matches"], int)
    assert isinstance(align_step.params["homography_found"], bool)

    # classical method, honestly labelled as a fallback for the not-yet-built model
    step = next(s for s in r.execution if s.name == "analysis")
    assert step.fallback is True
    assert "SIFT" not in step.method  # alignment no longer happens in this step


def test_change_detection_without_known_gsd_reports_percentage_only(tmp_path):
    before, after = _before_after_arrays(3, 50, 50, seed=2)
    a = tmp_path / "a.jpg"
    Image.fromarray(np.transpose(before, (1, 2, 0)), mode="RGB").save(a, format="JPEG", quality=95)
    b = tmp_path / "b.jpg"
    Image.fromarray(np.transpose(after, (1, 2, 0)), mode="RGB").save(b, format="JPEG", quality=95)

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.intent == "change"
    assert r.status == "success"
    assert r.computed["pct_changed"] > 1.0
    assert "area_changed_km2" not in r.computed
    assert any("ground sample distance" in w.lower() for w in r.warnings)
    assert "ground sample distance is unknown" in r.answer.lower()


def test_change_detection_failure_reported_as_partial_with_error_step(tmp_path, monkeypatch):
    """A break in any of the three change-detection stages must degrade to a
    clean partial result, not crash -- and the failing step must still show
    up in the trace (contract.py's ExecutionTrace records a step even when
    its block raises)."""
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator_module, "align_images", _boom)

    t = from_origin(500000, 4000000, 10, 10)
    before, after = _before_after_arrays(3, 60, 60, seed=3)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.intent == "change"
    assert r.status == "partial"
    assert "model_unavailable" in r.warnings
    failed_step = next(s for s in r.execution if s.name == "preprocess_align")
    assert "error:" in failed_step.output_summary.lower()
    assert "boom" in failed_step.output_summary.lower()
    # the earlier, successful step is still recorded
    assert any(s.name == "preprocess_resample" for s in r.execution)


def test_optical_sar_pair_forces_optical_sar_intent_regardless_of_query(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    sar = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    r = run_analysis([opt, sar], "has vegetation decreased here")

    assert r.input_config == "optical_sar"
    assert r.intent == "optical_sar"  # overridden away from whatever the router said
    # T8: a real handler is now registered, so a genuine optical+SAR pair
    # succeeds rather than falling through to _unavailable_handler.
    assert r.status == "success"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "forced to 'optical_sar'" in override_step.output_summary
    assert _step_names(r)[-4:] == ["preprocess_resample", "optical_landcover", "sar_processing", "fusion"]


def test_optical_sar_answer_points_are_the_sentences_answer_is_joined_from(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    sar = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    r = run_analysis([opt, sar], "has vegetation decreased here")

    assert r.status == "success"
    assert len(r.answer_points) == 2
    assert "\n".join(r.answer_points) == r.answer
    assert r.answer_points[0].lower().startswith("water")
    assert r.answer_points[1].lower().startswith("built-up")


def test_optical_sar_works_regardless_of_upload_order(tmp_path):
    """check_inputs only guarantees the SET {optical, sar}, not which upload
    came first -- the handler must select by .modality, not position."""
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    sar = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    r_optical_first = run_analysis([opt, sar], "has vegetation decreased here")
    r_sar_first = run_analysis([sar, opt], "has vegetation decreased here")

    assert r_optical_first.status == "success"
    assert r_sar_first.status == "success"
    assert r_optical_first.computed == r_sar_first.computed


def test_optical_sar_resamples_when_sizes_differ(tmp_path):
    opt = _optical(tmp_path / "opt.tif", width=40, height=40, crs=UTM43,
                   transform=from_origin(500000, 4000000, 5, 5))
    sar = _sar(tmp_path / "sar.tif", width=20, height=20, crs=UTM43,
              transform=from_origin(500000, 4000000, 10, 10))

    r = run_analysis([opt, sar], "has vegetation decreased here")

    assert r.status == "success"
    resample_step = next(s for s in r.execution if s.name == "preprocess_resample")
    assert resample_step.params["resampled"] is True
    assert resample_step.params["method"] == "reproject (CRS-aware, nearest)"
    assert resample_step.params["sar_original_size"] == [20, 20]
    assert resample_step.params["target_size"] == [40, 40]


def test_optical_sar_panchromatic_optical_reports_water_from_sar_only(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    band = np.full((40, 40), 80, dtype=np.uint8)
    ys, xs = np.meshgrid(np.arange(20), np.arange(40), indexing="ij")
    band[20:, :] = np.where(((xs // 3) + (ys // 3)) % 2 == 1, 220, 40)  # bright checkerboard
    pan_path = tmp_path / "pan.tif"
    _write_geotiff(pan_path, band[np.newaxis, :, :], crs=UTM43, transform=t)

    sar = _sar(tmp_path / "sar.tif", width=40, height=40, crs=UTM43, transform=t)

    r = run_analysis([str(pan_path), sar], "has vegetation decreased here",
                     modalities=["optical", "sar"])

    assert r.status == "success"
    assert r.computed["water_optical_pct"] is None
    assert r.computed["water_agreement_pct"] is None
    assert r.computed["water_union_pct"] == r.computed["water_sar_pct"]
    assert isinstance(r.computed["built_up_optical_pct"], float)
    assert isinstance(r.computed["built_up_agreement_pct"], float)
    assert any("water fusion used sar only" in w.lower() for w in r.warnings)
    assert "optical could not compute a water estimate" in r.answer.lower()


def test_input_preview_evidence_for_multiband_uint16_and_float_sar(tmp_path):
    """T18: every loaded input gets a browser-safe input_N_preview PNG,
    regardless of band count/dtype/modality -- a real GeoTIFF pair (4-band
    uint16 optical + 1-band float SAR) is exactly what a plain <img> can't
    render, so this is the case the fix exists for."""
    t = from_origin(500000, 4000000, 10, 10)
    optical_array = np.random.randint(0, 65535, size=(4, 40, 40)).astype(np.uint16)
    optical_path = tmp_path / "a.tif"
    _write_geotiff(optical_path, optical_array, crs=UTM43, transform=t)

    sar_array = (np.random.rand(1, 40, 40) * -30).astype(np.float32)
    sar_path = tmp_path / "b.tif"
    _write_geotiff(sar_path, sar_array, crs=UTM43, transform=t)

    r = run_analysis(
        [str(optical_path), str(sar_path)],
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
    )

    assert r.status == "success"
    preview0 = next(e for e in r.evidence if e.id == "input_0_preview")
    assert preview0.kind == "image"
    assert preview0.modality == "optical"
    preview1 = next(e for e in r.evidence if e.id == "input_1_preview")
    assert preview1.kind == "image"
    assert preview1.modality == "sar"
    assert (tmp_path / "input_0_preview.png").exists()
    assert (tmp_path / "input_1_preview.png").exists()


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


def test_unclear_out_of_scope_stays_unclear_for_bitemporal_too(tmp_path):
    """Same out-of-scope precedence as the optical_sar case above, now that
    bitemporal also defaults a non-out-of-scope 'unclear' to 'change' -- a
    genuinely out-of-scope query must still not get defaulted."""
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "what will this look like in 2030")

    assert r.input_config == "bitemporal"
    assert r.intent == "unclear"
    assert r.status == "success"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "no override" in override_step.output_summary.lower()


def test_bitemporal_unclear_non_out_of_scope_defaults_to_change(tmp_path):
    """The router has no "built-up area" vocabulary, so this abstains as
    'unclear' with a plain no-rule-matched reason (not out-of-scope) -- a
    bitemporal pair should default to 'change' rather than abstain, since
    bitemporal input is fundamentally a change question."""
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "Has the built-up area increased, decreased, or remained unchanged?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "defaulted to 'change'" in override_step.output_summary
    assert "bitemporal" in override_step.output_summary.lower()


def test_confidence_router_is_null_when_override_replaces_unclear(tmp_path):
    """T10/T14: the router's own confidence is meaningless once its 'unclear'
    verdict has been entirely discarded for a structural default -- shown as
    null, with the reason moved into `basis`, rather than a misleading ~0.0."""
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "Has the built-up area increased, decreased, or remained unchanged?")

    assert r.intent == "change"
    assert r.confidence.router is None
    assert "router suggested 'unclear'" in r.confidence.basis
    assert "overridden to 'change'" in r.confidence.basis
    assert "abstained" in r.confidence.basis.lower()
    assert "defaulted to 'change'" in r.confidence.basis


def test_confidence_router_is_null_on_confident_override_not_just_unclear(tmp_path):
    """T14: nulled for *any* override, not just one starting from 'unclear'
    -- an optical+SAR pair force-routes to 'optical_sar' even when the router
    confidently scored a different intent (here, 'vegetation', already proven
    elsewhere in this codebase to score confidently for this exact phrase),
    and that score would be just as misleading to display unqualified."""
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical(tmp_path / "opt.tif", crs=UTM43, transform=t)
    sar = _sar(tmp_path / "sar.tif", crs=UTM43, transform=t)

    r = run_analysis([opt, sar], "has vegetation decreased here")

    assert r.intent == "optical_sar"
    assert r.confidence.router is None
    assert "router suggested 'vegetation'" in r.confidence.basis
    assert "overridden to 'optical_sar'" in r.confidence.basis


def test_confidence_structured_fields_filled_when_overridden(tmp_path):
    """T15: router_suggested_intent/score + override_reason are additive
    fields alongside the existing `basis` prose (unchanged, for the report)
    -- let the live UI render the same override facts as separate bullets."""
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _optical(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "Has the built-up area increased, decreased, or remained unchanged?")

    assert r.intent == "change"
    assert r.confidence.router_suggested_intent == "unclear"
    assert r.confidence.router_suggested_score is not None
    assert r.confidence.override_reason is not None
    assert "defaulted to 'change'" in r.confidence.override_reason
    assert r.confidence.method_basis == "classical OpenCV alignment + threshold method; no calibrated confidence score"


def test_confidence_structured_override_fields_null_when_not_overridden(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "Where are the buildings?")

    assert r.intent == "locate"
    assert r.confidence.router is not None
    assert r.confidence.router_suggested_intent is None
    assert r.confidence.router_suggested_score is None
    assert r.confidence.override_reason is None
    assert r.confidence.method_basis is not None


def test_confidence_router_real_value_when_not_overriding_unclear(tmp_path):
    """A confident router intent redirected to a different handler (not an
    'unclear' override) keeps the router's real score -- it's still an
    accurate number, just for a different final intent."""
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "Where are the buildings?")

    assert r.intent == "locate"
    assert r.confidence.router is not None
    assert isinstance(r.confidence.router, float)


def test_confidence_router_real_value_when_genuinely_stays_unclear(tmp_path):
    """The terminal 'stays unclear' path (no override happened) is untouched
    -- the router's real confidence there is accurate."""
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "what will this look like in 2030")

    assert r.intent == "unclear"
    assert r.confidence.router is not None


def test_single_image_non_describe_query_becomes_vqa(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "how much vegetation is visible here")

    assert r.input_config == "single"
    assert r.intent == "vqa"
    assert r.status == "success"  # T17: real handler now, not the T10 stub
    assert r.computed["question_type"] == "amount"
    assert r.computed["class_asked"] == "vegetation"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "vegetation" in override_step.output_summary  # original router intent recorded
    assert _step_names(r) == ["load_raster", "compatibility_check", "route",
                              "orchestration_override", "landcover", "analysis"]


def test_single_image_locate_query_is_not_overridden_to_vqa(tmp_path):
    path = _optical(tmp_path / "a.tif")

    r = run_analysis([path], "Where are the buildings?")

    assert r.input_config == "single"
    assert r.intent == "locate"  # locate is single-image-native; no override


# --- T13: bitemporal "locate" -> "change", single-image locate handler ------

def test_bitemporal_locate_query_overridden_to_change(tmp_path):
    """The real-use bug: 'Did any buildings appear?' used to route to
    'locate', which has no bitemporal handler -> _unavailable_handler ->
    'not available yet', no evidence."""
    t = from_origin(500000, 4000000, 10, 10)
    before, after = _built_up_increase_before_after(seed=31)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "Did any buildings appear?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    assert r.status == "success"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "locate" in override_step.output_summary.lower()

    # "buildings" is already in _BUILT_UP_QUERY_HINTS -- verify the answer
    # actually leads with the built-up clause + mentions regions, not just
    # that the intent resolved.
    assert r.answer.lower().startswith("built-up")
    assert r.computed["built_up_direction"] == "increased"
    assert len(r.computed["regions"]) >= 1


def test_apply_overrides_bitemporal_locate_to_change():
    final_intent, note = orchestrator_module._apply_overrides("bitemporal", "locate", "")
    assert final_intent == "change"
    assert "locate" in note.lower()


# --- T17: _apply_overrides widening (single-image unclear/locate -> vqa) ----

def test_apply_overrides_single_unclear_defaults_to_vqa():
    final_intent, note = orchestrator_module._apply_overrides(
        "single", "unclear", "ambiguous between two intents", "some query")
    assert final_intent == "vqa"
    assert "defaulted to 'vqa'" in note


def test_apply_overrides_single_out_of_scope_unclear_is_untouched():
    final_intent, note = orchestrator_module._apply_overrides(
        "single", "unclear", "out of scope: prediction", "what will this look like")
    assert final_intent == "unclear"
    assert note is None


def test_apply_overrides_single_locate_water_or_vegetation_forced_to_vqa():
    final_intent, note = orchestrator_module._apply_overrides(
        "single", "locate", "", "Where is the water?")
    assert final_intent == "vqa"
    assert "water" in note.lower()

    final_intent, note = orchestrator_module._apply_overrides(
        "single", "locate", "", "Where are the trees?")
    assert final_intent == "vqa"
    assert "vegetation" in note.lower()


def test_apply_overrides_single_locate_buildings_stays_locate():
    final_intent, note = orchestrator_module._apply_overrides(
        "single", "locate", "", "Where are the buildings?")
    assert final_intent == "locate"
    assert note is None


def test_single_image_unclear_non_out_of_scope_defaults_to_vqa(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = _uniform_optical(tmp_path / "a.tif", [20, 60, 90], crs=UTM43, transform=t)

    r = run_analysis([path], "Show the location of water regions in this image.")

    assert r.input_config == "single"
    assert r.intent == "vqa"
    assert r.confidence.router is None
    assert "defaulted to 'vqa'" in r.confidence.basis


def test_single_image_locate_water_query_overridden_to_vqa(tmp_path):
    """T17: 'locate' is built-up-only, so a water 'where is' query would
    otherwise get an honest-but-useless not-computable answer from
    _locate_handler -- it's forced to 'vqa' instead, which can actually
    answer it."""
    t = from_origin(500000, 4000000, 10, 10)
    path = _uniform_optical(tmp_path / "a.tif", [20, 60, 90], crs=UTM43, transform=t)

    r = run_analysis([path], "Where is the water?")

    assert r.input_config == "single"
    assert r.intent == "vqa"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "locate" in override_step.output_summary.lower()
    assert "water" in override_step.output_summary.lower()
    assert r.status == "success"
    assert r.computed["question_type"] == "location"
    assert r.computed["class_asked"] == "water"
    assert r.computed["total_region_count"] >= 1
    assert len(r.computed["regions"]) >= 1
    regions_evidence = next(e for e in r.evidence if e.id == "water_regions")
    assert regions_evidence.kind == "boxes"
    assert (tmp_path / "water_regions.jpg").exists()


def test_single_image_locate_vegetation_query_overridden_to_vqa(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = _uniform_optical(tmp_path / "a.tif", [50, 200, 100], crs=UTM43, transform=t)

    r = run_analysis([path], "Where are the trees?")

    assert r.input_config == "single"
    assert r.intent == "vqa"
    override_step = next(s for s in r.execution if s.name == "orchestration_override")
    assert "locate" in override_step.output_summary.lower()
    assert "vegetation" in override_step.output_summary.lower()
    assert r.status == "success"
    assert r.computed["question_type"] == "location"
    assert r.computed["class_asked"] == "vegetation"
    assert r.computed["total_region_count"] >= 1
    regions_evidence = next(e for e in r.evidence if e.id == "vegetation_regions")
    assert regions_evidence.kind == "boxes"
    assert (tmp_path / "vegetation_regions.jpg").exists()


# --- T17: VQA question/class classification (unit-level, no router) --------

def test_classify_vqa_class_water_synonyms():
    for phrase in ("is there a river here", "how much lake coverage", "any sea visible", "the water level"):
        assert orchestrator_module._classify_vqa_class(phrase) == "water"


def test_classify_vqa_class_vegetation_synonyms():
    for phrase in ("how much forest is there", "any trees visible", "greenery coverage", "is there vegetation"):
        assert orchestrator_module._classify_vqa_class(phrase) == "vegetation"


def test_classify_vqa_class_built_up_synonyms():
    for phrase in ("any buildings here", "how urban is this area", "how many houses", "built-up percentage"):
        assert orchestrator_module._classify_vqa_class(phrase) == "built_up"


def test_classify_vqa_class_unrecognised():
    assert orchestrator_module._classify_vqa_class("describe this scene") is None


def test_classify_vqa_question_type_hints():
    assert orchestrator_module._classify_vqa_question_type("where is the water") == "location"
    assert orchestrator_module._classify_vqa_question_type("how much vegetation is there") == "amount"
    assert orchestrator_module._classify_vqa_question_type("is there any water") == "presence"
    assert orchestrator_module._classify_vqa_question_type("describe this scene") == "open_ended"


# --- T17: _vqa_handler -------------------------------------------------------

def test_vqa_presence_yes(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = _uniform_optical(tmp_path / "a.tif", [20, 60, 90], crs=UTM43, transform=t)

    r = run_analysis([path], "Is there any water in this image?")

    assert r.intent == "vqa"
    assert r.status == "success"
    assert r.computed["question_type"] == "presence"
    assert r.computed["class_asked"] == "water"
    assert "yes" in r.answer.lower()
    assert " ".join(r.answer_points) == r.answer


def test_vqa_presence_no(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = _uniform_optical(tmp_path / "a.tif", [200, 150, 100], crs=UTM43, transform=t)

    r = run_analysis([path], "Is there any water in this image?")

    assert r.intent == "vqa"
    assert r.computed["question_type"] == "presence"
    assert r.computed["class_asked"] == "water"
    assert "no" in r.answer.lower()


def test_vqa_amount_vegetation(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = _uniform_optical(tmp_path / "a.tif", [50, 200, 100], crs=UTM43, transform=t)

    r = run_analysis([path], "How much of the image is vegetation?")

    assert r.intent == "vqa"
    assert r.status == "success"
    assert r.computed["question_type"] == "amount"
    assert r.computed["class_asked"] == "vegetation"
    assert r.computed["vegetation_percentage"] == 100.0
    assert "100.0%" in r.answer
    assert " ".join(r.answer_points) == r.answer


def test_vqa_open_ended_unrecognised_class_uses_vlm_fallback(tmp_path):
    """'What condition is this area in?' names no recognised class, so this
    hits the VLM-with-context fallback."""
    t = from_origin(500000, 4000000, 10, 10)
    path = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)

    r = run_analysis([path], "What condition is this area in?")

    assert r.intent == "vqa"
    assert r.computed["class_asked"] is None
    assert r.status == "partial"
    assert "model_unavailable" in r.warnings
    assert r.computed["water_percentage"] is not None
    assert r.computed["vegetation_percentage"] is not None


def test_vqa_open_ended_vlm_unavailable(tmp_path):
    """Runs for real on this CPU-only machine -- no CUDA, so _load_vlm()
    raises and the handler degrades honestly instead of crashing."""
    t = from_origin(500000, 4000000, 10, 10)
    path = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)

    r = run_analysis([path], "What is the dominant land cover type here?")

    assert r.intent == "vqa"
    assert r.computed["question_type"] == "open_ended"
    assert r.status == "partial"
    assert "model_unavailable" in r.warnings
    assert "vision-language model" in r.answer.lower()
    # grounding data is still returned even though the VLM call failed
    assert r.computed["water_percentage"] is not None
    assert r.computed["vegetation_percentage"] is not None
    assert r.computed["built_up_percentage"] is not None


def test_vqa_open_ended_vlm_available(tmp_path, monkeypatch):
    t = from_origin(500000, 4000000, 10, 10)
    path = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)

    def fake_caption(model, processor, image_path, prompt):
        return "This is a mixed scene with some open ground.", 0.1, 42

    monkeypatch.setattr(orchestrator_module, "_load_vlm", lambda: (object(), object(), fake_caption))

    r = run_analysis([path], "What is the dominant land cover type here?")

    assert r.intent == "vqa"
    assert r.status == "success"
    assert r.answer == "This is a mixed scene with some open ground."
    assert r.answer_points == [r.answer]
    analysis_step = next(s for s in r.execution if s.name == "analysis")
    assert analysis_step.fallback is False


def test_vqa_sar_water_computable_vegetation_not(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = _sar(tmp_path / "a.tif", crs=UTM43, transform=t)

    r_water = run_analysis([path], "Is there any water in this image?")
    assert r_water.intent == "vqa"
    assert r_water.status == "success"
    assert r_water.computed["water_percentage"] is not None

    r_veg = run_analysis([path], "How much of the image is vegetation?")
    assert r_veg.intent == "vqa"
    assert r_veg.computed["vegetation_percentage"] is None
    assert "not computable" in r_veg.answer.lower()
    assert "sar" in r_veg.answer.lower()


def test_vqa_panchromatic_water_not_computable_built_up_still_works(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    band = np.full((60, 60), 30, dtype=np.uint8)
    ys, xs = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    band[:, :] = np.where(((xs // 3) + (ys // 3)) % 2 == 1, 220, 20)
    path = tmp_path / "pan.tif"
    _write_geotiff(path, band[np.newaxis, :, :], crs=UTM43, transform=t)

    r_water = run_analysis([str(path)], "Is there any water in this image?", modalities=["optical"])
    assert r_water.intent == "vqa"
    assert r_water.computed["water_percentage"] is None
    assert "not computable" in r_water.answer.lower()

    r_built = run_analysis([str(path)], "How much of the image is built-up?", modalities=["optical"])
    assert r_built.intent == "vqa"
    assert r_built.computed["built_up_percentage"] is not None
    assert r_built.computed["built_up_percentage"] > 0


def test_locate_handler_single_optical_image_finds_built_up_regions(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    _, image_with_block = _built_up_increase_before_after(seed=32)
    path = tmp_path / "a.tif"
    _write_geotiff(path, image_with_block, crs=UTM43, transform=t)

    r = run_analysis([str(path)], "Where are the buildings?")

    assert r.input_config == "single"
    assert r.intent == "locate"
    assert r.status == "success"
    assert r.computed["total_region_count"] >= 1
    assert len(r.computed["regions"]) >= 1
    region = r.computed["regions"][0]
    assert len(region["bbox"]) == 4
    assert region["dominant_class_change"] is None  # no before/after concept here

    boxes_evidence = next(e for e in r.evidence if e.kind == "boxes")
    assert "rule-based" in boxes_evidence.label.lower()
    assert "not object detection" in boxes_evidence.label.lower()

    lowered = r.answer.lower()
    assert "not true building detection" in lowered
    assert "not available" in lowered
    assert "yolo" in r.confidence.basis.lower() or "dota" in r.confidence.basis.lower()

    job_dir = path.parent
    assert (job_dir / "built_up_regions.jpg").exists()


def test_locate_answer_points_are_the_sentences_answer_is_joined_from(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    _, image_with_block = _built_up_increase_before_after(seed=33)
    path = tmp_path / "a.tif"
    _write_geotiff(path, image_with_block, crs=UTM43, transform=t)

    r = run_analysis([str(path)], "Where are the buildings?")

    assert r.status == "success"
    assert len(r.answer_points) == 2
    assert " ".join(r.answer_points) == r.answer


def test_locate_handler_non_optical_single_image_is_honest_not_computable(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = tmp_path / "a.tif"
    _sar(path, crs=UTM43, transform=t)

    r = run_analysis([str(path)], "Where are the buildings?")

    assert r.input_config == "single"
    assert r.intent == "locate"
    assert r.status == "success"
    assert r.computed["regions"] == []
    assert "sar" in r.answer.lower() or "optical" in r.answer.lower()


def test_locate_handler_panchromatic_single_image_still_finds_built_up(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    band = np.full((60, 60), 30, dtype=np.uint8)
    ys, xs = np.meshgrid(np.arange(20), np.arange(60), indexing="ij")
    band[20:40, :] = np.where(((xs // 3) + (ys // 3)) % 2 == 1, 220, 20)  # bright checkerboard
    path = tmp_path / "pan.tif"
    _write_geotiff(path, band[np.newaxis, :, :], crs=UTM43, transform=t)

    r = run_analysis([str(path)], "Where are the buildings?", modalities=["optical"])

    assert r.intent == "locate"
    assert r.status == "success"
    assert r.computed["built_up_percentage"] > 0
    assert r.computed["total_region_count"] >= 1


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


def test_original_filename_preserved_in_metadata(tmp_path):
    # Saved on disk as input_0.tif (main.py's convention); the caller's own
    # upload name should still show up in metadata, not the saved name.
    path = tmp_path / "input_0.tif"
    _optical(path)

    r = run_analysis([str(path)], "describe this", original_filenames=["Mumbai25.jpg"])

    assert r.metadata.inputs[0].filename == "Mumbai25.jpg"


def test_metadata_bounds_present_for_georeferenced_input(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    path = tmp_path / "a.tif"
    _optical(path, crs=UTM43, transform=t)

    r = run_analysis([str(path)], "describe this")

    bounds = r.metadata.inputs[0].bounds
    assert bounds is not None
    west, south, east, north = bounds
    assert west < east
    assert south < north


def test_metadata_bounds_none_without_georeferencing(tmp_path):
    pixels = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
    path = tmp_path / "input_0.jpg"
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG")

    r = run_analysis([str(path)], "describe this")

    assert r.metadata.inputs[0].bounds is None


def test_warnings_are_prefixed_with_original_filename(tmp_path):
    pixels = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
    path = tmp_path / "input_0.jpg"
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG")

    r = run_analysis([str(path)], "describe this", original_filenames=["Mumbai25.jpg"])

    assert any(w.startswith("Mumbai25.jpg: No georeferencing available") for w in r.warnings)


def test_identical_per_file_warnings_are_not_repeated(tmp_path):
    # Two different saved files that happen to carry the same original
    # filename (e.g. duplicate upload) must not double the same warning text.
    pixels = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
    a = tmp_path / "input_0.jpg"
    Image.fromarray(pixels, mode="RGB").save(a, format="JPEG")
    b = tmp_path / "input_1.jpg"
    Image.fromarray(pixels, mode="RGB").save(b, format="JPEG")

    r = run_analysis([str(a), str(b)], "what changed",
                      original_filenames=["photo.jpg", "photo.jpg"])

    georef_warning = "photo.jpg: No georeferencing available for this format; crs and gsd_m are unset."
    assert r.warnings.count(georef_warning) == 1


# --- T9: per-class change, regions, and query-aware answers -----------------

def test_change_detection_built_up_increase_and_regions(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    before, after = _built_up_increase_before_after(seed=11)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.status == "success"
    assert r.computed["built_up_direction"] == "increased"
    assert r.computed["built_up_change_pct_points"] > 0
    assert r.computed["water_direction"] is not None
    assert r.computed["vegetation_direction"] is not None
    assert len(r.computed["regions"]) >= 1
    largest = r.computed["regions"][0]
    assert largest["area_px"] > 0
    assert len(largest["bbox"]) == 4


def test_change_answer_built_up_query_leads_with_built_up(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    before, after = _built_up_increase_before_after(seed=12)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r_built_up = run_analysis([str(a), str(b)], "Has the built-up area increased, decreased, or remained unchanged?")
    r_generic = run_analysis([str(a), str(b)], "What changed between these two images and where?")

    assert r_built_up.answer.lower().startswith("built-up")
    assert not r_generic.answer.lower().startswith("built-up")


def test_change_answer_points_are_the_sentences_answer_is_joined_from(tmp_path):
    """T14: answer_points is the same content as `answer`, just split into
    separate bullet strings rather than pre-joined into one paragraph."""
    t = from_origin(500000, 4000000, 10, 10)
    before, after = _built_up_increase_before_after(seed=13)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.status == "success"
    assert len(r.answer_points) > 1
    assert " ".join(r.answer_points) == r.answer


def test_change_detection_panchromatic_pair_water_vegetation_not_computable(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    before = np.random.randint(0, 255, size=(1, 60, 60), dtype=np.uint8)
    after = np.random.randint(0, 255, size=(1, 60, 60), dtype=np.uint8)
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "What changed between these two images?",
                      modalities=["optical", "optical"])

    assert r.status == "success"
    assert r.computed["water_direction"] is None
    assert r.computed["vegetation_direction"] is None
    assert r.computed["built_up_direction"] is not None  # built-up needs no colour info
    assert not any(e.id in ("water_change_mask", "vegetation_change_mask") for e in r.evidence)
    lowered = r.answer.lower()
    assert "water change could not be computed" in lowered
    assert "vegetation change could not be computed" in lowered


def test_change_detection_sar_pair_vegetation_not_computable_water_and_built_up_real(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _sar(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _sar(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "What changed between these two images?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    assert r.status == "success"
    assert r.computed["vegetation_direction"] is None
    assert r.computed["water_direction"] is not None
    assert r.computed["built_up_direction"] is not None
    landcover_before = next(s for s in r.execution if s.name == "landcover_before")
    assert landcover_before.tool == "sar.extract_sar"
    assert not any(e.id == "vegetation_change_mask" for e in r.evidence)
    assert any(e.id.startswith("before_water") for e in r.evidence)
    assert not any(e.id.startswith("before_vegetation") for e in r.evidence)


def test_change_detection_mixed_modality_skips_per_class_but_keeps_pixel_diff(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _optical(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _unknown_modality(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "What changed between these two images?")

    assert r.input_config == "bitemporal"
    assert r.intent == "change"
    assert r.status == "success"
    assert r.computed["water_direction"] is None
    assert r.computed["vegetation_direction"] is None
    assert r.computed["built_up_direction"] is None
    assert "pct_changed" in r.computed  # pixel-level diff still computed
    landcover_before = next(s for s in r.execution if s.name == "landcover_before")
    assert "skipped" in landcover_before.output_summary.lower()
    assert any("modality" in w.lower() for w in r.warnings)
    # regions still detected, just without a dominant class label
    for region in r.computed["regions"]:
        assert region["dominant_class_change"] is None


# --- T9: bitemporal vegetation/water handlers --------------------------------

def test_vegetation_bitemporal_rgb_proxy_never_says_ndvi(tmp_path):
    a = _uniform_optical(tmp_path / "a.tif", [50, 60, 100])
    b = _uniform_optical(tmp_path / "b.tif", [50, 200, 100])

    r = run_analysis([a, b], "has vegetation increased here")

    assert r.input_config == "bitemporal"
    assert r.intent == "vegetation"
    assert r.status == "success"
    assert r.computed["is_true_index"] is False
    # T10: the answer explicitly disclaims NDVI ("not NDVI") rather than
    # claiming it -- never "mean NDVI" or similar affirmative usage.
    assert "mean ndvi" not in r.answer.lower()
    assert "not ndvi" in r.answer.lower()
    assert "colour proxy" in r.answer.lower()
    assert "vegetation" in orchestrator_module.SERVICE_REGISTRY


def test_vegetation_bitemporal_proxy_answer_leads_with_area_not_proxy_mean_pct(tmp_path):
    a = _uniform_optical(tmp_path / "a.tif", [50, 60, 100])
    b = _uniform_optical(tmp_path / "b.tif", [50, 200, 100])

    r = run_analysis([a, b], "has vegetation increased here")

    assert r.status == "success"
    assert r.computed["is_true_index"] is False
    # T10: leads with the class-based area percentage-point sentence, not a
    # "% change" narrated from the proxy's own (uncalibrated) mean value.
    assert r.answer.lower().startswith("vegetation area")
    assert "percentage points" in r.answer.lower()
    # the proxy means are still available, just not narrated as a % change
    assert "vegetation_proxy_before" in r.computed
    assert "vegetation_proxy_after" in r.computed


def test_vegetation_answer_points_are_the_sentences_answer_is_joined_from(tmp_path):
    a = _uniform_optical(tmp_path / "a.tif", [50, 60, 100])
    b = _uniform_optical(tmp_path / "b.tif", [50, 200, 100])

    r = run_analysis([a, b], "has vegetation increased here")

    assert r.status == "success"
    assert len(r.answer_points) == 2
    assert " ".join(r.answer_points) == r.answer


def test_vegetation_bitemporal_true_ndvi_with_nir(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _uniform_optical(tmp_path / "a.tif", [50, 60, 100, 200], crs=UTM43, transform=t)
    b = _uniform_optical(tmp_path / "b.tif", [50, 60, 50, 200], crs=UTM43, transform=t)

    r = run_analysis([a, b], "has vegetation increased here")

    assert r.status == "success"
    assert r.computed["is_true_index"] is True
    assert "ndvi" in r.answer.lower()
    assert r.computed["vegetation_before_pct"] is not None


def test_water_bitemporal_true_ndwi_with_nir(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _uniform_optical(tmp_path / "a.tif", [30, 200, 80, 50], crs=UTM43, transform=t)
    b = _uniform_optical(tmp_path / "b.tif", [30, 100, 80, 150], crs=UTM43, transform=t)

    r = run_analysis([a, b], "has water decreased here")

    assert r.status == "success"
    assert r.intent == "water"
    assert r.computed["is_true_index"] is True
    assert "ndwi" in r.answer.lower()


def test_vegetation_bitemporal_change_regions_and_evidence(tmp_path):
    """T16: bitemporal vegetation/water gets the same connected-components
    regions treatment as the generic `change` handler, run on this class's
    own XOR mask -- so the "Change regions" tab and a boxed evidence image
    work for vegetation/water results too, not just the generic intent."""
    t = from_origin(500000, 4000000, 10, 10)
    width = height = 60
    rng = np.random.default_rng(21)
    # Dim, textured, non-vegetation-like background (identical on both dates,
    # for SIFT texture); a strong green block appears only in `after`, inside
    # one sub-region -- vegetation-classified after but not before.
    base = rng.integers(20, 80, size=(3, height, width), dtype=np.uint8)
    before = base.copy()
    after = base.copy()
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    after[0, y0:y1, x0:x1] = 20   # red
    after[1, y0:y1, x0:x1] = 220  # green -- excess-green proxy spikes
    after[2, y0:y1, x0:x1] = 20   # blue

    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "has vegetation increased here")

    assert r.status == "success"
    assert r.intent == "vegetation"
    assert r.computed["total_region_count"] >= 1
    assert len(r.computed["regions"]) >= 1
    region = r.computed["regions"][0]
    assert len(region["bbox"]) == 4
    assert region["area_px"] > 0

    regions_evidence = next(e for e in r.evidence if e.id == "vegetation_change_regions")
    assert regions_evidence.kind == "boxes"
    assert (tmp_path / "vegetation_change_regions.jpg").exists()


def test_vegetation_bitemporal_panchromatic_forced_optical_not_computable(tmp_path):
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_geotiff(a, np.random.randint(0, 255, size=(1, 40, 40), dtype=np.uint8))
    _write_geotiff(b, np.random.randint(0, 255, size=(1, 40, 40), dtype=np.uint8))

    r = run_analysis([str(a), str(b)], "has vegetation increased here", modalities=["optical", "optical"])

    assert r.status == "success"
    assert r.computed["pct_change"] is None
    assert "could not be computed" in r.answer.lower()


def test_vegetation_bitemporal_sar_pair_not_computable_guard(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    a = _sar(tmp_path / "a.tif", crs=UTM43, transform=t)
    b = _sar(tmp_path / "b.tif", crs=UTM43, transform=t)

    r = run_analysis([a, b], "has vegetation increased here")

    assert r.status == "success"
    assert r.computed["pct_change"] is None
    assert any("sar" in w.lower() or "optical" in w.lower() for w in r.warnings)


def test_change_detection_regions_truncated_to_top_10_with_total_count(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    rng = np.random.default_rng(21)
    size = 250
    before = rng.integers(0, 255, size=(3, size, size), dtype=np.uint8)
    after = before.copy()
    block = 10
    n_blocks = 15
    for i in range(n_blocks):
        y = x = 10 + 16 * i
        after[:, y:y + block, x:x + block] = 255 - before[:, y:y + block, x:x + block]
    a = tmp_path / "a.tif"
    _write_geotiff(a, before, crs=UTM43, transform=t)
    b = tmp_path / "b.tif"
    _write_geotiff(b, after, crs=UTM43, transform=t)

    r = run_analysis([str(a), str(b)], "What changed between these two images?")

    assert r.status == "success"
    assert r.computed["total_region_count"] >= 10
    assert len(r.computed["regions"]) <= 10
    if r.computed["total_region_count"] > 10:
        assert len(r.computed["regions"]) == 10
        assert str(r.computed["total_region_count"]) in r.answer


def test_vegetation_handler_uses_aligned_overlap_not_native_frame(tmp_path, monkeypatch):
    """Spy: confirms the vegetation handler passes real alignment info through
    to calculate_vegetation_index, rather than silently falling back to the
    old (pre-T9) native-full-frame default."""
    a = _uniform_optical(tmp_path / "a.tif", [50, 60, 100])
    b = _uniform_optical(tmp_path / "b.tif", [50, 200, 100])

    captured = {}
    real_fn = orchestrator_module.calculate_vegetation_index

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "calculate_vegetation_index", _spy)

    r = run_analysis([a, b], "has vegetation increased here")

    assert r.status == "success"
    assert captured.get("target_size") is not None
    assert "valid_mask" in captured and captured["valid_mask"] is not None
