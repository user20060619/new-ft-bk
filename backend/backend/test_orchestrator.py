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

    # evidence: mask + overlay, actually written to the same job directory
    assert {e.kind for e in r.evidence} == {"mask", "overlay"}
    assert set(r.report_assets) == {e.id for e in r.evidence}
    job_dir = a.parent
    assert (job_dir / "change_mask.png").exists()
    assert (job_dir / "change_overlay.jpg").exists()

    # resampling and alignment are their own auditable steps, separate from
    # the diff/threshold analysis itself
    assert _step_names(r)[-3:] == ["preprocess_resample", "preprocess_align", "analysis"]

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
    assert "square kilometres" in r.answer.lower()


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


def test_original_filename_preserved_in_metadata(tmp_path):
    # Saved on disk as input_0.tif (main.py's convention); the caller's own
    # upload name should still show up in metadata, not the saved name.
    path = tmp_path / "input_0.tif"
    _optical(path)

    r = run_analysis([str(path)], "describe this", original_filenames=["Mumbai25.jpg"])

    assert r.metadata.inputs[0].filename == "Mumbai25.jpg"


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
