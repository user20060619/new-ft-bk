# SatQuery AI — 3-day Claude Code task list

How to use: paste **one task at a time**. Start each in plan mode (Shift+Tab), read the plan, then let it build. After each task: run the check yourself, then `git commit`. If a task goes sideways, `git restore .` and re-prompt with what went wrong. Don't stack tasks in one prompt.

Before starting, ask teammates for: co-registered optical + SAR GeoTIFF pairs and bi-temporal GeoTIFF pairs (P3), an importable `vqa()` / `caption()` function and LoRA status (P1), whether a trained change model exists (P2).

---

## DAY 1 — Backend foundation

### T0 — Orientation (no code changes)
```
Read CLAUDE.md, then read backend/backend/main.py, pipeline.py, router/intent.py, fusion/explain.py, services/geo_service.py, and frontend/src/services/api.js. Don't change anything. Tell me: how route() and explain() are called and what they return, which functions in geo_service are reusable, and anything in CLAUDE.md that doesn't match the code.
```
Check: its summary matches CLAUDE.md "Known problems".

### T1 — Raster loader
```
Create backend/backend/io/raster.py with load_raster(path) -> RasterInput (dataclass: array as (bands,H,W) float32, filename, modality guess, band count, original dtype, width, height, crs, transform, gsd_m, date if in tags, nodata). Use rasterio for TIFF/GeoTIFF; fall back to PIL/cv2 for PNG/JPEG with crs/gsd None. Modality guess: 1-2 bands with float data or values in a typical dB range -> "sar"; >=3 bands -> "optical"; 1 band uint -> "unknown" (add a warning). Allow an explicit modality override argument. Add pytest tests with synthetic rasters written to tmp_path (3-band uint8 GeoTIFF with CRS, 4-band uint16, 1-band float32 SAR, plain JPEG). Run the tests.
```
Check: `pytest -q` passes.

### T2 — Compatibility check
```
Create backend/backend/io/compat.py with check_inputs(list[RasterInput]) -> CompatResult (input_config: single|bitemporal|optical_sar, accepted bool, errors with codes, warnings, list of steps performed). Rules: 1 input -> single; 2 optical -> bitemporal; 1 optical + 1 sar -> optical_sar; 2 sar -> bitemporal (SAR change). For pairs: if both have CRS, they must match and bounds must overlap (else incompatible_pair); if sizes differ, flag "resample_required" (resampling happens later). More than 2 inputs or unreadable file -> structured error. Tests for each case. Run tests.
```

### T3 — Contract + execution tracer
```
Create backend/backend/contract.py implementing the unified response contract from CLAUDE.md as pydantic models, plus an ExecutionTrace helper with a context manager `with trace.step(name, tool, method, params) as s:` that records timing, output_summary and fallback. Include a helper to build a failed response from an error code. Tests. Run tests.
```

### T4 — Unified /analyze through the router
```
Rewrite POST /analyze in main.py to accept 1 or 2 files (files: list[UploadFile]) plus query plus optional modality hints. Save uploads keeping their original extension. Flow, each step recorded in ExecutionTrace: load_raster -> check_inputs -> route(query, n_images) -> orchestration override (if input_config is optical_sar, intent becomes optical_sar; if bitemporal and router says describe/vqa, keep it but note it in the trace) -> dispatch to a service registry -> explain() for wording where it applies -> unified response. Don't edit router/intent.py; do the override in a new backend/backend/orchestrator.py. For now, services not yet built return status "partial" with a model_unavailable warning — no fake numbers. Delete generate_text_answer. Keep /vqa as a thin alias that forwards to the new flow so the current frontend doesn't break. Add an API test with FastAPI TestClient. Run tests.
```
Check: `curl` a single image and a pair; response has `execution` and `input_config`.

### T5 — Honesty fixes in geo_service
```
Refactor geo_service so its functions take RasterInput arrays instead of file paths. Compute true NDVI/NDWI only when a NIR band is identifiable (4+ band optical; allow a band-order config, default B,G,R,NIR). Otherwise compute "vegetation_proxy"/"water_proxy" from RGB, name them that way in computed output and add a warning. Report area_km2 only when gsd_m is known. Remove the hardcoded 0.85 confidence. Keep the existing alignment + diff visualisation code working. Tests. Run tests.
```

---

## DAY 2 — Mandatory analysis paths

### T6 — Land-cover extraction (optical)
```
Create backend/backend/services/landcover.py: extract_optical(RasterInput) -> masks for water, vegetation, built_up plus per-class percentages and method labels (true index vs proxy). Built-up from brightness + edge/texture density on non-water non-vegetation pixels, labelled rule-based. Save masks as PNG evidence. Tests on synthetic images with known regions.
```

### T7 — SAR processing
```
Create backend/backend/services/sar.py: to_db (skip if data already looks like dB), lee_filter (window configurable, default 5), water_mask via Otsu threshold on low backscatter, built_up_mask via high-backscatter percentile threshold, percentages, method labels, PNG evidence (filtered SAR display + masks). Tests with synthetic SAR (speckled low-backscatter water patch, bright built-up patch).
```

### T8 — Optical–SAR fusion service
```
Create backend/backend/services/optical_sar.py: given a co-registered optical + SAR pair (resample SAR to optical grid if needed, record in trace), run landcover on optical and sar.py on SAR, then produce for water and built-up: optical-only, SAR-only, agreement mask and union; percentages of each; and a note on what SAR adds (e.g. pixels SAR flags as water where optical is cloudy/bright). Evidence: optical, SAR, fused overlay with legend labels. Register it in the orchestrator for intent optical_sar and answer queries like "use optical and SAR together to identify built-up and water regions" via templated text with computed numbers. Tests. Run tests.
```

### T9 — Change understanding
```
Extend the bitemporal path: align (existing SIFT/homography code), run landcover on both dates, compute per-class change (percentage points and direction: increased/decreased/unchanged with a stated tolerance), plus the existing diff mask and regions for "where". Answer "has built-up area increased/decreased/unchanged", "what changed and where". Mark the diff method as fallback: true unless a P2 model is available. Evidence: before, after, change overlay, per-class change masks. Tests.
```

### T10 — VLM wiring (VQA + captioning)
```
Create backend/backend/services/vlm_adapter.py with vqa(image, question, context) and caption(image). Lazily import P1's functions from models/vlm/load.py (ask me for the exact function names if unclear). Pass computed land-cover stats as context so numbers come from computation, not generation. If the model can't load (no GPU, missing weights), return status partial with model_unavailable — never a fake answer. Record model name, adapter version (or "base, no RS adapter") and params in the trace. Register for intents vqa and describe.
```
Check on the demo laptop with a GPU, not just your machine.

---

## DAY 3 — Frontend, report, freeze

### T11 — Unified upload
```
In the frontend, replace the Change Detection / Ask About an Image mode tabs with one upload area accepting 1-2 files (TIFF/GeoTIFF/PNG/JPEG), an optional modality selector per file (auto / optical / SAR), and one query box. Update services/api.js to call the new /analyze with files[]. After a response, show a badge for input_config. Show backend validation errors clearly. Keep changes in App.jsx minimal; extract new components into their own files.
```

### T12 — Result + execution panel
```
Render the unified response: answer, computed values, evidence chosen by intent (single: source + relevant masks; bitemporal: before/after slider + overlay + per-class changes; optical_sar: optical, SAR, fused with labels), warnings in a visible note, and a collapsible "Execution details" panel listing the execution steps (name, tool, method, key params, output summary, ms, fallback badge). Show confidence only when not null, with its basis on hover. Projector-friendly: large text, high contrast.
```

### T13 — Report
```
Update utils/report.js to build the report from the unified response: header (SatQuery AI, date/time, query, input filenames + metadata), then Input -> Question -> Evidence (only report_assets) -> Answer -> Execution summary table -> Warnings/limitations. Keep the print-to-PDF approach.
```

### T14 — End-to-end check
```
Write backend/scripts/e2e_demo.py that runs the full /analyze flow on every file in data/demo_pairs (or a path I give) for these queries: a VQA question, "describe this image", "what changed between these two dates and where", "has the built-up area increased", "use the optical and SAR images together to identify built-up and water regions". Print status, intent, input_config, answer, warnings and total time per case, and save all responses as JSON. Report anything that fails.
```
Run it on P3's demo laptop, fix failures, then freeze: bug fixes only.

---

## Cut unless everything above is done
Login/demo account, admin approval, YOLO locate, benchmark evaluation harness.
