# SatQuery AI — Claude Code context

SIH26167 (ISRO), Smart India Hackathon 2026, second-round prototype. **Deadline: 3 days.**
I am doing the P4 (backend) and P5 (frontend) work. Other members own ML models (P1), vision models (P2) and data/demo machine (P3).

## What the official problem statement requires (graded)

An agentic web app that answers natural-language queries over remote-sensing images:

1. **Single-image VQA** (mandatory) + one more single-image task (we choose **captioning**).
2. **Bi-temporal change understanding** (mandatory): change description / change VQA, e.g. "Has built-up area increased, decreased or stayed the same?" and "where did it change?"
3. **Optical–SAR paired analysis** (mandatory, the *principal focus*): extract complementary info from co-registered optical + SAR, e.g. built-up and water regions.
4. **Agentic orchestration**: system itself classifies the query, checks inputs (count, modality, format, metadata, compatibility), selects tools, runs them, fuses outputs, returns evidence + confidence + an **auditable execution trace** (task, tools, key params).
5. Inputs: **GeoTIFF/TIFF**; PNG/JPEG only for public benchmark images. Final evaluation uses Cartosat-2S optical + RISAT SAR pairs.
6. Downloadable report.

## Repo layout (actual)

- Backend runs from `backend/` as package `backend.*`: `backend/backend/main.py` (FastAPI), `pipeline.py`, `router/intent.py` + `rules.yaml` (P1), `fusion/explain.py` (P1), `services/geo_service.py` (OpenCV), `storage/`.
- `services/vlm_service.py`, `change_service.py`, `detect_service.py`, `models/change/siamese_unet.py`, `models/vlm/lora_train.py` are placeholders that `raise NotImplementedError` — owned by P1/P2. **Do not import them at module level.**
- `models/vlm/load.py` works (Qwen2.5-VL-3B, 4-bit, GPU). Load lazily only.
- Frontend: `frontend/src/App.jsx` (large, single file), `services/api.js`, `utils/report.js` (print-to-PDF), `OverlayCanvas.jsx`, `MapView.jsx`.
- `backend/frontend/` is a stale duplicate. Ignore it.

## Known problems to fix (verified)

- `/analyze` bypasses the router with its own keyword matcher `generate_text_answer`.
- Uploads are saved as `.jpg` regardless of format and read with `cv2.imread` → GeoTIFF/multiband/16-bit broken.
- `geo_service` uses BGR channel 2 (red) as "NIR" → the "NDVI/NDWI" on RGB are not real indices.
- Area in km² assumes 10 m pixels even when resolution is unknown.
- `"confidence": 0.85` hardcoded in `geo_service.py`.
- `/analyze`, `/vqa`, `/query` return three different response shapes.
- No SAR support anywhere.

## Non-negotiable rules

1. **Never fabricate numbers.** Every number in a response must be computed from pixels or metadata. If it can't be computed, omit it and add a `warnings` entry.
2. **Label methods honestly.** Classical/rule-based methods (thresholds, OpenCV diff, colour proxies) must say so in the execution trace (`method`) and set `fallback: true` where a learned model was intended.
3. RGB-only inputs: call indices "vegetation-like / water-like colour proxy", never NDVI/NDWI. True NDVI/NDWI only when a NIR band exists.
4. km² only when ground sample distance is known from metadata; otherwise percentage only.
5. Don't edit `router/intent.py`, `rules.yaml`, `fusion/explain.py` or P1/P2 model files. Wrap them instead. If a change there is truly required, stop and tell me.
6. Heavy models load lazily; the backend must start and serve non-VLM paths on a CPU-only machine.
7. Every new backend module gets pytest tests using small synthetic rasters (write them with rasterio into `tmp_path`). Run the tests before saying a task is done.
8. Keep changes small and task-scoped. Don't refactor unrelated code. Don't add auth/login/admin.

## Unified response contract (all analysis responses)

```json
{
  "request_id": "uuid",
  "status": "success | partial | failed",
  "input_config": "single | bitemporal | optical_sar",
  "intent": "vqa | describe | change | vegetation | water | optical_sar | unclear",
  "answer": "natural-language answer",
  "computed": { "any deterministic measurements": 0 },
  "evidence": [
    { "id": "e1", "kind": "image | mask | overlay | boxes | heatmap",
      "label": "SAR water mask", "modality": "optical | sar | fused | none", "url": "/outputs/<id>/x.png" }
  ],
  "confidence": { "router": 0.0, "analysis": null, "basis": "how it was computed, or why null" },
  "execution": [
    { "step": 1, "name": "compatibility_check", "tool": "input.compat", "method": "rasterio metadata",
      "params": {}, "output_summary": "2 inputs accepted: optical + SAR", "ms": 12, "fallback": false }
  ],
  "metadata": { "inputs": [ { "filename": "", "modality": "", "bands": 0, "dtype": "",
                 "width": 0, "height": 0, "crs": null, "gsd_m": null, "date": null } ] },
  "warnings": [ "strings" ],
  "report_assets": [ "evidence ids to include in the PDF" ]
}
```

Errors use the same shape with `status: "failed"` and a machine-readable code in `warnings`, e.g. `incompatible_pair`, `missing_band`, `unsupported_format`, `model_unavailable`.

## Commands

- Backend: `cd backend && uvicorn backend.main:app --reload --port 8000`
- Tests: `cd backend && pytest -q`
- Frontend: `cd frontend && npm run dev`
