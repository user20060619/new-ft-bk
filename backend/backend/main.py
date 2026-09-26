"""
SatQuery AI Backend - Main API Server
Final integration version matching frontend contract
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import uvicorn

from backend.pipeline import answer_query
from backend.contract import build_failed_response
from backend.orchestrator import run_analysis

app = FastAPI(
    title="SatQuery AI",
    description="Satellite change detection and query system for SIH 2026",
    version="1.0.0"
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for generated files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve static files for visualizations
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
async def health_check():
    """Backend health status endpoint"""
    return {
        "status": "online",
        "service": "SatQuery AI",
        "message": "Satellite change detection backend is running.",
        "timestamp": datetime.now().isoformat()
    }


def _save_upload(upload: UploadFile, job_dir: str, index: int) -> str:
    """Save keeping the original extension so load_raster can dispatch on it
    (GeoTIFF vs PNG/JPEG) -- CLAUDE.md known problem: uploads used to always
    be forced to .jpg."""
    ext = Path(upload.filename).suffix if upload.filename else ""
    dest = os.path.join(job_dir, f"input_{index}{ext}")
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@app.post("/analyze")
async def analyze_images(
    files: list[UploadFile] = File(default=[]),
    query: str = Form(...),
    modality: list[str] = Form(default=[]),
):
    """Agentic analysis over 1 or 2 satellite images: load_raster ->
    check_inputs -> route -> orchestration override -> dispatch -> unified
    response. `modality` is an optional per-file hint (same order as `files`),
    e.g. modality=optical&modality=sar.

    File-count and pairing errors (0, >2 files, unreadable, incompatible
    pair) are reported through the same response shape via check_inputs --
    they are not raised as HTTP errors, so the frontend always gets one shape.
    """
    try:
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        saved_paths = [_save_upload(f, job_dir, i) for i, f in enumerate(files)]
        original_filenames = [f.filename or Path(p).name for f, p in zip(files, saved_paths)]

        response = run_analysis(saved_paths, query, modalities=modality or None,
                                 original_filenames=original_filenames, request_id=job_id)
    except Exception as e:
        response = build_failed_response("internal_error", message=f"{type(e).__name__}: {e}",
                                          request_id=job_id)

    return JSONResponse(content=response.model_dump())


@app.post("/query")
async def query_endpoint(query: str, image_ids: list[str] = []):
    """Legacy endpoint for backward compatibility"""
    return answer_query(query, image_ids)


@app.post("/vqa")
async def vqa(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    """Single-image visual question answering.

    Thin alias: forwards to the same orchestration flow as /analyze with one
    file, so the current frontend (which still reads `message`/`stats`) keeps
    working while everything underneath is unified.
    """
    try:
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        saved_path = _save_upload(image, job_dir, 0)
        original_filename = image.filename or Path(saved_path).name

        response = run_analysis([saved_path], query, original_filenames=[original_filename],
                                 request_id=job_id)
    except Exception as e:
        response = build_failed_response("internal_error", message=f"{type(e).__name__}: {e}",
                                          request_id=job_id)

    payload = response.model_dump()
    payload["message"] = response.answer
    payload["success"] = response.status != "failed"
    if response.computed:
        payload["stats"] = response.computed

    return JSONResponse(content=payload)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
