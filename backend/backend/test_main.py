#!/usr/bin/env python3
"""API-level tests for main.py, using FastAPI's TestClient.  Owner: P4.

    python -m pytest backend/test_main.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.main import app  # noqa: E402

client = TestClient(app)

UTM43 = CRS.from_epsg(32643)
UNIFIED_RESPONSE_KEYS = {
    "request_id", "status", "input_config", "intent", "answer", "computed",
    "evidence", "confidence", "execution", "metadata", "warnings", "report_assets",
}


def _write_geotiff(path, array, crs=None, transform=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)


def _optical_bytes(tmp_path, name="a.tif", crs=None, transform=None):
    array = np.random.randint(0, 255, size=(3, 30, 30), dtype=np.uint8)
    path = tmp_path / name
    _write_geotiff(path, array, crs=crs, transform=transform)
    return path.read_bytes()


def _sar_bytes(tmp_path, name="s.tif", crs=None, transform=None):
    array = (np.random.rand(1, 30, 30) * -30).astype(np.float32)
    path = tmp_path / name
    _write_geotiff(path, array, crs=crs, transform=transform)
    return path.read_bytes()


def test_health_check():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "online"


def test_analyze_single_image_returns_unified_contract_shape(tmp_path):
    data = _optical_bytes(tmp_path)

    resp = client.post(
        "/analyze",
        files=[("files", ("a.tif", data, "image/tiff"))],
        data={"query": "What is visible in this image?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert UNIFIED_RESPONSE_KEYS <= set(body)
    assert body["input_config"] == "single"
    assert body["intent"] == "describe"
    # No GPU in this environment -> honest degradation, not a fake caption.
    assert body["status"] == "partial"
    assert "model_unavailable" in body["warnings"]


def test_analyze_optical_sar_pair_forces_optical_sar_intent(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical_bytes(tmp_path, "opt.tif", crs=UTM43, transform=t)
    sar = _sar_bytes(tmp_path, "sar.tif", crs=UTM43, transform=t)

    resp = client.post(
        "/analyze",
        files=[
            ("files", ("opt.tif", opt, "image/tiff")),
            ("files", ("sar.tif", sar, "image/tiff")),
        ],
        data={"query": "has vegetation decreased here"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["input_config"] == "optical_sar"
    assert body["intent"] == "optical_sar"


def test_analyze_unclear_query_wins_over_optical_sar_override(tmp_path):
    t = from_origin(500000, 4000000, 10, 10)
    opt = _optical_bytes(tmp_path, "opt.tif", crs=UTM43, transform=t)
    sar = _sar_bytes(tmp_path, "sar.tif", crs=UTM43, transform=t)

    resp = client.post(
        "/analyze",
        files=[
            ("files", ("opt.tif", opt, "image/tiff")),
            ("files", ("sar.tif", sar, "image/tiff")),
        ],
        data={"query": "what will this look like in 2030"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["input_config"] == "optical_sar"
    assert body["intent"] == "unclear"
    assert body["status"] == "success"


def test_analyze_too_many_files_returns_structured_error(tmp_path):
    files = [("files", (f"{n}.tif", _optical_bytes(tmp_path, f"{n}.tif"), "image/tiff"))
             for n in "abc"]

    resp = client.post("/analyze", files=files, data={"query": "describe this"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["warnings"][0] == "too_many_inputs"


def test_analyze_no_files_returns_structured_error():
    resp = client.post("/analyze", data={"query": "describe this"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["warnings"][0] == "no_inputs"


def test_vqa_forwards_to_new_flow_and_keeps_legacy_fields(tmp_path):
    data = _optical_bytes(tmp_path)

    resp = client.post(
        "/vqa",
        files={"image": ("a.tif", data, "image/tiff")},
        data={"query": "What is visible in this image?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    # legacy fields the current frontend reads directly
    assert body["message"] == body["answer"]
    assert body["success"] is True  # status == "partial", not "failed"
    # new unified fields are also present
    assert UNIFIED_RESPONSE_KEYS <= set(body)
    assert body["intent"] == "describe"
    assert body["status"] == "partial"


def test_vqa_non_describe_query_reports_intent_vqa(tmp_path):
    """Same override rule as /analyze: a single image with a query the router
    places in vegetation/water/change is answered as 'vqa', not forced into a
    bitemporal-style analysis it has no second image to support."""
    data = _optical_bytes(tmp_path)

    resp = client.post(
        "/vqa",
        files={"image": ("a.tif", data, "image/tiff")},
        data={"query": "how much vegetation is visible here"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "vqa"
    assert body["status"] == "partial"
    assert body["success"] is True


def test_vqa_unreadable_file_reports_failure_not_500(tmp_path):
    resp = client.post(
        "/vqa",
        files={"image": ("bad.xyz", b"not an image", "application/octet-stream")},
        data={"query": "describe this"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["success"] is False
    assert body["warnings"][0] == "unreadable_file"
