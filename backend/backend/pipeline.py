#!/usr/bin/env python3
"""End-to-end query pipeline: router -> tool -> fusion.

Owner: P1.  This is the integration seam I own (plan section P1, 9-10 Sep),
brought forward to 31 August so that on the 9th we are swapping implementations
behind an interface we have already exercised, rather than discovering the
interface was wrong.

Right now the four services are STUBS returning fixed numbers.  P4 and P2
replace them with real implementations; nothing above this line changes.

    python backend/pipeline.py "has vegetation decreased" --images 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion.explain import explain  # noqa: E402
from router.intent import route
from rsio.raster import load_raster  # noqa: E402
from services.geo_service import calculate_vegetation_index  # noqa: E402

# --- service stubs ---------------------------------------------------------
# Shapes match docs/api_contract_request.md exactly.  When P4 and P2 ship the
# real services these functions are replaced by calls into them and the
# signature stays identical.

def _stub_geo_ndvi(image_ids: list[str]) -> dict[str, Any]:
    return {"mean_ndvi_before": 0.612, "mean_ndvi_after": 0.525,
            "pct_change": -14.2, "area_changed_km2": 3.47,
            "threshold_used": 0.3}

def _real_geo_ndvi(image_ids: list[str]) -> dict[str, Any]:
    """Run real vegetation-index calculation using geo_service.

    fusion/explain.py's vegetation templates say "Mean NDVI" unconditionally
    and require area_changed_km2 -- both hard requirements this file cannot
    edit around. So a colour-proxy result (no NIR band) or an unknown-GSD
    result is returned as an `error` dict instead of being routed through the
    NDVI template, rather than mislabel a proxy as NDVI or fabricate an area.
    """
    if len(image_ids) < 2:
        return {"error": "Two images are required for vegetation analysis"}

    before = load_raster(image_ids[0])
    after = load_raster(image_ids[1])
    result = calculate_vegetation_index(before, after)

    if not result["is_true_index"]:
        return {"error": (
            "NIR band not identifiable; only an RGB colour proxy is available "
            f"({result['vegetation_proxy_before']:.2f} -> "
            f"{result['vegetation_proxy_after']:.2f}), which this endpoint "
            "does not template as NDVI"
        )}

    if "area_changed_km2" not in result:
        return {"error": (
            "Ground sample distance is unknown, so the changed area could not "
            "be computed; area_changed_km2 is required to report this result"
        )}

    return {
        "mean_ndvi_before": result["index_before"],
        "mean_ndvi_after": result["index_after"],
        "pct_change": result["pct_change"],
        "area_changed_km2": result["area_changed_km2"],
        "threshold_used": 0.3,
    }


def _stub_geo_ndwi(image_ids: list[str]) -> dict[str, Any]:
    return {"mean_ndwi_before": 0.310, "mean_ndwi_after": 0.239,
            "pct_change": -22.8, "area_changed_km2": 1.88,
            "threshold_used": 0.0}


def _stub_change(image_ids: list[str]) -> dict[str, Any]:
    return {"changed_pixels": 21740, "total_pixels": 262144,
            "pct_changed": 8.3, "area_changed_km2": 2.11}


def _stub_detect(image_ids: list[str]) -> dict[str, Any]:
    return {"count": 47, "classes": [{"class": "building", "count": 47}],
            "boxes": [[120, 88, 176, 141, "building", 0.91]]}


def _stub_caption(image_ids: list[str]) -> str:
    return ("A residential area with regularly spaced buildings, "
            "intersected by two main roads and bordered by open grassland.")


# --- real VLM, loaded lazily -----------------------------------------------
# Kept out of module import so the router and fusion stay usable without a GPU.
# P5 and P6 can run the whole pipeline on a CPU laptop; only `describe` needs
# the model, and only when --vlm is passed.

_VLM: tuple = ()


def _real_caption(image_path: str) -> str:
    global _VLM
    if not _VLM:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vlm"))
        from load import load_model, caption as _cap
        model, processor, _ = load_model()
        _VLM = (model, processor, _cap)
    model, processor, _cap = _VLM
    return _cap(model, processor, image_path)[0]


SERVICES = {
    "geo_service.ndvi": _real_geo_ndvi,
    "geo_service.ndwi": _stub_geo_ndwi,
    "change_service.detect": _stub_change,
    "detect_service.detect": _stub_detect,
}

OVERLAY = {
    "vegetation": "mask", "water": "mask",
    "change": "mask", "locate": "boxes",
}


def answer_query(query: str, image_ids: list[str] | None = None,
                 use_vlm: bool = False) -> dict[str, Any]:
    """One query in, one API-shaped response out.

    `use_vlm=True` runs the real model for the describe intent; otherwise a
    stub caption is used so the pipeline runs without a GPU.
    """
    image_ids = image_ids or []
    t0 = time.perf_counter()

    decision = route(query, n_images=len(image_ids))

    if decision.intent == "unclear":
        exp = explain("unclear", clarification=decision.clarification)
        tool_out: dict[str, Any] = {}
    elif decision.intent == "describe":
        if use_vlm and image_ids and Path(image_ids[0]).exists():
            caption = _real_caption(image_ids[0])
        else:
            caption = _stub_caption(image_ids)
        exp = explain("describe", caption=caption)
        tool_out = {}
    else:
        service = SERVICES[decision.tool]
        tool_out = service(image_ids)
        exp = explain(decision.intent, computed=tool_out)

    return {
        "intent": decision.intent,
        "confidence": decision.confidence,
        "tool": decision.tool,
        "answer": exp.answer,
        "computed": exp.computed,
        "overlay": {
            "type": OVERLAY.get(decision.intent, "none"),
            "url": None,
            "width": 512, "height": 512,
        },
        "meta": {
            "source": "live" if use_vlm else "stub",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "router_reason": decision.reason,
            "generated_text": exp.generated,
            "template": exp.template_used,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--vlm", action="store_true",
                    help="use the real model for describe (needs a GPU)")
    ap.add_argument("--image", action="append", default=None,
                    help="real image path; repeat for two images")
    args = ap.parse_args()

    queries = [" ".join(args.query)] if args.query else [
        "What is visible in this image?",
        "Has vegetation decreased here?",
        "Did the reservoir shrink?",
        "What changed between these two images?",
        "Where are the buildings?",
        "What will this look like in 2030?",
    ]
    ids = args.image or [f"img_{i}" for i in range(args.images)]

    for q in queries:
        r = answer_query(q, ids, use_vlm=args.vlm)
        if args.json:
            print(json.dumps(r, indent=2))
            continue
        gen = "GENERATED" if r["meta"]["generated_text"] else "computed"
        print(f"\n\033[1m{q}\033[0m")
        print(f"  intent   {r['intent']} ({r['confidence']})  via {r['tool']}")
        print(f"  answer   [{gen}] {r['answer']}")
        if r["computed"]:
            print(f"  computed {r['computed']}")
        print(f"  overlay  {r['overlay']['type']}   {r['meta']['elapsed_ms']}ms")


if __name__ == "__main__":
    main()
