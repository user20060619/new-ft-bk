"""CLI: run one or two real files through load_raster + check_inputs and
print what came out, for eyeballing real data before wiring it into the API.

    python -m backend.rsio.inspect <file> [<file>]
"""
from __future__ import annotations

import argparse
import sys

from .compat import check_inputs
from .raster import RasterInput, load_raster


def _load(path: str) -> RasterInput | None:
    try:
        return load_raster(path)
    except Exception as e:  # noqa: BLE001 -- CLI: report and continue, never crash
        print(f"  ERROR: could not load {path}: {type(e).__name__}: {e}")
        return None


def _print_raster(path: str, r: RasterInput | None) -> None:
    print(f"\n{path}")
    if r is None:
        print("  (unreadable)")
        return
    print(f"  modality   {r.modality}")
    print(f"  bands      {r.bands}")
    print(f"  dtype      {r.dtype}")
    print(f"  size       {r.width} x {r.height}")
    print(f"  crs        {r.crs}")
    print(f"  gsd_m      {r.gsd_m}")
    print(f"  date       {r.date}")
    if r.warnings:
        print("  warnings:")
        for w in r.warnings:
            print(f"    - {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Print load_raster metadata for one or two files, and "
                    "check_inputs compatibility for a pair."
    )
    ap.add_argument("files", nargs="+", help="one or two raster/image files")
    args = ap.parse_args(argv)

    if len(args.files) > 2:
        print("Only 1 or 2 files are supported.", file=sys.stderr)
        return 1

    loaded = [_load(f) for f in args.files]
    for path, r in zip(args.files, loaded):
        _print_raster(path, r)

    if len(args.files) == 2:
        result = check_inputs(loaded)
        print("\ncompatibility check")
        print(f"  input_config  {result.input_config}")
        print(f"  accepted      {result.accepted}")
        if result.errors:
            print("  errors:")
            for e in result.errors:
                print(f"    - {e.code}: {e.message}")
        if result.warnings:
            print("  warnings:")
            for w in result.warnings:
                print(f"    - {w}")
        print("  steps:")
        for s in result.steps:
            print(f"    - {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
