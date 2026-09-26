#!/usr/bin/env python3
"""Tests for the inspect CLI.  Owner: P4.

    python -m pytest backend/rsio/test_inspect.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.inspect import main  # noqa: E402


def _write_geotiff(path, array, crs=None, transform=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=count,
        dtype=array.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(array)


def test_inspect_single_file_prints_metadata(tmp_path, capsys):
    array = np.random.randint(0, 255, size=(3, 20, 20), dtype=np.uint8)
    path = tmp_path / "a.tif"
    _write_geotiff(path, array, crs=CRS.from_epsg(32643), transform=from_origin(0, 0, 10, 10))

    code = main([str(path)])

    out = capsys.readouterr().out
    assert code == 0
    assert "modality   optical" in out
    assert "bands      3" in out
    assert "size       20 x 20" in out
    assert "gsd_m      10.0" in out
    assert "compatibility check" not in out


def test_inspect_pair_prints_compat_result(tmp_path, capsys):
    transform = from_origin(500000, 4000000, 10, 10)
    crs = CRS.from_epsg(32643)
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_geotiff(a, np.random.randint(0, 255, size=(3, 20, 20), dtype=np.uint8), crs=crs, transform=transform)
    _write_geotiff(b, np.random.randint(0, 255, size=(3, 20, 20), dtype=np.uint8), crs=crs, transform=transform)

    code = main([str(a), str(b)])

    out = capsys.readouterr().out
    assert code == 0
    assert "compatibility check" in out
    assert "input_config  bitemporal" in out
    assert "accepted      True" in out


def test_inspect_unreadable_file_does_not_crash(tmp_path, capsys):
    bad = tmp_path / "missing.tif"  # never written

    code = main([str(bad)])

    out = capsys.readouterr().out
    assert code == 0
    assert "ERROR" in out
    assert "(unreadable)" in out


def test_inspect_too_many_files_errors(tmp_path, capsys):
    array = np.random.randint(0, 255, size=(3, 10, 10), dtype=np.uint8)
    paths = []
    for name in ("a.tif", "b.tif", "c.tif"):
        p = tmp_path / name
        _write_geotiff(p, array)
        paths.append(str(p))

    code = main(paths)

    assert code == 1
    assert "Only 1 or 2 files" in capsys.readouterr().err
