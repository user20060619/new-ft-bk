"""Format-agnostic raster loader.

Owner: P4.  Fixes the "uploads saved as .jpg and read with cv2.imread" problem
(CLAUDE.md known problems) by giving every downstream service one input type,
`RasterInput`, regardless of whether the source was a GeoTIFF or a PNG/JPEG.

Rule this file exists to enforce: never invent metadata.  `crs`, `transform`,
`gsd_m` and `date` are set only when the source actually carries them; when
they cannot be determined, they are left `None` and a reason is appended to
`warnings` instead of guessing (e.g. assuming a pixel size).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

_RASTER_EXTENSIONS = {".tif", ".tiff"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

_MODALITIES = {"optical", "sar", "unknown"}

_DATE_TAG_KEYS = ("TIFFTAG_DATETIME", "DATE", "ACQUISITION_DATE", "date", "acquisition_date")

# Typical Sentinel-1/RISAT-style backscatter range once converted to dB.
# Public: shared with services/sar.py's to_db() so both agree on what
# "already looks like dB" means, rather than each maintaining its own copy.
SAR_DB_MIN = -60.0
SAR_DB_MAX = 15.0

# Polarisation / sensor names that show up in band descriptions or dataset
# tags of real SAR products (RISAT, Sentinel-1 GRD, ...).
_SAR_HINTS = ("vv", "vh", "hh", "hv", "sar", "risat", "sentinel-1", "grd")

DEFAULT_MAX_PIXELS = 4096 * 4096


@dataclass
class RasterInput:
    """One loaded image, normalised for the analysis pipeline.

    `array` is always (bands, height, width) float32, whatever the source
    dtype was -- `dtype` keeps a record of that original dtype since the cast
    itself is lossless-in-range but not identity.
    """
    array: np.ndarray
    filename: str
    modality: str
    bands: int
    dtype: str
    width: int
    height: int
    crs: str | None
    transform: tuple[float, float, float, float, float, float] | None
    gsd_m: float | None
    date: str | None
    nodata: float | None
    warnings: list[str] = field(default_factory=list)


def _extract_date(tags: dict[str, Any]) -> str | None:
    for key in _DATE_TAG_KEYS:
        raw = tags.get(key)
        if not raw:
            continue
        try:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").date().isoformat()
        except ValueError:
            return raw
    return None


def _compute_gsd(transform, crs) -> tuple[float | None, list[str]]:
    if transform is None:
        return None, ["No geotransform available; ground sample distance is unknown."]

    px, py = abs(transform.a), abs(transform.e)
    pixel_size = (px + py) / 2 if px and py else (px or py)
    if not pixel_size:
        return None, ["Geotransform has zero pixel size; ground sample distance is unknown."]

    if crs is None:
        return None, ["No CRS present; ground sample distance could not be expressed in metres."]

    if crs.is_geographic:
        return None, ["CRS is geographic (degrees); ground sample distance in metres could not be computed."]

    try:
        units = crs.linear_units
    except Exception:
        units = None
    if units and "met" not in units.lower():
        return None, [f"CRS linear unit is '{units}', not metres; ground sample distance left unset."]

    return float(pixel_size), []


def _has_sar_hint(tags: dict[str, Any], band_descriptions: tuple, band_tags: list[dict]) -> bool:
    """Case-insensitive, whole-word search for polarisation/sensor hints
    (VV, VH, HH, HV, SAR, RISAT, Sentinel-1, GRD) across dataset tags, band
    descriptions and per-band tags."""
    parts: list[str] = [str(v) for v in tags.values()]
    parts.extend(str(d) for d in band_descriptions if d)
    for bt in band_tags:
        parts.extend(str(v) for v in bt.values())
    haystack = " ".join(parts).lower()
    return any(re.search(rf"\b{re.escape(hint)}\b", haystack) for hint in _SAR_HINTS)


def _hint_says_sar(bands: int, original_dtype: str, tags: dict[str, Any],
                    band_descriptions: tuple, band_tags: list[dict]) -> bool:
    """Cheap, metadata-only SAR check -- usable before any pixel is read, so
    it can also decide the resampling method for a large-image downsample."""
    if bands >= 3:
        return False
    is_float_source = np.issubdtype(np.dtype(original_dtype), np.floating)
    return is_float_source or _has_sar_hint(tags, band_descriptions, band_tags)


def looks_like_sar_db(array: np.ndarray) -> bool:
    """True when every finite value already sits in a plausible SAR dB range
    (negative, within [SAR_DB_MIN, SAR_DB_MAX]).  Public and shared: used both
    for modality guessing here and by services/sar.py's to_db() to decide
    whether a "convert to dB" request should just be a no-op, so the two
    never drift into disagreeing about what "already dB" means."""
    finite = array[np.isfinite(array)]
    return (
        finite.size > 0
        and finite.min() < 0
        and finite.min() >= SAR_DB_MIN
        and finite.max() <= SAR_DB_MAX
    )


def _guess_modality(array: np.ndarray, original_dtype: str, tags: dict[str, Any] | None = None,
                     band_descriptions: tuple = (), band_tags: list[dict] | None = None
                     ) -> tuple[str, list[str]]:
    """>=3 bands -> optical.  1-2 bands -> sar if the source is float, or has a
    polarisation/sensor hint, or its values sit in a plausible dB range;
    otherwise -> unknown (the caller can override with modality="sar")."""
    bands = array.shape[0]
    if bands >= 3:
        return "optical", []

    if _hint_says_sar(bands, original_dtype, tags or {}, band_descriptions, band_tags or []):
        return "sar", []

    if looks_like_sar_db(array):
        return "sar", []

    return "unknown", [
        "Could not confidently determine modality from a single/dual-band "
        "integer image with no SAR-like (dB range) values or polarisation/sensor "
        "hints; defaulting to 'unknown'. Pass modality=\"sar\" to load_raster() "
        "explicitly if this is SAR data."
    ]


def _load_with_rasterio(path: Path, max_pixels: int) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    warnings: list[str] = []

    with rasterio.open(path) as src:
        width, height, bands = src.width, src.height, src.count
        original_dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
        crs = src.crs
        has_geotransform = not src.transform.is_identity
        tags = src.tags()
        band_descriptions = src.descriptions
        band_tags = [src.tags(i) for i in range(1, bands + 1)]
        nodata = src.nodata

        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            out_width = max(1, int(width * scale))
            out_height = max(1, int(height * scale))
            is_sar = _hint_says_sar(bands, original_dtype, tags, band_descriptions, band_tags)
            # SAR: nearest-neighbour, so no pixel is a blend of dB (log-scale)
            # values -- averaging dB values does not equal the dB of the
            # averaged power, which would quietly fabricate a derived number.
            # Optical: average, the standard/visually-correct image-pyramid
            # downsample for linear reflectance-like bands.
            resampling = Resampling.nearest if is_sar else Resampling.average
            array = src.read(
                out_shape=(bands, out_height, out_width), resampling=resampling
            ).astype(np.float32)
            transform = (
                src.transform @ src.transform.scale(width / out_width, height / out_height)
                if has_geotransform else None
            )
            warnings.append(
                f"Image downsampled from {width}x{height} to {out_width}x{out_height} "
                f"to stay within max_pixels={max_pixels} per band "
                f"({'nearest' if is_sar else 'average'} resampling, "
                f"{'SAR' if is_sar else 'optical'} branch)."
            )
            width, height = out_width, out_height
        else:
            array = src.read().astype(np.float32)
            transform = src.transform if has_geotransform else None

        info = {
            "original_dtype": original_dtype,
            "crs": crs,
            "transform": transform,
            "tags": tags,
            "band_descriptions": band_descriptions,
            "band_tags": band_tags,
            "nodata": nodata,
            "width": width,
            "height": height,
            "bands": bands,
        }
    return array, info, warnings


def _load_with_pil(path: Path, max_pixels: int) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    warnings: list[str] = []

    with Image.open(path) as img:
        width, height = img.size
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            out_width = max(1, int(width * scale))
            out_height = max(1, int(height * scale))
            # PNG/JPEG in this pipeline are always optical benchmark images
            # (CLAUDE.md), so an area/average-like filter matches the optical
            # branch above; BOX also lets JPEG's decoder skip full-res decode.
            img.draft(img.mode, (out_width, out_height))
            img = img.resize((out_width, out_height), Image.Resampling.BOX)
            warnings.append(
                f"Image downsampled from {width}x{height} to {img.size[0]}x{img.size[1]} "
                f"to stay within max_pixels={max_pixels} (PIL box/average resampling)."
            )
        arr = np.array(img)
        original_dtype = str(arr.dtype)

    if arr.ndim == 2:
        chw = arr[np.newaxis, :, :]
    else:
        chw = np.transpose(arr, (2, 0, 1))

    bands, height, width = chw.shape
    return chw.astype(np.float32), {
        "original_dtype": original_dtype,
        "bands": bands,
        "width": width,
        "height": height,
    }, warnings


def load_raster(path: str | Path, modality: str | None = None,
                 max_pixels: int = DEFAULT_MAX_PIXELS) -> RasterInput:
    """Load a TIFF/GeoTIFF via rasterio, or a PNG/JPEG via PIL.

    `modality` overrides the automatic guess ("optical" | "sar" | "unknown")
    when the caller already knows what the source is.

    `max_pixels` bounds width*height (per band); an image over that is read
    already-downsampled via rasterio/PIL, never at full size first, and the
    transform/gsd_m are scaled to match the returned array.
    """
    path = Path(path)
    ext = path.suffix.lower()
    warnings: list[str] = []

    if ext in _RASTER_EXTENSIONS:
        array, info, load_warnings = _load_with_rasterio(path, max_pixels)
        warnings.extend(load_warnings)
        crs = info["crs"].to_string() if info["crs"] else None
        transform = info["transform"]
        transform_tuple = tuple(transform)[:6] if transform is not None else None
        gsd_m, gsd_warnings = _compute_gsd(transform, info["crs"])
        warnings.extend(gsd_warnings)
        date = _extract_date(info["tags"])
        nodata = info["nodata"]
    elif ext in _IMAGE_EXTENSIONS:
        array, info, load_warnings = _load_with_pil(path, max_pixels)
        warnings.extend(load_warnings)
        crs = None
        transform_tuple = None
        gsd_m = None
        date = None
        nodata = None
        warnings.append("No georeferencing available for this format; crs and gsd_m are unset.")
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for {path.name}")

    if modality is not None:
        if modality not in _MODALITIES:
            raise ValueError(f"Unknown modality override '{modality}'; expected one of {_MODALITIES}")
        resolved_modality = modality
    else:
        resolved_modality, modality_warnings = _guess_modality(
            array, info["original_dtype"],
            tags=info.get("tags"),
            band_descriptions=info.get("band_descriptions", ()),
            band_tags=info.get("band_tags"),
        )
        warnings.extend(modality_warnings)

    return RasterInput(
        array=array,
        filename=path.name,
        modality=resolved_modality,
        bands=info["bands"],
        dtype=info["original_dtype"],
        width=info["width"],
        height=info["height"],
        crs=crs,
        transform=transform_tuple,
        gsd_m=gsd_m,
        date=date,
        nodata=nodata,
        warnings=warnings,
    )
