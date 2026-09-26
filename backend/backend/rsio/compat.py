"""Input compatibility check: decide what kind of request a set of rasters is,
and whether the analysis pipeline can actually run on them.

Owner: P4.  This is the "compatibility_check" step of the execution trace
(CLAUDE.md's agentic-orchestration requirement) -- it runs before any tool is
selected, so a bad pair is rejected with a machine-readable reason instead of
producing a confident-looking answer from mismatched images.

This module never re-projects or resamples; it only detects when that would
be required and says so (`resample_required`), or refuses the pair when it
cannot be made comparable at all (`incompatible_pair`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .raster import RasterInput

SINGLE = "single"
BITEMPORAL = "bitemporal"
OPTICAL_SAR = "optical_sar"


@dataclass
class CompatError:
    code: str
    message: str


@dataclass
class CompatResult:
    input_config: str | None
    accepted: bool
    errors: list[CompatError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


def _bounds(transform: tuple[float, float, float, float, float, float],
            width: int, height: int) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box (left, bottom, right, top) of the raster
    footprint, computed from all four corners so an arbitrary rotation
    (b, d != 0) is still handled correctly."""
    a, b, c, d, e, f = transform
    xs = [c, c + width * a, c + height * b, c + width * a + height * b]
    ys = [f, f + width * d, f + height * e, f + width * d + height * e]
    return min(xs), min(ys), max(xs), max(ys)


def _overlaps(box_a: tuple[float, float, float, float],
              box_b: tuple[float, float, float, float]) -> bool:
    left_a, bottom_a, right_a, top_a = box_a
    left_b, bottom_b, right_b, top_b = box_b
    return left_a < right_b and left_b < right_a and bottom_a < top_b and bottom_b < top_a


def _classify_pair(a: RasterInput, b: RasterInput) -> tuple[str, list[str]]:
    warnings: list[str] = []
    modalities = {a.modality, b.modality}

    if "unknown" in modalities:
        warnings.append(
            "at least one input has unknown modality; classification may be unreliable"
        )

    if modalities == {"optical", "sar"}:
        return OPTICAL_SAR, warnings
    return BITEMPORAL, warnings


def check_inputs(inputs: list[RasterInput | None]) -> CompatResult:
    """Classify a set of loaded inputs and check whether they can be compared.

    `None` entries represent a file the caller tried and failed to load
    (`load_raster` raised) -- passed through here rather than swallowed, so
    the failure becomes a structured `unreadable_file` error instead of a
    crash or a silently short list.
    """
    steps = [f"received {len(inputs)} input(s)"]

    unreadable = [i for i, r in enumerate(inputs) if r is None]
    if unreadable:
        errors = [CompatError("unreadable_file", f"input at index {i} could not be read")
                  for i in unreadable]
        steps.append(f"{len(unreadable)} input(s) unreadable")
        return CompatResult(None, False, errors, [], steps)

    if len(inputs) == 0:
        steps.append("no inputs supplied")
        return CompatResult(None, False,
                             [CompatError("no_inputs", "at least one input is required")],
                             [], steps)

    if len(inputs) > 2:
        steps.append("rejected: more than two inputs")
        return CompatResult(
            None, False,
            [CompatError("too_many_inputs", f"expected 1 or 2 inputs, got {len(inputs)}")],
            [], steps,
        )

    if len(inputs) == 1:
        steps.append("single input -> input_config=single")
        return CompatResult(SINGLE, True, [], [], steps)

    a, b = inputs
    steps.append(f"modalities: {a.modality!r}, {b.modality!r}")

    input_config, warnings = _classify_pair(a, b)
    steps.append(f"classified as {input_config}")

    errors: list[CompatError] = []

    if a.crs is not None and b.crs is not None:
        steps.append("checking CRS match")
        if a.crs != b.crs:
            errors.append(CompatError(
                "incompatible_pair", f"CRS mismatch: {a.crs} vs {b.crs}"
            ))
        elif a.transform is not None and b.transform is not None:
            steps.append("checking bounding-box overlap")
            box_a = _bounds(a.transform, a.width, a.height)
            box_b = _bounds(b.transform, b.width, b.height)
            if not _overlaps(box_a, box_b):
                errors.append(CompatError(
                    "incompatible_pair", "bounding boxes do not overlap"
                ))
        else:
            warnings.append(
                "CRS matches but at least one input has no geotransform; "
                "overlap could not be verified"
            )
    else:
        warnings.append(
            "at least one input has no CRS; geographic compatibility could not be verified"
        )

    if (a.width, a.height) != (b.width, b.height):
        warnings.append("resample_required")
        steps.append("size mismatch flagged: resample_required")

    accepted = not errors
    steps.append("accepted" if accepted else "rejected")
    return CompatResult(input_config, accepted, errors, warnings, steps)
