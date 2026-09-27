#!/usr/bin/env python3
"""T10 regression: locks in the water/vegetation proxy calibration against
the real Mumbai JPEG pair, so the channel-order bug (RGB_PROXY_ORDER used to
mislabel red/blue for real photos, flagging ~99% of Mumbai25.jpg as water) or
an equally bad regression fails loudly here, not just on synthetic fixtures.
Owner: P4.

Deliberately not the "small synthetic raster" style the rest of this test
suite uses (CLAUDE.md rule 7) -- this is specifically a real-image regression
test, testing the *calibration*, not the algorithm's logic (already covered
by test_landcover.py's synthetic fixtures). Skips cleanly if the committed
benchmark images are ever moved/removed, rather than failing for an unrelated
reason.

    python -m pytest backend/services/test_water_proxy_real_image.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.rsio.raster import load_raster  # noqa: E402
from backend.services.landcover import extract_optical  # noqa: E402

_TEST_IMAGES_DIR = Path(__file__).resolve().parents[3] / "test-images"
_MUMBAI_25 = _TEST_IMAGES_DIR / "Mumbai25.jpg"
_MUMBAI_26 = _TEST_IMAGES_DIR / "Mumbai26.jpg"

pytestmark = pytest.mark.skipif(
    not (_MUMBAI_25.exists() and _MUMBAI_26.exists()),
    reason="test-images/Mumbai25.jpg or Mumbai26.jpg not found",
)


@pytest.mark.parametrize("path", [_MUMBAI_25, _MUMBAI_26])
def test_water_and_vegetation_proxy_plausible_on_real_photo(path, tmp_path):
    r = load_raster(path)
    result = extract_optical(r, tmp_path)

    assert result.water_is_true_index is False
    assert result.vegetation_is_true_index is False

    # Wide-but-meaningful bands: loose enough to survive minor recalibration,
    # tight enough to fail loudly if the channel-order bug (water ~99%, or a
    # collapse to ~0%) or an equally bad regression ever comes back.
    assert 1.0 < result.water_percentage < 25.0, (
        f"implausible water_percentage={result.water_percentage} for {path.name}"
    )
    assert 1.0 < result.vegetation_percentage < 35.0, (
        f"implausible vegetation_percentage={result.vegetation_percentage} for {path.name}"
    )
    assert isinstance(result.built_up_percentage, float)


def test_water_proxy_not_degenerate_between_the_two_dates(tmp_path):
    """The two dates shouldn't both collapse to the same degenerate extreme
    (e.g. both ~99% or both ~0%) -- a weak but real cross-check that the
    proxy is responding to actual image content, not a constant."""
    before = load_raster(_MUMBAI_25)
    after = load_raster(_MUMBAI_26)

    result_before = extract_optical(before, tmp_path, prefix="before_")
    result_after = extract_optical(after, tmp_path, prefix="after_")

    for result, name in [(result_before, "Mumbai25"), (result_after, "Mumbai26")]:
        assert result.water_percentage < 50.0, f"{name} water_percentage still implausibly high"
        assert result.water_percentage > 0.5, f"{name} water_percentage implausibly near zero"
