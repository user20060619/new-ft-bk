#!/usr/bin/env python3
"""route() + _apply_overrides() coverage for the five representative queries
from the official problem statement (CLAUDE.md's graded requirements), plus a
few natural optical_sar phrasings.  Owner: P4.

Read-only: only calls route() and _apply_overrides(); does not touch
router/intent.py or rules.yaml.

    python -m pytest backend/test_problem_statement_queries.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.orchestrator import _apply_overrides  # noqa: E402
from backend.router.intent import route  # noqa: E402

# (query, input_config, n_images) -- input_config is whatever check_inputs
# would have produced for that scenario; n_images is how many files that
# implies for route()'s own n_images-dependent scoring.
REPRESENTATIVE_QUERIES = [
    ("Describe the land-cover and major objects visible in this image.", "single", 1),
    ("Highlight the water body referred to in the query.", "single", 1),
    ("What changed between these two dates, and where did the change occur?", "bitemporal", 2),
    ("Use the optical and SAR images together to identify built-up and water-covered regions.",
     "optical_sar", 2),
    ("Has the built-up area increased, decreased, or remained unchanged?", "bitemporal", 2),
]

OPTICAL_SAR_VARIANTS = [
    "find water and buildings using both images",
    "combine optical and radar",
]


def _resolve(query: str, input_config: str, n_images: int):
    decision = route(query, n_images=n_images)
    final_intent, override_note = _apply_overrides(input_config, decision.intent, decision.reason)
    return decision, final_intent, override_note


@pytest.mark.parametrize("query,input_config,n_images", REPRESENTATIVE_QUERIES)
def test_representative_query_does_not_abstain(query, input_config, n_images):
    decision, final_intent, override_note = _resolve(query, input_config, n_images)
    assert final_intent != "unclear", (
        f"{query!r} (input_config={input_config}) resolved to 'unclear' -- "
        f"router said intent={decision.intent!r} confidence={decision.confidence} "
        f"reason={decision.reason!r}; override_note={override_note!r}"
    )


@pytest.mark.parametrize("query", OPTICAL_SAR_VARIANTS)
def test_optical_sar_variant_does_not_abstain(query):
    decision, final_intent, override_note = _resolve(query, "optical_sar", 2)
    assert final_intent != "unclear", (
        f"{query!r} (input_config=optical_sar) resolved to 'unclear' -- "
        f"router said intent={decision.intent!r} confidence={decision.confidence} "
        f"reason={decision.reason!r}; override_note={override_note!r}"
    )


def test_router_out_of_scope_reason_prefix_is_stable():
    """_apply_overrides distinguishes a genuine out-of-scope rejection from
    every other abstention reason by checking that `decision.reason` starts
    with this exact literal prefix -- router/intent.py's own wording, which
    this file cannot edit to guarantee. If P1 ever changes it, this test
    must fail loudly here instead of _apply_overrides silently losing the
    "out of scope still wins" guarantee for optical_sar/bitemporal pairs."""
    decision = route("what will this look like in 2030", n_images=2)
    assert decision.intent == "unclear"
    assert decision.reason.startswith("out of scope:")


@pytest.mark.parametrize("input_config", ["optical_sar", "bitemporal"])
def test_out_of_scope_query_still_abstains_for_optical_sar_and_bitemporal(input_config):
    """The new "default instead of abstain" behaviour must not swallow a
    genuinely out-of-scope query -- this is the regression the "out of
    scope still wins" guard exists to prevent."""
    decision, final_intent, override_note = _resolve("what will this look like in 2030", input_config, 2)
    assert decision.reason.startswith("out of scope:")
    assert final_intent == "unclear"
    assert override_note is None
