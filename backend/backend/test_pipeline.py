#!/usr/bin/env python3
"""End-to-end pipeline tests.  Owner: P1.

Checks the contract, not the models: every response has the right shape, the
computed/generated split holds, and nothing crashes on bad input.

These stay valid after P4 and P2 replace the stubs -- that is the point of
writing them against the contract instead of against implementations.

    python backend/test_pipeline.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import answer_query  # noqa: E402

REQUIRED_TOP = {"intent", "confidence", "tool", "answer", "computed", "overlay", "meta"}
VALID_INTENTS = {"describe", "vegetation", "water", "change", "locate", "unclear"}
VALID_OVERLAYS = {"mask", "boxes", "none"}

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


@pytest.mark.xfail(reason="stub ids; pipeline superseded by orchestrator in T4")
def test_shape():
    """Every response carries every contract field, whatever the intent."""
    for q, n in [("describe this", 1), ("vegetation change", 2),
                 ("where are buildings", 1), ("asdfgh", 1), ("", 1)]:
        r = answer_query(q, [f"i{k}" for k in range(n)])
        check(REQUIRED_TOP <= set(r), f"{q!r}: missing {REQUIRED_TOP - set(r)}")
        check(r["intent"] in VALID_INTENTS, f"{q!r}: bad intent {r['intent']}")
        check(r["overlay"]["type"] in VALID_OVERLAYS,
              f"{q!r}: bad overlay {r['overlay']['type']}")
        check(isinstance(r["answer"], str) and r["answer"].strip(),
              f"{q!r}: empty answer")
        check(0.0 <= r["confidence"] <= 1.0,
              f"{q!r}: confidence out of range {r['confidence']}")


@pytest.mark.xfail(reason="stub ids; pipeline superseded by orchestrator in T4")
def test_computed_generated_split():
    """The project's central claim, asserted as a test.

    Exactly one intent may produce generated text. Every intent that reports a
    number must have that number in `computed`, not invented in prose.
    """
    r = answer_query("what is visible here", ["i0"])
    check(r["meta"]["generated_text"] is True, "describe should be generated")
    check(r["computed"] == {}, "describe must not carry computed values")

    for q, n in [("has vegetation decreased", 2), ("did the lake shrink", 2),
                 ("what changed", 2), ("where are the buildings", 1)]:
        r = answer_query(q, [f"i{k}" for k in range(n)])
        check(r["meta"]["generated_text"] is False,
              f"{q!r}: numeric intent must not be generated")
        check(r["computed"], f"{q!r}: numeric intent must return computed values")


@pytest.mark.xfail(reason="stub ids; pipeline superseded by orchestrator in T4")
def test_numbers_appear_verbatim():
    """A computed number must reach the sentence unaltered.

    This is what stops a future refactor from quietly routing numbers through
    a language model.
    """
    r = answer_query("has vegetation decreased", ["a", "b"])
    pct = abs(r["computed"]["pct_change"])
    check(f"{pct:.1f}" in r["answer"],
          f"computed pct {pct} not found verbatim in: {r['answer']}")

    r = answer_query("where are the buildings", ["a"])
    check(str(r["computed"]["count"]) in r["answer"],
          f"computed count missing from: {r['answer']}")


def test_abstention():
    """Out-of-scope and underspecified queries abstain rather than guess."""
    for q, n in [("what will this look like in 2030", 2),
                 ("show me hyperspectral bands", 1),
                 ("what is the population here", 1),
                 ("", 1), ("   ", 1), ("hello", 1)]:
        r = answer_query(q, [f"i{k}" for k in range(n)])
        check(r["intent"] == "unclear", f"{q!r}: should abstain, got {r['intent']}")
        check(r["tool"] is None, f"{q!r}: abstention must not select a tool")
        check(r["overlay"]["type"] == "none", f"{q!r}: abstention has no overlay")


def test_image_count_guards():
    """Change needs two images; asking with one must not fabricate an answer."""
    r = answer_query("what changed between these", ["only_one"])
    check(r["intent"] == "unclear", "change with 1 image must abstain")
    check("two images" in r["answer"].lower(),
          f"should say what is missing: {r['answer']}")

    r = answer_query("what changed between these", ["a", "b"])
    check(r["intent"] == "change", "change with 2 images should route")


def test_no_images():
    r = answer_query("describe this", [])
    check(r["intent"] == "unclear", "no images must abstain")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        before = len(failures)
        try:
            t()
        except Exception as e:                        # noqa: BLE001
            failures.append(f"{t.__name__} raised {type(e).__name__}: {e}")
        mark = "ok  " if len(failures) == before else "FAIL"
        print(f"{mark} {t.__name__}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
