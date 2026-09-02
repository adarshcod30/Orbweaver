"""Tests for ring context and the per-account lookup.

The horizon test is the one that matters. Context features are the easiest
place in this whole project to leak the future into the past — they join one
window's output onto another window's inputs — and the only thing standing
between that and a flattering number is the assertion that the days really do
not overlap.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbweaver.config import load_config
from orbweaver.features import ring_context

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)

needs_windows = pytest.mark.skipif(
    not (PROC / "edges_week2_late_manifest.json").exists(),
    reason="run `make windows-weighted` first")
needs_run = pytest.mark.skipif(
    not (PROC / "ring_context.json").exists(),
    reason="run `make ring-context` first")


@needs_windows
def test_context_window_strictly_precedes_the_feature_window():
    """Read from the manifests, not from the tag names: `early` and `late` are
    only labels, and the days are the fact."""
    h = ring_context.assert_horizon(CFG, "early", "late")
    assert h["context_days"][1] < h["feature_days"][0]
    assert h["gap_days"] >= 1


@needs_windows
def test_the_horizon_check_actually_refuses_an_overlap():
    """A check that never fails is not a check."""
    with pytest.raises(ValueError):
        ring_context.assert_horizon(CFG, "late", "early")
    with pytest.raises(ValueError):
        ring_context.assert_horizon(CFG, "late", "late")


def test_combining_context_never_reorders_against_the_score():
    """Context may break ties and may lift, but it must not let a low-scoring
    account overtake a high-scoring one by more than the multiplicative arm
    allows - and the lexicographic arm must not reorder at all."""
    score = np.array([0.90, 0.50, 0.10])
    ctx = np.array([0.0, 1.0, 1.0])
    out = ring_context.combine(score, ctx)

    lex = out["score_then_context"]
    assert np.argsort(-lex).tolist() == [0, 1, 2], "tie-break must not reorder"

    mult = out["score_times_context"]
    assert mult[0] == pytest.approx(0.90)      # no context, unchanged
    assert mult[1] == pytest.approx(1.00)      # doubled at most
    assert mult[2] == pytest.approx(0.20)


def test_context_cannot_rescue_a_zero_score():
    score = np.array([0.0, 0.4])
    ctx = np.array([1.0, 0.0])
    out = ring_context.combine(score, ctx)
    assert out["score_times_context"][0] == 0.0
    assert out["score_times_context"][1] == pytest.approx(0.4)


@pytest.mark.skipif(not (PROC / "scores_week2.parquet").exists(),
                    reason="run `make score` first")
def test_check_answers_for_a_member_and_a_non_member():
    from orbweaver.console.check import CheckIndex

    idx = CheckIndex(CFG)
    # an account that is in a surfaced ring, if any ring artefact exists
    member = None
    for name in ("ring_report_deep.json", "ring_report.json"):
        p = PROC / name
        if p.exists():
            for c in json.loads(p.read_text()).get("case_files", []):
                if c.get("members_sample"):
                    member = int(c["members_sample"][0])
                    break
        if member is not None:
            break

    if member is not None:
        r = idx.check(member)
        assert r["known"] and r["in_a_ring"]
        assert r["ring"]["size"] >= CFG.rings.k_min
        assert "assumption" in r          # the rupee caveat travels with it

    # an account outside the id range must be refused, not guessed at
    bad = idx.check(idx.n + 10)
    assert bad["known"] is False


@pytest.mark.skipif(not (PROC / "scores_week2.parquet").exists(),
                    reason="run `make score` first")
def test_check_is_fast_enough_to_sit_in_a_request():
    """If this needs a file read per call it is not servable, and the whole
    argument that a per-transaction system could consume it falls over."""
    import time

    from orbweaver.console.check import CheckIndex

    idx = CheckIndex(CFG)
    rng = np.random.default_rng(0)
    accounts = rng.integers(0, idx.n, size=200)
    t0 = time.perf_counter()
    for a in accounts:
        idx.check(int(a))
    per_call_ms = (time.perf_counter() - t0) * 1000.0 / accounts.size
    assert per_call_ms < 5.0, f"{per_call_ms:.2f} ms per lookup is too slow"


@needs_run
def test_result_reports_both_arms_and_the_baseline():
    out = json.loads((PROC / "ring_context.json").read_text())
    assert "score alone" in out["results"]
    assert out["horizon"]["context_days"][1] < out["horizon"]["feature_days"][0]
    assert any(k.startswith("score_times_context") or
               k.startswith("score_then_context") for k in out["results"])
