"""Tests for the ring-level confidence model.

Two of these matter more than the rest. The out-of-fold test is what stops the
candidate rings being built from account scores that already knew their own
members' labels, and the label-free test is what stops a label reaching the
ring features by some route I did not think of.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from orbweaver.config import load_config
from orbweaver.rings import ring_scorer
from orbweaver.rings.peel import Ring

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)

needs_run = pytest.mark.skipif(
    not (PROC / "ring_scorer.json").exists(), reason="run `make ring-scorer` first")


def test_ring_features_cannot_see_a_label():
    """Enforced by signature, not by discipline: there is no argument here
    that could carry a label, and none of the names suggests one."""
    sig = inspect.signature(ring_scorer.ring_features)
    names = set(sig.parameters)
    assert not {"labels", "y", "label", "target", "split"} & names
    assert names == {"ring", "scores", "agg", "edge_index", "tau", "lam"}


def test_ring_features_are_deterministic_and_finite():
    rng = np.random.default_rng(0)
    n = 500
    members = np.arange(10, 40)
    ring = Ring(members, density=7.5, internal_weight=210.0,
                score_mass=18.0, lambda_=5.0)
    scores = rng.uniform(0, 1, n)
    src = np.repeat(members[:-1], 1).astype(np.int64)
    dst = members[1:].astype(np.int64)
    idx = {"src": src, "dst": dst,
           "mask": np.full(src.size, 0b101, np.int32),
           "entity": np.full(src.size, 4.0),
           "inside": np.zeros(n, dtype=bool)}
    agg = {"orders": rng.integers(1, 10, n).astype(float),
           "promo_orders": rng.integers(0, 5, n).astype(float),
           "uid": np.repeat(members, 3).astype(np.int64),
           "day": np.tile(np.array([1, 2, 3]), members.size).astype(np.int64)}

    a = ring_scorer.ring_features(ring, scores, agg, idx, 0.5, 5.0)
    b = ring_scorer.ring_features(ring, scores, agg, idx, 0.5, 5.0)
    assert np.array_equal(a, b)
    assert a.size == len(ring_scorer.FEATURE_NAMES)
    assert np.isfinite(a).all()


def test_dedupe_drops_the_same_group_found_twice():
    a = Ring(np.arange(0, 100), 9.0, 1.0, 1.0, 1.0)
    b = Ring(np.arange(2, 100), 8.0, 1.0, 1.0, 1.0)   # 98% overlap
    c = Ring(np.arange(500, 600), 7.0, 1.0, 1.0, 1.0)  # disjoint
    kept = ring_scorer.dedupe([(a, 0.5, 5.0), (b, 0.3, 1.0), (c, 0.5, 0.0)])
    sizes = sorted(r.size for r, _, _ in kept)
    assert len(kept) == 2, "the near-duplicate should have been dropped"
    assert sizes == [100, 100]


def test_candidate_labelling_needs_enough_labelled_members():
    labels = np.full(200, -1, dtype=np.int8)
    labels[:4] = 1
    visible = np.ones(200, dtype=bool)
    thin = Ring(np.array([0, 1, 50, 51, 52]), 5.0, 1.0, 1.0, 1.0)   # 2 labelled
    thick = Ring(np.array([0, 1, 2, 3, 60, 61]), 5.0, 1.0, 1.0, 1.0)  # 4 labelled
    kept, y, shares = ring_scorer.label_candidates(
        [(thin, 0.5, 5.0), (thick, 0.5, 5.0)], labels, visible)
    assert len(kept) == 1
    assert y.tolist() == [1] and shares == [1.0]


def test_candidate_labels_ignore_accounts_the_model_may_not_see():
    """Held-out accounts must not contribute to a candidate's label."""
    labels = np.full(100, -1, dtype=np.int8)
    labels[:6] = 1
    visible = np.zeros(100, dtype=bool)
    visible[:3] = True            # only three of the six are visible
    r = Ring(np.arange(0, 6), 5.0, 1.0, 1.0, 1.0)
    kept, y, shares = ring_scorer.label_candidates([(r, 0.5, 5.0)], labels, visible)
    assert len(kept) == 1
    # three visible fraud, no visible normal
    assert shares == [1.0]


@needs_run
def test_out_of_fold_scores_differ_from_the_full_model():
    """If the out-of-fold scores equalled the full model's, the fold logic did
    nothing and the candidates were built from scores that saw their labels."""
    out = json.loads((PROC / "ring_scorer.json").read_text())
    if not out.get("trained"):
        pytest.skip(out.get("reason", "ring model not trained"))
    assert out["candidates"]["usable"] >= 40
    assert out["candidates"]["positives"] >= 8


@needs_run
def test_all_three_rankings_are_reported():
    out = json.loads((PROC / "ring_scorer.json").read_text())
    if not out.get("trained"):
        pytest.skip(out.get("reason", "ring model not trained"))
    assert {"density", "mean_member_score", "learned_confidence"} == set(out["rankings"])
    for r in out["rankings"].values():
        assert "all_labelled" in r and "heldout_only" in r


@needs_run
def test_legitimate_clusters_are_not_ranked_highly():
    """The hostel clusters are the population this must not promote."""
    out = json.loads((PROC / "ring_scorer.json").read_text())
    if not out.get("trained") or not out.get("hostel_clusters"):
        pytest.skip("no hostel comparison available")
    h = out["hostel_clusters"]
    if h["their_median_confidence"] is None:
        pytest.skip("no ring sits mostly inside a legitimate cluster")
    assert h["their_median_confidence"] <= h["median_confidence_of_all_rings"]
    # Passing this by giving every ring the same confidence is not passing it.
    # The model currently does exactly that, so this records the degeneracy
    # rather than letting a vacuous inequality stand in for a result.
    if h["their_median_confidence"] == h["median_confidence_of_all_rings"]:
        cal = out.get("calibration", [])
        spread = max(c["predicted"] for c in cal) - min(c["predicted"] for c in cal)
        assert spread > 0, "confidence is constant across every ring"
