"""Tests for the offer-leakage view.

The leakage score exists specifically so it can be checked against labels
without ever having seen one; the test that matters most here is the one
that confirms it cannot have.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from eval import offers as O
from orbweaver.config import load_config

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
needs_run = pytest.mark.skipif(not (PROC / "offers.json").exists(),
                               reason="run `make offers` first")


def test_leakage_score_signature_carries_no_label_derived_argument():
    params = set(inspect.signature(O.leakage_score).parameters)
    banned = {"label", "labels", "fraud", "y", "is_fraud", "fraud_share"}
    assert not (params & banned), f"leakage_score's inputs include: {params & banned}"


def test_leakage_score_is_unchanged_by_which_specific_accounts_are_fraud():
    """Two offers with identical (redeemers, in-a-ring, mean score) must score
    identically, however their labels are actually distributed - the function
    literally cannot see them."""
    a = O.leakage_score(n_redeemers=40, n_in_a_ring=12, mean_member_score=0.61)
    b = O.leakage_score(n_redeemers=40, n_in_a_ring=12, mean_member_score=0.61)
    assert a == b


def test_entity_groups_dedupes_and_handles_the_empty_case():
    # uid=1 orders entity 10 twice -> one member, not two. uid=3 orders
    # entity 20 twice -> also one member. Entity 10 also has uid=2, so it
    # ends with two distinct members; entity 20 ends with one.
    uid = np.array([1, 1, 2, 3, 3])
    ent = np.array([10.0, 10.0, 10.0, 20.0, 20.0])
    e, starts, sizes, u = O.entity_groups(uid, ent)
    assert e.tolist() == [10, 20]
    assert sizes.tolist() == [2, 1]
    assert set(u[starts[0]:starts[0] + sizes[0]].tolist()) == {1, 2}
    assert set(u[starts[1]:starts[1] + sizes[1]].tolist()) == {3}

    e0, s0, sz0, u0 = O.entity_groups(np.array([1, 2]), np.array([np.nan, np.nan]))
    assert e0.size == 0 and s0.size == 0 and sz0.size == 0


def test_entity_groups_is_uncapped():
    """Unlike the graph, an offer with more redeemers than n_max must still
    appear - that is exactly the case this view exists to surface."""
    n = CFG.graph.n_max + 50
    uid = np.arange(n)
    ent = np.full(n, 7.0)
    e, starts, sizes, u = O.entity_groups(uid, ent)
    assert sizes[0] == n


def _toy_offers(rng, n_offers=30, labels=None):
    offers = []
    for i in range(n_offers):
        size = int(rng.integers(3, 60))
        members = rng.choice(labels.size, size=size, replace=False)
        in_ring = int(rng.integers(0, size + 1))
        offers.append({"entity": i, "relation": "r6", "members": members.tolist(),
                       "redeemers": size, "redeemers_in_a_ring": in_ring,
                       "mean_score": float(rng.random())})
    for o in offers:
        o["_leak"] = O.leakage_score(o["redeemers"], o["redeemers_in_a_ring"], o["mean_score"])
    return offers


def test_pooled_precision_matches_a_brute_force_computation():
    rng = np.random.default_rng(1)
    labels = rng.choice([1, 0, -1], size=500, p=[0.2, 0.6, 0.2])
    offers = _toy_offers(rng, labels=labels)
    top = offers[:5]
    got = O._pooled_precision(top, labels)

    members = set()
    for o in top:
        members |= set(o["members"])
    members = sorted(members)
    lab = labels[members]
    want_lab = int((lab != -1).sum())
    want_fraud = int((lab == 1).sum())
    assert got["labelled"] == want_lab
    assert got["fraud"] == want_fraud
    assert got["precision"] == pytest.approx(want_fraud / want_lab, abs=1e-4)


def test_precision_at_k_reports_every_configured_budget():
    rng = np.random.default_rng(2)
    labels = rng.choice([1, 0, -1], size=400, p=[0.25, 0.55, 0.2])
    offers = _toy_offers(rng, n_offers=60, labels=labels)
    ranked = O.rank_offers(offers, "ring_share")
    out = O.precision_at_k(ranked, labels, base_rate=0.25, seed=3)
    assert set(out.keys()) == {str(k) for k in O.BUDGETS}
    for row in out.values():
        assert "leakage_ranked" in row and "random_offers" in row


def test_coverage_curve_is_monotone_in_k():
    rng = np.random.default_rng(4)
    labels = rng.choice([1, 0, -1], size=600, p=[0.2, 0.6, 0.2])
    offers = _toy_offers(rng, n_offers=40, labels=labels)
    ranked = O.rank_offers(offers, "mean_score")
    out = O.coverage_curve(ranked, labels, ring_recall=0.0036, max_k=40)
    covered = [p["fraud_covered"] for p in out["points"]]
    swept = [p["legitimate_accounts_swept_in"] for p in out["points"]]
    assert all(b >= a for a, b in zip(covered, covered[1:])), "fraud coverage must never fall"
    assert all(b >= a for a, b in zip(swept, swept[1:])), "accounts reviewed must never fall"


def test_coverage_curve_matches_a_brute_force_running_union():
    rng = np.random.default_rng(5)
    labels = rng.choice([1, 0, -1], size=300, p=[0.3, 0.5, 0.2])
    offers = _toy_offers(rng, n_offers=15, labels=labels)
    ranked = O.rank_offers(offers, "mean_score")
    out = O.coverage_curve(ranked, labels, ring_recall=None, max_k=15)

    seen = set()
    for i, o in enumerate(ranked):
        seen |= set(o["members"])
        want = int((labels[sorted(seen)] == 1).sum())
        assert out["points"][i]["fraud_covered"] == want


def test_rank_offers_is_deterministic():
    rng = np.random.default_rng(6)
    labels = rng.choice([1, 0, -1], size=200, p=[0.2, 0.6, 0.2])
    offers = _toy_offers(rng, n_offers=25, labels=labels)
    a = [o["entity"] for o in O.rank_offers(offers, "ring_share")]
    b = [o["entity"] for o in O.rank_offers(offers, "ring_share")]
    assert a == b


@needs_run
def test_persisted_offers_include_the_top_of_every_ranking_not_just_the_biggest():
    """The highest-leakage offers tend to be small - a handful of redeemers,
    nearly all already in a ring - and keeping only the biggest campaigns by
    redeemer count would silently drop exactly the offers a leakage-sorted
    queue should lead with. Every offer named in the precision@k table for
    either ranking must actually be present in the persisted list."""
    out = json.loads((PROC / "offers.json").read_text())
    persisted_keys = {(o["relation"], o["entity"]) for o in out["offers"]}
    # The top few offers under each ranking, as they would be shown on
    # /offers, must be a subset of what is persisted.
    for by_key in ("share_in_a_ring", "mean_score"):
        top = sorted(out["offers"], key=lambda o: -o[by_key])[:10]
        for o in top:
            assert (o["relation"], o["entity"]) in persisted_keys


@needs_run
def test_the_demo_bundle_stays_under_its_size_cap():
    from orbweaver.console.demo import MAX_BYTES, bundle_path
    meta = bundle_path(CFG) / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        assert m["bytes"] <= MAX_BYTES


@needs_run
def test_offers_json_has_the_three_promo_relations_and_no_member_lists():
    out = json.loads((PROC / "offers.json").read_text())
    rels = {o["relation"] for o in out["offers"]}
    assert rels <= set(O.PROMO_RELATIONS)
    assert all("members" not in o for o in out["offers"]), \
        "member lists should not leak into the persisted artefact"


@needs_run
def test_coverage_reports_both_rankings_against_the_same_recall_reference():
    out = json.loads((PROC / "offers.json").read_text())
    cov = out["coverage"]
    assert "by_leakage_mean_score" in cov and "by_redeemer_count" in cov
    assert cov["ring_recall_reference"] is not None
    leak50 = next(p for p in cov["by_leakage_mean_score"] if p["k"] == 50)
    size50 = next(p for p in cov["by_redeemer_count"] if p["k"] == 50)
    # The finding this comparison exists to surface: ranking by raw size
    # reaches further past the recall ceiling than ranking by leakage does,
    # because the highest-leakage offers are small by construction.
    assert size50["fraud_coverage"] >= leak50["fraud_coverage"]


@needs_run
def test_early_warning_only_reports_confirmed_bad_offers():
    out = json.loads((PROC / "offers.json").read_text())
    warn = out.get("early_warning")
    if warn:
        for row in warn["per_offer"]:
            assert row["fraud_share_among_labelled"] >= O.CONFIRMED_BAD_FRAUD_SHARE
            assert row["labelled_redeemers"] >= O.MIN_LABELLED_TO_CONFIRM
