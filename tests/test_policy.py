"""Tests for the capacity-aware review policy.

The knapsack is the part worth testing hardest: it is the only place in this
project where an exact combinatorial optimum is claimed, and a subtly wrong
table would quietly return a good-looking but suboptimal review set that no
metric would flag.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from orbweaver.config import load_config
from orbweaver.rings import policy as P

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
needs_run = pytest.mark.skipif(not (PROC / "policy.json").exists(),
                               reason="run `make policy` first")


def brute_force(values, weights, capacity):
    best, chosen = 0.0, []
    for k in range(len(values) + 1):
        for combo in itertools.combinations(range(len(values)), k):
            w = sum(weights[i] for i in combo)
            v = sum(values[i] for i in combo)
            if w <= capacity and v > best + 1e-12:
                best, chosen = v, list(combo)
    return best, chosen


def test_knapsack_matches_brute_force():
    rng = np.random.default_rng(7)
    for trial in range(25):
        n = int(rng.integers(1, 13))
        values = (rng.random(n) * 1000).round(2).tolist()
        weights = rng.integers(1, 15, n).tolist()
        cap = int(rng.integers(1, 40))
        want, _ = brute_force(values, weights, cap)
        got = P.knapsack(values, weights, cap)
        assert sum(weights[i] for i in got) <= cap, "budget exceeded"
        assert sum(values[i] for i in got) == pytest.approx(want, rel=1e-9), \
            f"trial {trial}: knapsack found {sum(values[i] for i in got)}, optimum is {want}"


def test_knapsack_edge_cases():
    assert P.knapsack([], [], 10) == []
    assert P.knapsack([5.0], [3], 0) == []
    assert P.knapsack([5.0], [11], 10) == [], "an item heavier than the budget"
    assert P.knapsack([0.0, 0.0], [1, 1], 5) == [], "worthless items are not taken"
    assert P.knapsack([1.0, 1.0], [3, 3], 6) == [0, 1]


def _ring(p, value, legit, size, density=1.0, fraud_value=None, legit_value=None):
    return {"p": p, "value_at_stake_inr": float(value),
            "legitimate_value_exposed_inr": float(legit),
            "review_minutes": max(1, int(round(3 + 0.25 * size))), "size": size,
            "density": density,
            "realised_fraud_value_inr": float(value if fraud_value is None else fraud_value),
            "realised_legitimate_value_inr": float(legit if legit_value is None else legit_value)}


def test_the_budget_is_never_exceeded():
    rng = np.random.default_rng(11)
    rings = [_ring(float(rng.random()), rng.integers(100, 9000), rng.integers(0, 9000),
                   int(rng.integers(5, 90))) for _ in range(60)]
    for b in P.BUDGETS_MIN:
        rev, _ = P.plan_capacity_aware(rings, b, 0.1)
        assert sum(rings[i]["review_minutes"] for i in rev) <= b
        rev2, _ = P.plan_density_order(rings, b)
        assert sum(rings[i]["review_minutes"] for i in rev2) <= b


def test_doing_nothing_stops_nothing_and_scores_zero_savings():
    rings = [_ring(0.9, 5000, 1000, 20), _ring(0.4, 800, 4000, 10)]
    rev, held = P.plan_do_nothing(rings)
    out = P._outcome(rings, rev, held, churn=0.1, accuracy=1.0, cfg=CFG)
    assert out == {**out, "fraud_value_stopped_inr": 0.0,
                   "legitimate_value_harmed_inr": 0.0, "minutes_used": 0.0}
    total = sum(r["realised_fraud_value_inr"] for r in rings)
    assert P.savings(out, total) == 0.0


def test_expected_value_is_monotone_in_the_budget():
    """More analyst minutes can never make the optimiser worse off. Realised
    rupees need not be monotone - the knapsack maximises expected value and
    the labels are not consulted - so the guarantee is on the expected side."""
    rng = np.random.default_rng(3)
    rings = [_ring(float(rng.random()), rng.integers(100, 9000), rng.integers(0, 9000),
                   int(rng.integers(5, 90))) for _ in range(40)]
    prev = -np.inf
    for b in (0, 30, 60, 120, 240, 480):
        rev, held = P.plan_capacity_aware(rings, b, 0.1)
        ev = sum(P.expected_values(rings[i], 0.1)[0] for i in rev)
        ev += sum(max(P.expected_values(rings[i], 0.1)[1], 0.0) for i in held)
        assert ev >= prev - 1e-9, f"budget {b} scored worse than a smaller one"
        prev = ev


def test_reviewing_is_never_worse_than_holding_the_same_ring():
    for p in (0.0, 0.3, 0.7, 1.0):
        r = _ring(p, 5000, 4000, 30)
        ev_review, ev_hold = P.expected_values(r, 0.25)
        assert ev_review >= ev_hold - 1e-12


def test_churn_only_moves_auto_hold_decisions():
    """Churn is the cost of holding someone wrongly. It must not change which
    rings are worth an analyst's time, only which unreviewed ones get held."""
    rng = np.random.default_rng(5)
    rings = [_ring(float(rng.random()), rng.integers(100, 9000), rng.integers(0, 9000),
                   int(rng.integers(5, 90))) for _ in range(50)]
    holds = []
    for c in P.CHURN:
        rev, held = P.plan_capacity_aware(rings, 120, c)
        holds.append(len(held))
    assert holds == sorted(holds, reverse=True), \
        "more churn should not increase the number of rings auto-held"


def test_higher_churn_never_increases_the_number_held():
    rings = [_ring(0.6, 5000, v, 20) for v in (0, 1000, 5000, 20000, 100000)]
    counts = [len(P.plan_capacity_aware(rings, 0, c)[1]) for c in (0.01, 0.1, 0.5, 0.9)]
    assert counts == sorted(counts, reverse=True)


def test_a_perfect_reviewer_harms_nobody():
    rings = [_ring(0.5, 5000, 9000, 20)]
    out = P._outcome(rings, {0}, set(), churn=0.25, accuracy=1.0, cfg=CFG)
    assert out["legitimate_value_harmed_inr"] == 0.0
    out90 = P._outcome(rings, {0}, set(), churn=0.25, accuracy=0.9, cfg=CFG)
    assert out90["legitimate_value_harmed_inr"] > 0.0
    assert out90["fraud_value_stopped_inr"] < out["fraud_value_stopped_inr"]


@needs_run
def test_decisions_use_no_labels():
    """A ring's recommended action must be derivable from what is known
    tonight. If two rings have the same p, value, exposure and minutes they
    must get the same action, whatever their labels turned out to be."""
    out = json.loads((PROC / "policy.json").read_text())
    seen = {}
    for r in out["recommendations_at_120_minutes"]:
        key = (round(r["expected_net_if_reviewed_inr"], 2),
               round(r["expected_net_if_auto_held_inr"], 2), r["review_minutes"])
        if key in seen:
            assert seen[key] == r["action"], f"same economics, different action: {key}"
        seen[key] = r["action"]


@needs_run
def test_every_budget_reports_all_four_policies_and_respects_itself():
    out = json.loads((PROC / "policy.json").read_text())
    for b, by in out["final_night"]["budgets"].items():
        assert len(by) == 4
        for name, o in by.items():
            assert o["minutes_used"] <= int(b) + 1e-9, f"{name} overspent at {b}"
        assert by["do nothing"]["fraud_value_stopped_inr"] == 0.0


@needs_run
def test_the_capacity_aware_policy_beats_the_baselines_or_says_so():
    out = json.loads((PROC / "policy.json").read_text())
    for b, by in out["final_night"]["budgets"].items():
        mine = by["capacity-aware"]["net_inr"]
        for name in ("density order until the budget is spent", "do nothing"):
            assert mine >= by[name]["net_inr"] - 1e-6, \
                f"at {b} minutes the capacity-aware policy lost to {name}"


@needs_run
def test_assumptions_travel_with_the_numbers():
    out = json.loads((PROC / "policy.json").read_text())
    a = out["assumptions"]
    assert "no monetary amounts" in a["note"]
    assert "not a calibrated ring-level" in a["probability_caveat"]
