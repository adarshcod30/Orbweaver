"""Tests for the label-budget sweep.

Nesting is the property most worth pinning down: a subset built independently
per fraction could easily fail to nest without anyone noticing, since each
individual subset would still look stratified and correctly sized on its own.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from eval import label_budget as LB
from eval.split import Split, make_split
from orbweaver.config import load_config

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
needs_run = pytest.mark.skipif(not (PROC / "label_budget.json").exists(),
                               reason="run `make label-budget` first")


def _toy_split(seed=0, n=4000, fraud_rate=0.25):
    rng = np.random.default_rng(seed)
    labels = np.full(n, -1, dtype=np.int8)
    pool = rng.permutation(n)[: n // 2]
    n_fraud = int(round(fraud_rate * pool.size))
    labels[pool[:n_fraud]] = 1
    labels[pool[n_fraud:]] = 0
    test = np.sort(pool[: pool.size // 5])
    train_pool = np.sort(np.setdiff1d(pool, test))
    return Split(train=train_pool, val=np.empty(0, dtype=np.int64), test=test,
                train_pool=train_pool, labels=labels, n_users_week2=n)


def test_subsets_are_nested_within_a_seed():
    split = _toy_split()
    perm = LB.label_permutation(split.train_pool, split.labels, seed=7)
    prev = None
    for frac in LB.FRACTIONS:
        cur = set(LB.stratified_subset(perm, frac).tolist())
        if prev is not None:
            assert prev <= cur, f"fraction {frac} is not a superset of the previous one"
        prev = cur
    assert prev == set(split.train_pool.tolist()), "the 100% prefix must be the whole pool"


def test_subsets_are_stratified():
    split = _toy_split(fraud_rate=0.3)
    perm = LB.label_permutation(split.train_pool, split.labels, seed=3)
    pool_rate = float((split.labels[split.train_pool] == 1).mean())
    for frac in (0.05, 0.2, 0.5):
        sub = LB.stratified_subset(perm, frac)
        rate = float((split.labels[sub] == 1).mean())
        assert abs(rate - pool_rate) < 0.03, f"fraction {frac}: rate {rate} vs pool {pool_rate}"


def test_subsets_are_deterministic_under_the_seed():
    split = _toy_split()
    perm_a = LB.label_permutation(split.train_pool, split.labels, seed=42)
    perm_b = LB.label_permutation(split.train_pool, split.labels, seed=42)
    for frac in LB.FRACTIONS:
        a = LB.stratified_subset(perm_a, frac)
        b = LB.stratified_subset(perm_b, frac)
        assert np.array_equal(a, b)


def test_different_seeds_give_different_subsets():
    split = _toy_split(n=8000)
    perm_a = LB.label_permutation(split.train_pool, split.labels, seed=1)
    perm_b = LB.label_permutation(split.train_pool, split.labels, seed=2)
    a = LB.stratified_subset(perm_a, 0.1)
    b = LB.stratified_subset(perm_b, 0.1)
    assert not np.array_equal(a, b)


def test_held_out_accounts_never_appear_in_any_subset():
    split = _toy_split()
    for seed in range(3):
        perm = LB.label_permutation(split.train_pool, split.labels, seed=seed)
        for frac in LB.FRACTIONS:
            sub = LB.make_subset_split(CFG, split, perm, frac, seed=seed)
            assert not np.intersect1d(sub.train, split.test).size
            assert not np.intersect1d(sub.val, split.test).size
            assert not np.intersect1d(sub.train_pool, split.test).size


def test_the_hundred_percent_point_reuses_the_real_split_unmodified():
    split = _toy_split()
    perm = LB.label_permutation(split.train_pool, split.labels, seed=0)
    sub = LB.make_subset_split(CFG, split, perm, 1.0, seed=0)
    assert sub is split, "100% must be the identical Split object, not a rebuilt copy"


def test_subset_size_matches_the_requested_fraction():
    split = _toy_split(n=20000)
    perm = LB.label_permutation(split.train_pool, split.labels, seed=5)
    for frac in (0.01, 0.05, 0.2):
        sub = LB.stratified_subset(perm, frac)
        want = frac * split.train_pool.size
        assert abs(sub.size - want) / want < 0.15, f"fraction {frac}: got {sub.size}, wanted ~{want}"


def _points(auprcs, ring_precisions):
    return [{"fraction": f, "labelled_accounts_used": int(1000 * f),
            "auprc": {"mean": a}, "ring_precision": {"mean": r}}
           for f, a, r in zip(LB.FRACTIONS, auprcs, ring_precisions)]


def test_knee_finds_the_first_fraction_beating_the_base_rate():
    pts = _points(auprcs=[0.1] * 8, ring_precisions=[0.05, 0.10, 0.15, 0.30, 0.40, 0.50, 0.6, 0.7])
    knee = LB.find_knee(pts)
    assert knee["beats_base_rate_at"]["fraction"] == LB.FRACTIONS[3]


def test_knee_reports_none_when_the_base_rate_is_never_beaten():
    pts = _points(auprcs=[0.1] * 8, ring_precisions=[0.05] * 8)
    knee = LB.find_knee(pts)
    assert knee["beats_base_rate_at"] is None


def test_knee_finds_diminishing_returns_by_the_stated_increment():
    # jumps of 0.05 four times, then jumps below the 0.01 threshold
    auprcs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.303, 0.305, 0.306]
    pts = _points(auprcs=auprcs, ring_precisions=[0.5] * 8)
    knee = LB.find_knee(pts)
    assert knee["diminishing_returns_after"]["fraction"] == LB.FRACTIONS[5]


def test_a_single_noisy_early_step_is_not_diminishing_returns():
    # the first step is a small, noisy gain (this is exactly what the real
    # sweep's smallest, three-seed fractions produced) but every later step
    # keeps gaining at or above the threshold all the way to the end - there
    # is no plateau anywhere in this curve, so none should be reported.
    auprcs = [0.10, 0.104, 0.13, 0.16, 0.18, 0.20, 0.22, 0.25]
    pts = _points(auprcs=auprcs, ring_precisions=[0.5] * 8)
    knee = LB.find_knee(pts)
    assert knee["diminishing_returns_after"] is None


def test_knee_finds_the_real_plateau_past_an_early_noisy_dip():
    # same small noisy first step, but this time a genuine plateau follows
    # from the 10%-to-20% step onward - the knee should land there, not on
    # the earlier noise.
    auprcs = [0.10, 0.104, 0.15, 0.20, 0.25, 0.252, 0.253, 0.254]
    pts = _points(auprcs=auprcs, ring_precisions=[0.5] * 8)
    knee = LB.find_knee(pts)
    assert knee["diminishing_returns_after"]["fraction"] == LB.FRACTIONS[5]


@needs_run
def test_the_hundred_percent_point_reproduces_todays_numbers_exactly():
    out = json.loads((PROC / "label_budget.json").read_text())
    hundred = out["points"][-1]
    assert hundred["fraction"] == 1.0
    assert hundred["auprc"]["mean"] == pytest.approx(0.3796, abs=1e-4)
    assert hundred["ring_precision"]["mean"] == pytest.approx(0.7292, abs=1e-4)


@needs_run
def test_every_fraction_reports_three_seed_runs():
    out = json.loads((PROC / "label_budget.json").read_text())
    for p in out["points"]:
        assert len(p["runs"]) == LB.N_SEEDS


@needs_run
def test_labelled_accounts_grow_monotonically_with_fraction():
    out = json.loads((PROC / "label_budget.json").read_text())
    counts = [p["labelled_accounts_used"] for p in out["points"]]
    assert all(b >= a for a, b in zip(counts, counts[1:]))
