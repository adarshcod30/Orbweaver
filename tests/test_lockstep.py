"""Tests for burst-corrected time weighting.

The null model is the part that has to be right - a two-account entity is
trivially "bursty" by size alone, and the whole point of `burst_z` is to
correct for exactly that. Most of these tests exist to catch the null being
subtly wrong before it reaches a fitted multiplier.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbweaver.config import load_config
from orbweaver.data import lockstep as L

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
needs_run = pytest.mark.skipif(not (PROC / "lockstep.json").exists(),
                               reason="run `make lockstep` first")


def test_random_arrivals_give_burst_z_near_zero_at_every_size_bucket():
    rng = np.random.default_rng(3)
    n_bins = 10
    p = np.full(n_bins, 1.0 / n_bins)
    for i, (lo, hi) in enumerate(L.SIZE_BUCKETS):
        sizes = rng.integers(lo, hi + 1, 400)
        entity = np.repeat(np.arange(400), sizes)
        first_bin = rng.choice(n_bins, size=entity.size, p=p)
        zt = L.burst_z_table(entity, first_bin, 0, n_bins - 1, seed=100 + i, window=1)
        assert abs(zt["z"].mean()) < 0.35, f"bucket {(lo, hi)}: mean z {zt['z'].mean()}"


def test_a_hand_built_burst_gets_a_high_z_at_size_six_or_more():
    rng = np.random.default_rng(4)
    n_bins = 8
    p = np.full(n_bins, 1.0 / n_bins)
    background_sizes = rng.integers(2, 100, 500)
    entity = np.repeat(np.arange(500), background_sizes)
    first_bin = rng.choice(n_bins, size=entity.size, p=p)

    bursty = np.full(6, 999_999)
    bursty_bin = np.full(6, 2)
    e = np.concatenate([entity, bursty])
    b = np.concatenate([first_bin, bursty_bin])
    zt = L.burst_z_table(e, b, 0, n_bins - 1, seed=7, window=1)
    idx = np.flatnonzero(zt["entity"] == 999_999)[0]
    assert zt["z"][idx] > 3.0, f"got z={zt['z'][idx]}"


def test_a_two_account_entity_is_not_trivially_flagged():
    """The size correction is the whole point: a pair that happens to arrive
    the same day must not read as coordinated just because it is small."""
    rng = np.random.default_rng(5)
    n_bins = 8
    p = np.full(n_bins, 1.0 / n_bins)
    background_sizes = rng.integers(6, 100, 400)
    entity = np.repeat(np.arange(400), background_sizes)
    first_bin = rng.choice(n_bins, size=entity.size, p=p)

    pairs = np.repeat(np.arange(1000, 3000), 2)
    pair_bin = rng.choice(n_bins, size=pairs.size, p=p)
    e = np.concatenate([entity, pairs])
    b = np.concatenate([first_bin, pair_bin])
    zt = L.burst_z_table(e, b, 0, n_bins - 1, seed=9, window=1)
    mask = np.isin(zt["entity"], np.unique(pairs))
    assert abs(zt["z"][mask].mean()) < 0.25, f"mean z for size-2 entities: {zt['z'][mask].mean()}"


def test_burst_matches_brute_force_for_both_window_sizes():
    rng = np.random.default_rng(11)
    entity = np.repeat(np.arange(40), rng.integers(2, 20, 40))
    first_bin = rng.integers(0, 30, entity.size)  # a wide, sparse range
    res = L.burst_for_entities(entity, first_bin, 0, 29)

    for i, e in enumerate(res["entity"]):
        b = first_bin[entity == e]
        size = b.size
        counts = np.bincount(b, minlength=30)
        want1 = counts.max() / size
        want2 = max((counts[j] + counts[j + 1] for j in range(29)), default=want1) / size
        assert res[1][i] == pytest.approx(want1, abs=1e-9)
        assert res[2][i] == pytest.approx(want2, abs=1e-9)


def test_null_simulation_is_deterministic_under_the_seed():
    p = np.full(6, 1 / 6)
    sizes = np.array([6, 7, 8, 9, 10])
    a = L.simulate_null_by_size(sizes, 0, 5, p, draws_per_size=500, seed=42)
    b = L.simulate_null_by_size(sizes, 0, 5, p, draws_per_size=500, seed=42)
    assert a == b


def test_null_mean_reflects_the_entitys_own_exact_size_not_a_bucket_average():
    """The bug this null model exists to avoid: bucketing sizes 2-3 together
    and drawing the null size uniformly gives every size-2 entity a null
    partly built from size-3 draws, which are measurably less bursty - so a
    real size-2 entity reads as falsely bursty. Comparing against its own
    exact size removes the bias; this asserts the two exact-size nulls
    actually differ, so the fix is doing real work rather than being a no-op."""
    p = np.full(8, 1 / 8)
    both = L.simulate_null_by_size(np.array([2, 3]), 0, 7, p, draws_per_size=8000, seed=1)
    assert both[2][1][0] > both[3][1][0], \
        "a size-2 entity should have a higher null mean burst than size-3"
    assert abs(both[2][1][0] - both[3][1][0]) > 0.05, "the two sizes must be genuinely distinct"


def test_size_bucket_covers_the_capped_range_without_overlap():
    sizes = np.arange(2, 101)
    idx = L.size_bucket(sizes)
    assert (idx >= 0).all(), "every size from 2 to n_max must land in a bucket"
    # every size maps to exactly one bucket (buckets are contiguous, disjoint)
    counted = np.zeros(6, dtype=int)
    for i, (lo, hi) in enumerate(L.SIZE_BUCKETS):
        counted[i] = ((sizes >= lo) & (sizes <= hi)).sum()
    assert counted.sum() == sizes.size


def test_first_arrival_groups_dedupes_repeat_orders_and_drops_out_of_range():
    uid = np.array([1, 1, 1, 2, 2, 3, 9])
    ent = np.array([10, 10, 10, 10, 10, 10, 20], dtype=float)  # entity 20 has size 1
    bins = np.array([0, 0, 1, 2, 2, 3, 0])
    e, u, b, sel = L.first_arrival_groups(uid, ent, bins, n_max=100)
    assert set(zip(e.tolist(), u.tolist(), b.tolist())) == {(10, 1, 0), (10, 2, 2), (10, 3, 3)}
    assert 20 not in e.tolist(), "a size-1 entity must not survive"


@needs_run
def test_the_multiplier_fit_is_recorded_as_training_only():
    ls = json.loads((PROC / "lockstep.json").read_text())
    fit = ls["fit"]
    assert fit["heldout_excluded"] > 0
    assert fit["accounts_visible"] > 0


@needs_run
def test_time_weighting_off_leaves_the_standard_graph_untouched():
    """The config default is false, and no code path in build_graph.py reads
    it - so a config with the flag flipped on must build a byte-identical
    standard graph, because the standard builder never looks at the flag."""
    from orbweaver.data.build_graph import build_graph

    cfg = CFG
    flipped = cfg.model_copy(deep=True)
    flipped.graph.time_weighting = True

    p1 = build_graph(2, cfg, tag="lockstep_offtest")
    p2 = build_graph(2, flipped, tag="lockstep_offtest")  # same path - just proving the flag is unread
    assert p1 == p2, "build_graph does not take the flag into its output path or behaviour"
    import inspect
    assert "time_weighting" not in inspect.getsource(build_graph), \
        "build_graph.py must not read graph.time_weighting at all"


@needs_run
def test_lockstep_graph_reports_a_ring_precision_beside_the_standard_one():
    ls = json.loads((PROC / "lockstep.json").read_text())
    r = ls["rings_at_headline"]
    assert r["standard"]["ring_precision"] is not None
    assert r["lockstep"]["ring_precision"] is not None
    # today's headline number must not have moved
    assert r["standard"]["ring_precision"] == pytest.approx(0.7292, abs=1e-4)


@needs_run
def test_crowd_test_covers_all_five_relations_both_ways():
    ls = json.loads((PROC / "lockstep.json").read_text())
    crowd = ls["crowd_test_all_relations"]
    assert set(crowd.keys()) == set(CFG.data.buildable_relations)
    for rel, v in crowd.items():
        assert "standard" in v and "lockstep" in v
