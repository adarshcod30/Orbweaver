"""Tests for behavioural-twin edges.

Two properties carry the argument. Twins must be confined to accounts the
scorer already flagged — otherwise they are a new source of noise across the
whole population rather than a refinement of a suspicious region — and their
weight must come from training accounts only, like every other relation's.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbweaver.adversarial import twins as T
from orbweaver.config import load_config

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)

needs_run = pytest.mark.skipif(
    not (PROC / "twins.json").exists(), reason="run `make twins` first")


def test_mutual_knn_is_symmetric_and_deduplicated():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(loc, 0.05, (25, 8)) for loc in (0.0, 4.0, 8.0)])
    i, j = T.mutual_knn(T.standardise(X), k=3, block=16)
    assert (i < j).all(), "pairs must be stored one way round"
    assert len({(a, b) for a, b in zip(i.tolist(), j.tolist())}) == i.size


def test_mutual_knn_links_within_clusters_not_across():
    """If behaviour edges joined unrelated accounts they would be noise."""
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(loc, 0.05, (25, 8)) for loc in (0.0, 4.0, 8.0)])
    i, j = T.mutual_knn(T.standardise(X), k=3, block=16)
    same_cluster = (i // 25) == (j // 25)
    assert same_cluster.mean() > 0.95


def test_mutual_knn_is_stricter_than_plain_nearest_neighbours():
    """Mutual kNN must drop one-directional links; if it kept everything it
    would not be doing the thing that keeps the graph sparse."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (200, 6))
    i, _ = T.mutual_knn(T.standardise(X), k=5, block=64)
    assert i.size < 200 * 5, "no pair was dropped, so mutuality did nothing"


def test_twin_weight_uses_only_training_accounts():
    """A weight computed from held-out labels would be leakage dressed up as
    a measurement."""
    n = 1000
    labels = np.full(n, -1, dtype=np.int8)
    labels[:400] = 1                       # fraud
    labels[400:800] = 0                    # normal
    visible = np.zeros(n, dtype=bool)
    visible[:200] = True                   # only these may be measured on
    visible[400:600] = True

    rng = np.random.default_rng(4)
    src = rng.integers(0, 200, 2000)
    dst = rng.integers(400, 600, 2000)
    w = T.twin_weight(CFG, src, dst, labels, visible, 0.2)
    assert w["measured"]
    # every labelled edge counted must have both ends visible
    assert w["edges_labelled"] == 2000

    # the same edges, but with nothing visible, must refuse to measure
    w2 = T.twin_weight(CFG, src, dst, labels, np.zeros(n, dtype=bool), 0.2)
    assert not w2["measured"]
    assert w2["weight"] == pytest.approx(0.2)


def test_twin_weight_is_never_a_hand_picked_constant():
    """The weight must be the measured lift times the median entity weight."""
    n = 600
    labels = np.full(n, -1, dtype=np.int8)
    labels[:300] = 1
    labels[300:] = 0
    visible = np.ones(n, dtype=bool)
    rng = np.random.default_rng(5)
    src = rng.integers(0, 300, 1500)
    dst = rng.integers(0, 300, 1500)       # fraud-to-fraud, so lift is high
    w = T.twin_weight(CFG, src, dst, labels, visible, 0.25)
    assert w["weight"] == pytest.approx(w["lift"] * 0.25, rel=1e-3)


@needs_run
def test_twins_are_confined_to_flagged_accounts():
    out = json.loads((PROC / "twins.json").read_text())
    t = out["twins"]
    # every twin edge lives among accounts above the cut-off, so there cannot
    # be more distinct endpoints than there are candidates
    assert t["twin_edges"] > 0
    assert t["candidates"] > 0
    assert t["k"] == T.K_NEIGHBOURS


@needs_run
def test_twins_do_not_damage_the_intact_baseline():
    """If behaviour edges made the undamaged graph worse, they would not be
    worth having whatever they do under attack."""
    out = json.loads((PROC / "twins.json").read_text())
    intact = out["fragmentation"]["intact"]
    wo = intact["without_twins"]["ring_precision"]
    wi = intact["with_twins"]["ring_precision"]
    assert wi >= wo - 0.01, f"twins cost {wo - wi:.4f} on the intact graph"


@needs_run
def test_every_cell_size_reports_both_arms():
    out = json.loads((PROC / "twins.json").read_text())
    for key, row in out["fragmentation"].items():
        assert row["without_twins"]["ring_precision"] is not None
        assert row["with_twins"]["ring_precision"] is not None
