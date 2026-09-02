"""Planted-ring tests for the extractor.

Written before the extractor was run on real data. If peeling cannot recover
rings we planted ourselves in a graph we control, no number it produces on
PPA means anything.

The planted graphs mimic the real structure: a sparse background of
common-entity edges (low rarity weight) with dense cliques of rare-entity
edges (high rarity weight) hidden inside it.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbweaver.rings.peel import build_csr, extract_rings, peel_once

SEED = 20260902


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def planted_graph(n_background: int = 5_000, ring_sizes=(8, 20, 40),
                  bg_degree: int = 4, bg_weight: float = 0.10,
                  ring_weight: float = 0.95, seed: int = SEED):
    """Sparse low-weight background + dense high-weight planted cliques."""
    rng = np.random.default_rng(seed)
    n_ring = sum(ring_sizes)
    n = n_background + n_ring

    # Background: random pairs, common entities => low rarity weight.
    m = n_background * bg_degree // 2
    bs = rng.integers(0, n, size=m)
    bd = rng.integers(0, n, size=m)
    ok = bs != bd
    bs, bd = bs[ok], bd[ok]
    bw = np.full(bs.size, bg_weight)

    # Planted rings occupy the last n_ring ids, contiguously.
    rings, cur = [], n_background
    rs, rd = [], []
    for size in ring_sizes:
        members = np.arange(cur, cur + size)
        rings.append(members)
        i, j = np.triu_indices(size, k=1)
        rs.append(members[i]); rd.append(members[j])
        cur += size
    rs = np.concatenate(rs); rd = np.concatenate(rd)
    rw = np.full(rs.size, ring_weight)

    src = np.concatenate([bs, rs]); dst = np.concatenate([bd, rd])
    w = np.concatenate([bw, rw])
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = lo.astype(np.int64) * n + hi
    _, first = np.unique(key, return_index=True)
    csr = build_csr(lo[first].astype(np.int64), hi[first].astype(np.int64), w[first], n)

    # Scores: ring members suspicious, background not. Deliberately noisy.
    scores = rng.uniform(0.0, 0.3, size=n)
    for members in rings:
        scores[members] = rng.uniform(0.6, 0.95, size=members.size)
    return csr, scores, rings, n


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_planted_rings_recovered(lam):
    """Planted rings must come back with Jaccard >= 0.9 at lambda 0 and 1."""
    csr, scores, rings, _ = planted_graph()
    got = extract_rings(csr, scores, lambda_=lam, k_min=5, top_k=3)
    assert len(got) == 3, f"expected 3 rings, got {len(got)}"

    for planted in rings:
        best = max(jaccard(planted, r.members) for r in got)
        assert best >= 0.9, (
            f"planted ring of size {planted.size} best Jaccard {best:.3f} at lambda={lam}"
        )


def test_lambda_zero_uses_no_model_scores():
    """At lambda=0 the extractor must be a pure function of the graph. If
    permuting the scores can move the answer, then a learned model has leaked
    into the decision path, which is exactly what I am avoiding."""
    csr, scores, _, n = planted_graph()
    rng = np.random.default_rng(7)
    a = extract_rings(csr, scores, lambda_=0.0, k_min=5, top_k=3)
    b = extract_rings(csr, rng.permutation(scores), lambda_=0.0, k_min=5, top_k=3)
    for ra, rb in zip(a, b):
        assert np.array_equal(ra.members, rb.members)
        assert ra.density == pytest.approx(rb.density)


def test_densest_ring_is_found_first():
    """Rings come out in descending density: the 40-clique before the 8."""
    csr, scores, rings, _ = planted_graph()
    got = extract_rings(csr, scores, lambda_=0.0, k_min=5, top_k=3)
    assert [r.size for r in got] == [40, 20, 8]
    assert got[0].density > got[1].density > got[2].density


def test_reported_density_matches_definition():
    """g(S) must equal (internal weight + lambda * score mass) / |S| exactly."""
    csr, scores, _, _ = planted_graph()
    for lam in (0.0, 1.0, 5.0):
        for r in extract_rings(csr, scores, lambda_=lam, k_min=5, top_k=2):
            expect = (r.internal_weight + lam * r.score_mass) / r.size
            assert r.density == pytest.approx(expect, rel=1e-9)


def test_k_min_is_respected():
    csr, scores, _, _ = planted_graph()
    for r in extract_rings(csr, scores, lambda_=1.0, k_min=25, top_k=3):
        assert r.size >= 25


def test_camouflage_does_not_break_extraction():
    """FRAUDAR's premise: a fraudster adds edges to random ordinary users as
    camouflage. Rarity weighting makes those edges cheap, so the ring should
    survive. This is why w_r(e) = 1/log(2+|users(e)|) exists."""
    csr, scores, rings, n = planted_graph()
    rng = np.random.default_rng(11)
    target = rings[-1]                              # the 40-clique
    # 20 camouflage edges per ring member, to random background users,
    # through very common entities => low weight.
    cs = np.repeat(target, 20)
    cd = rng.integers(0, 5_000, size=cs.size)
    cw = np.full(cs.size, 0.08)

    src = np.concatenate([csr.indices[:0], cs]); dst = np.concatenate([csr.indices[:0], cd])
    # rebuild with the original edges plus camouflage
    orig_u, orig_v, orig_w = [], [], []
    for v in range(csr.n_nodes):
        lo, hi = csr.indptr[v], csr.indptr[v + 1]
        nb, ww = csr.indices[lo:hi], csr.weights[lo:hi]
        m = nb > v
        orig_u.append(np.full(m.sum(), v)); orig_v.append(nb[m]); orig_w.append(ww[m])
    u = np.concatenate(orig_u + [np.minimum(src, dst)])
    v_ = np.concatenate(orig_v + [np.maximum(src, dst)])
    w = np.concatenate(orig_w + [cw])
    csr2 = build_csr(u, v_, w, csr.n_nodes)

    got = extract_rings(csr2, scores, lambda_=0.0, k_min=5, top_k=3)
    best = max(jaccard(target, r.members) for r in got)
    assert best >= 0.9, f"camouflage broke extraction: best Jaccard {best:.3f}"


# --- the batch peeler, used when the graph is too large for exact greedy ----


def planted_edge_list(**kw):
    """Same planted graph, as a flat edge list."""
    from orbweaver.rings.peel import EdgeList
    csr, scores, rings, n = planted_graph(**kw)
    u, v, w = [], [], []
    for a in range(csr.n_nodes):
        lo, hi = csr.indptr[a], csr.indptr[a + 1]
        nb, ww = csr.indices[lo:hi], csr.weights[lo:hi]
        m = nb > a
        u.append(np.full(int(m.sum()), a)); v.append(nb[m]); w.append(ww[m])
    return (EdgeList(np.concatenate(u).astype(np.int64),
                     np.concatenate(v).astype(np.int64),
                     np.concatenate(w), csr.n_nodes),
            scores, rings, csr)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_batch_peeler_recovers_planted_rings(lam):
    from orbweaver.rings.peel import extract_rings_batch
    edges, scores, rings, _ = planted_edge_list()
    got = extract_rings_batch(edges, scores, lambda_=lam, k_min=5, top_k=3)
    assert len(got) == 3
    for planted in rings:
        best = max(jaccard(planted, r.members) for r in got)
        assert best >= 0.9, f"batch peeler missed a ring: Jaccard {best:.3f}"


def test_batch_and_exact_peelers_agree():
    """The batch peeler trades a 2-approximation for 2(1+epsilon). It must
    still find the same rings here, or the speedup is not free."""
    from orbweaver.rings.peel import extract_rings_batch
    edges, scores, rings, csr = planted_edge_list()
    exact = extract_rings(csr, scores, lambda_=1.0, k_min=5, top_k=3)
    batch = extract_rings_batch(edges, scores, lambda_=1.0, k_min=5, top_k=3)
    assert len(exact) == len(batch)
    for a, b in zip(exact, batch):
        assert jaccard(a.members, b.members) >= 0.9
        # Batch density may be slightly lower; it must never be higher by
        # more than float noise, since exact greedy is the stronger bound.
        assert b.density <= a.density * 1.001


def test_batch_peeler_respects_k_min():
    from orbweaver.rings.peel import extract_rings_batch
    edges, scores, _, _ = planted_edge_list()
    for r in extract_rings_batch(edges, scores, lambda_=1.0, k_min=25, top_k=3):
        assert r.size >= 25


def test_k_max_prevents_the_giant_blob():
    """Once the genuinely tight rings are extracted and removed, the densest
    thing left is a big mediocre region, and without an upper size bound the
    extractor happily returns it as the next "ring". On the real week-2 graph
    ranks 1-7 were 51-226 accounts and rank 8 was 10,593.

    Plant one tight clique inside a large loose region and check the second
    ring, which is where the failure actually shows up.
    """
    from orbweaver.rings.peel import EdgeList, extract_rings_batch
    rng = np.random.default_rng(3)
    n_blob, n_tight = 4_000, 40
    n = n_blob + n_tight

    # Large region, density ~9.9: lower than the clique, so it loses at rank 1
    # and surfaces at rank 2 - exactly the real failure.
    m = n_blob * 11
    bs = rng.integers(0, n_blob, size=m)
    bd = rng.integers(0, n_blob, size=m)
    ok = bs != bd
    bs, bd, bw = bs[ok], bd[ok], np.full(int(ok.sum()), 0.9)

    # Small dense clique, density ~19.3.
    tight = np.arange(n_blob, n)
    i, j = np.triu_indices(n_tight, k=1)
    ts, td = tight[i], tight[j]
    tw = np.full(ts.size, 0.99)

    src = np.concatenate([bs, ts]).astype(np.int64)
    dst = np.concatenate([bd, td]).astype(np.int64)
    w = np.concatenate([bw, tw])
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    _, first = np.unique(lo * n + hi, return_index=True)
    edges = EdgeList(lo[first], hi[first], w[first], n)
    scores = np.zeros(n)

    unbounded = extract_rings_batch(edges, scores, lambda_=0.0, k_min=5, top_k=2)
    bounded = extract_rings_batch(edges, scores, lambda_=0.0, k_min=5,
                                  k_max=500, top_k=2)

    # Both must find the real ring first.
    assert jaccard(tight, unbounded[0].members) >= 0.9
    assert jaccard(tight, bounded[0].members) >= 0.9

    # The second ring is where it goes wrong.
    assert unbounded[1].size > 1_000, (
        "fixture no longer reproduces the blob: the unbounded peeler should "
        "return the whole loose region once the clique is removed")
    assert bounded[1].size <= 500
    assert all(r.size <= 500 for r in bounded)


def test_empty_and_tiny_inputs():
    csr = build_csr(np.array([0]), np.array([1]), np.array([0.5]), 2)
    scores = np.array([0.9, 0.9])
    assert peel_once(csr, scores, lambda_=1.0, k_min=5).size == 0
    assert extract_rings(csr, scores, lambda_=1.0, k_min=5, top_k=3) == []
