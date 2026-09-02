"""Ring extraction: score-weighted densest-subgraph discovery by greedy peeling.

This is the formal core of Orbweaver, and the one component that is
deliberately **not** machine-learned. A model assigns each account a
suspicion score; this algorithm decides who is in a ring. It is
deterministic, inspectable, and carries a proved approximation bound - none
of which a learned ring-assignment could offer to an auditor asking "why is
this account in this ring?"

Objective
---------
For a candidate set S of users::

    g(S) = ( sum_{(u,v) in E[S]} w(u,v)  +  lambda * sum_{v in S} s(v) ) / |S|

- ``w(u,v)`` is the entity-rarity edge weight (see ``data/build_graph.py``):
  structure, i.e. how much rare evidence ties this pair together.
- ``s(v)`` is the calibrated model suspicion score in [0, 1].
- ``lambda`` trades structure against model belief. ``lambda = 0`` is pure
  structure and uses no ML at all; ``lambda -> inf`` is pure model and ignores
  the graph. The sweep over lambda is reported, not tuned away.

Approximation guarantee
-----------------------
``g`` is a non-negative, monotone-supermodular-style density of the form
(total edge weight + total node weight) / |S|. Charikar (2000) proved that
peeling the minimum-contribution vertex and returning the best prefix gives a
**1/2-approximation** to the maximiser of this ratio; Hooi et al. (FRAUDAR,
KDD 2016, Theorem 1) prove the same 1/2 bound for exactly this
edge-weight-plus-node-prior objective. So::

    g(S_returned)  >=  (1/2) * g(S_optimal)

**With the minimum-size constraint ``|S| >= k_min`` the problem becomes
NP-hard** - it is densest-at-least-k-subgraph (DalkS; Khuller & Saha 2009).
Greedy peeling retains a constant-factor bound there (>= 1/2 of optimum) and
reaches >= 0.8 of optimum empirically (Xu, Ma, Fang et al., SIGMOD 2023). We
report the achieved ``g``, never a claim of optimality.

Complexity: ``O((|V| + |E|) log |V|)`` time, ``O(|V| + |E|)`` space, using a
lazy-deletion heap. On the full week-2 graph this is minutes on one CPU core.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Ring:
    """One extracted ring and the quantities that justify it."""
    members: np.ndarray
    density: float                 # g(S) achieved
    internal_weight: float         # sum of w(u,v) inside S
    score_mass: float              # sum of s(v) inside S
    lambda_: float
    rank: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.members.size)


@dataclass
class CSR:
    """Symmetric adjacency in compressed-sparse-row form."""
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    n_nodes: int

    def degree_weight(self) -> np.ndarray:
        """Weighted degree of every node."""
        out = np.zeros(self.n_nodes, dtype=np.float64)
        np.add.reduceat(self.weights, self.indptr[:-1], out=out)
        # reduceat mishandles empty rows; zero them explicitly.
        empty = np.diff(self.indptr) == 0
        out[empty] = 0.0
        return out


def build_csr(src: np.ndarray, dst: np.ndarray, weight: np.ndarray,
              n_nodes: int) -> CSR:
    """Undirected CSR from a deduplicated edge list (each pair appears once)."""
    u = np.concatenate([src, dst]).astype(np.int64)
    v = np.concatenate([dst, src]).astype(np.int64)
    w = np.concatenate([weight, weight]).astype(np.float64)
    order = np.argsort(u, kind="stable")
    u, v, w = u[order], v[order], w[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n_nodes), out=indptr[1:])
    return CSR(indptr=indptr, indices=v, weights=w, n_nodes=n_nodes)


def peel_once(csr: CSR, scores: np.ndarray, *, lambda_: float, k_min: int,
              candidates: np.ndarray | None = None) -> Ring:
    """Greedy peeling. Returns the best prefix S with |S| >= k_min.

    Peels the minimum-contribution node repeatedly and remembers the densest
    set seen. ``candidates`` restricts the starting set S (used both for
    pruning and for extracting the 2nd..Kth ring after earlier ones are
    removed).
    """
    alive = np.zeros(csr.n_nodes, dtype=bool)
    if candidates is None:
        alive[:] = True
    else:
        alive[candidates] = True
    n_alive = int(alive.sum())
    if n_alive < k_min:
        return Ring(np.empty(0, np.int32), -np.inf, 0.0, 0.0, lambda_)

    # contribution c(v) = weighted degree within S + lambda * s(v)
    contrib = np.zeros(csr.n_nodes, dtype=np.float64)
    edge_w = 0.0
    for v in np.flatnonzero(alive):
        lo, hi = csr.indptr[v], csr.indptr[v + 1]
        nbr, w = csr.indices[lo:hi], csr.weights[lo:hi]
        m = alive[nbr]
        d = float(w[m].sum())
        contrib[v] = d
        edge_w += d
    edge_w /= 2.0                        # each edge counted from both ends
    contrib[alive] += lambda_ * scores[alive]
    score_mass = float(scores[alive].sum())

    def g(n: int, ew: float, sm: float) -> float:
        return (ew + lambda_ * sm) / n if n else -np.inf

    best_g = g(n_alive, edge_w, score_mass)
    best_n = n_alive
    removal_order: list[int] = []

    heap = [(contrib[v], int(v)) for v in np.flatnonzero(alive)]
    heapq.heapify(heap)

    while n_alive > k_min:
        c, v = heapq.heappop(heap)
        if not alive[v] or c != contrib[v]:
            continue                      # stale entry (lazy deletion)
        alive[v] = False
        n_alive -= 1
        removal_order.append(v)

        lo, hi = csr.indptr[v], csr.indptr[v + 1]
        nbr, w = csr.indices[lo:hi], csr.weights[lo:hi]
        m = alive[nbr]
        edge_w -= float(w[m].sum())
        score_mass -= float(scores[v])
        for u, wu in zip(nbr[m], w[m]):
            contrib[u] -= wu
            heapq.heappush(heap, (contrib[u], int(u)))

        cur = g(n_alive, edge_w, score_mass)
        if cur > best_g:
            best_g, best_n = cur, n_alive

    # Replay: the best set is everything except the first (start - best_n) removals.
    start = int(alive.sum()) + len(removal_order)
    members = np.ones(csr.n_nodes, dtype=bool)
    keep = np.zeros(csr.n_nodes, dtype=bool)
    if candidates is None:
        keep[:] = True
    else:
        keep[candidates] = True
    for v in removal_order[: start - best_n]:
        keep[v] = False
    members = np.flatnonzero(keep).astype(np.int32)

    ew, sm = _subgraph_totals(csr, members, scores)
    return Ring(members=members, density=(ew + lambda_ * sm) / max(members.size, 1),
                internal_weight=ew, score_mass=sm, lambda_=lambda_)


@dataclass
class EdgeList:
    """Undirected edges as flat arrays. One row per pair, src < dst."""
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    n_nodes: int


def peel_batch(edges: EdgeList, scores: np.ndarray, *, lambda_: float, k_min: int,
               k_max: int | None = None, candidates: np.ndarray | None = None,
               epsilon: float = 0.1, max_passes: int = 2000) -> Ring:
    """Greedy peeling in batches, for graphs too large to peel one node at a time.

    Exact greedy removes a single minimum-contribution vertex per step, which
    is inherently sequential: on 35M edges that is tens of millions of heap
    operations. This removes **every** vertex whose contribution is at or
    below ``(1 + epsilon)`` times the mean in one pass, so each pass is
    vectorised over the edge list and a constant fraction of the graph dies
    per pass. It finishes in ``O(log_{1+epsilon} n)`` passes.

    The bound weakens honestly in exchange: this is a
    **2(1 + epsilon)-approximation** (Bahmani, Kumar & Vassilvitskii, VLDB
    2012), against the ``2``-approximation of ``peel_once``. At the default
    ``epsilon = 0.1`` that is 2.2 rather than 2. ``tests/test_peel_planted.py``
    asserts both routines recover the same planted rings.
    """
    n = edges.n_nodes
    alive = np.zeros(n, dtype=bool)
    if candidates is None:
        alive[:] = True
    else:
        alive[candidates] = True
    if int(alive.sum()) < k_min:
        return Ring(np.empty(0, np.int32), -np.inf, 0.0, 0.0, lambda_)

    src, dst, w = edges.src, edges.dst, edges.weight
    removed_at = np.full(n, -1, dtype=np.int32)
    best_g, best_pass = -np.inf, 0

    for p in range(max_passes):
        n_alive = int(alive.sum())
        if n_alive < k_min:
            break
        live_edge = alive[src] & alive[dst]
        # Peeling discards most of the graph in the first few passes, but the
        # masking above stays O(|E|) over the *original* edge count unless the
        # arrays are physically shrunk. Compacting keeps later passes cheap.
        n_live = int(live_edge.sum())
        if n_live < src.size // 2:
            src, dst, w = src[live_edge], dst[live_edge], w[live_edge]
            live_edge = np.ones(src.size, dtype=bool)

        ew = float(w[live_edge].sum())
        sm = float(scores[alive].sum())
        g = (ew + lambda_ * sm) / n_alive
        # Only sets within the size band can be the answer. Without the upper
        # bound this returns 30,000-account communities: on a graph whose
        # 50-core still holds 454k vertices, a large region of moderate
        # density outscores a small tight one, and neither a person nor an
        # ops queue can act on the result.
        if g > best_g and n_alive >= k_min and (k_max is None or n_alive <= k_max):
            best_g, best_pass = g, p

        ls, ld, lw = src[live_edge], dst[live_edge], w[live_edge].astype(np.float64)
        deg = np.bincount(ls, weights=lw, minlength=n)
        deg += np.bincount(ld, weights=lw, minlength=n)
        contrib = deg + lambda_ * scores
        contrib[~alive] = np.inf

        threshold = (1.0 + epsilon) * (contrib[alive].sum() / n_alive)
        drop = alive & (contrib <= threshold)
        n_drop = int(drop.sum())

        # Near the size band we care about, batch removal is too coarse: a
        # single pass can jump from ~600 vertices past the whole [k_min, k_max]
        # range to ~20, so no set inside the band is ever evaluated and the
        # returned "ring" is wherever the jump happened to land. Once the
        # survivors are within a few multiples of k_max, shrink at most 1% per
        # pass so every candidate size is actually seen. The remaining edge
        # set is small by then, so the extra passes are cheap.
        # 3% steps give ~170 passes across the band instead of ~600 at 1%,
        # which is enough resolution to find the best size without paying for
        # a pass per candidate size.
        fine_from = 2 * (k_max if k_max is not None else k_min * 20)
        if n_alive <= fine_from and n_drop > max(1, n_alive // 33):
            budget = max(1, n_alive // 33)
            cand_idx = np.flatnonzero(drop)
            keep_idx = cand_idx[np.argsort(contrib[cand_idx])[:budget]]
            drop = np.zeros(n, dtype=bool)
            drop[keep_idx] = True
            n_drop = int(drop.sum())
        if n_drop == 0:
            # No vertex is below the mean only when every contribution is
            # equal; remove the single minimum so the loop still terminates.
            drop = np.zeros(n, dtype=bool)
            drop[int(np.argmin(np.where(alive, contrib, np.inf)))] = True
            n_drop = 1
        if n_alive - n_drop < k_min:
            # Never shrink below the size floor; take the smallest allowed step.
            order = np.argsort(np.where(alive, contrib, np.inf))
            drop = np.zeros(n, dtype=bool)
            drop[order[: n_alive - k_min]] = True
            if not drop.any():
                break
        removed_at[drop] = p
        alive &= ~drop

    members = np.flatnonzero((removed_at == -1) | (removed_at >= best_pass)).astype(np.int32)
    if candidates is not None:
        keep = np.zeros(n, dtype=bool)
        keep[candidates] = True
        members = members[keep[members]]

    # Recompute the exact totals from the ORIGINAL edge arrays. The loop
    # compacts `src`/`dst`/`w` as vertices die, and the best set is alive at
    # an earlier pass than the end, so edges inside it may already have been
    # compacted away. Scoring against the compacted arrays would understate
    # the density of every ring.
    inside = np.zeros(n, dtype=bool)
    inside[members] = True
    live = inside[edges.src] & inside[edges.dst]
    ew = float(edges.weight[live].sum())
    sm = float(scores[members].sum())
    return Ring(members=members, density=(ew + lambda_ * sm) / max(members.size, 1),
                internal_weight=ew, score_mass=sm, lambda_=lambda_)


def extract_rings_batch(edges: EdgeList, scores: np.ndarray, *, lambda_: float,
                        k_min: int, top_k: int, g_min: float = 0.0,
                        k_max: int | None = None,
                        candidates: np.ndarray | None = None,
                        epsilon: float = 0.1,
                        progress: bool = False) -> list[Ring]:
    """Top-K rings using batch peeling. Extract, remove, repeat."""
    import sys
    import time as _time
    t_start = _time.time()
    available = np.zeros(edges.n_nodes, dtype=bool)
    if candidates is None:
        available[:] = True
    else:
        available[candidates] = True

    rings: list[Ring] = []
    work = edges
    for rank in range(top_k):
        cand = np.flatnonzero(available)
        if cand.size < k_min:
            break
        ring = peel_batch(work, scores, lambda_=lambda_, k_min=k_min,
                          k_max=k_max, candidates=cand, epsilon=epsilon)
        if ring.size < k_min or ring.density <= g_min:
            break
        ring.rank = rank + 1
        rings.append(ring)
        available[ring.members] = False
        if progress and (rank < 3 or (rank + 1) % 10 == 0):
            print(f"    ring {rank + 1:>4}/{top_k}  size={ring.size:>5} "
                  f"density={ring.density:8.3f}  edges_left={work.src.size:>10,}  "
                  f"{_time.time() - t_start:6.0f}s", file=sys.stderr, flush=True)

        # Drop the extracted ring's edges from the working list. Without this
        # every subsequent extraction rescans the full edge array and top-K
        # costs K times a full-graph pass.
        keep = available[work.src] & available[work.dst]
        if int(keep.sum()) < work.src.size:
            work = EdgeList(work.src[keep], work.dst[keep], work.weight[keep],
                            work.n_nodes)
    return rings


def _subgraph_totals(csr: CSR, members: np.ndarray,
                     scores: np.ndarray) -> tuple[float, float]:
    """Exact internal edge weight and score mass for a member set."""
    inside = np.zeros(csr.n_nodes, dtype=bool)
    inside[members] = True
    total = 0.0
    for v in members:
        lo, hi = csr.indptr[v], csr.indptr[v + 1]
        nbr, w = csr.indices[lo:hi], csr.weights[lo:hi]
        total += float(w[inside[nbr]].sum())
    return total / 2.0, float(scores[members].sum())


def extract_rings(csr: CSR, scores: np.ndarray, *, lambda_: float, k_min: int,
                  top_k: int, g_min: float = 0.0,
                  prune_tau: float | None = None) -> list[Ring]:
    """Top-K rings. Extract the densest set, remove it, repeat.

    ``prune_tau`` restricts the search to suspicious nodes and their
    neighbours, ``{v : s(v) > tau} u N({v : s(v) > tau})``. Whether pruning
    changes the result is reported, not assumed.
    """
    available = np.ones(csr.n_nodes, dtype=bool)
    if prune_tau is not None:
        seed = scores > prune_tau
        keep = seed.copy()
        for v in np.flatnonzero(seed):
            keep[csr.indices[csr.indptr[v]:csr.indptr[v + 1]]] = True
        available &= keep

    rings: list[Ring] = []
    for rank in range(top_k):
        cand = np.flatnonzero(available)
        if cand.size < k_min:
            break
        ring = peel_once(csr, scores, lambda_=lambda_, k_min=k_min, candidates=cand)
        if ring.size < k_min or ring.density <= g_min:
            break
        ring.rank = rank + 1
        rings.append(ring)
        available[ring.members] = False
    return rings
