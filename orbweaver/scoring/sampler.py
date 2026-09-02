"""Neighbour sampling for mini-batch GNN training, without a compiled backend.

PyTorch Geometric's `NeighborLoader` needs `pyg-lib` or `torch-sparse`, neither
of which publishes a wheel for Python 3.13 on arm64 macOS — the same wall DGL
hit. Rather than downgrade the project or build extensions from source, this
implements the sampler directly against a CSR adjacency in numpy. It is about
forty lines, it is the standard GraphSAGE scheme, and it has no dependency
that can stop working.

Sampling is **with replacement**, which is what the original GraphSAGE paper
does: it gives every node a fixed-size neighbourhood regardless of degree, so
batches are rectangular and memory stays flat. Nodes with no edges contribute
only themselves.
"""
from __future__ import annotations

import numpy as np


def build_csr(src: np.ndarray, dst: np.ndarray, n_nodes: int):
    """Symmetric CSR from a deduplicated undirected edge list."""
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n_nodes), out=indptr[1:])
    return indptr, v.astype(np.int64)


def sample_block(indptr: np.ndarray, indices: np.ndarray, frontier: np.ndarray,
                 fanout: int, rng: np.random.Generator):
    """Sample `fanout` neighbours per node. Returns (source, target) pairs."""
    rep = np.repeat(frontier, fanout)
    deg = indptr[rep + 1] - indptr[rep]
    live = deg > 0
    rep, deg = rep[live], deg[live]
    if rep.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    picks = indptr[rep] + (rng.random(rep.size) * deg).astype(np.int64)
    return indices[picks], rep


def sample_subgraph(indptr: np.ndarray, indices: np.ndarray, seeds: np.ndarray,
                    fanouts: tuple[int, ...], rng: np.random.Generator):
    """Multi-hop neighbourhood around `seeds`, relabelled to a compact range.

    Returns (nodes, edge_index, n_seeds). `nodes[:n_seeds]` are the seeds in
    their original order, so a model's output rows line up with the batch.
    """
    nodes = seeds.copy()
    all_src, all_dst = [], []
    frontier = seeds
    for fanout in fanouts:
        nbr, own = sample_block(indptr, indices, frontier, fanout, rng)
        if nbr.size == 0:
            break
        all_src.append(nbr)
        all_dst.append(own)
        nodes = np.concatenate([nodes, nbr])
        frontier = np.unique(nbr)

    # Keep the seeds first and deduplicate everything after them.
    _, first_idx = np.unique(nodes, return_index=True)
    keep = np.zeros(nodes.size, dtype=bool)
    keep[first_idx] = True
    keep[: seeds.size] = True
    order = np.flatnonzero(keep)
    order = np.concatenate([np.arange(seeds.size),
                            order[order >= seeds.size]])
    nodes = nodes[order]
    # Later duplicates of a seed may survive; drop them.
    seen = {}
    final = []
    for i, v in enumerate(nodes):
        if v not in seen:
            seen[v] = len(final)
            final.append(v)
    nodes = np.array(final, dtype=np.int64)

    if not all_src:
        return nodes, np.zeros((2, 0), dtype=np.int64), seeds.size

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    lut = np.full(int(max(nodes.max(), src.max(), dst.max())) + 1, -1, dtype=np.int64)
    lut[nodes] = np.arange(nodes.size)
    s, d = lut[src], lut[dst]
    ok = (s >= 0) & (d >= 0)
    return nodes, np.stack([s[ok], d[ok]]), seeds.size


def batches(nodes: np.ndarray, batch_size: int, rng: np.random.Generator,
            shuffle: bool = True):
    idx = rng.permutation(nodes.size) if shuffle else np.arange(nodes.size)
    for i in range(0, idx.size, batch_size):
        yield nodes[idx[i:i + batch_size]]
