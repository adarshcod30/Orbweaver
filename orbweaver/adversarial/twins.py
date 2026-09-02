"""Edges an attacker cannot cut by splitting the ring up.

Fragmentation works, and the reason it works is specific: it deletes the
*shared entities* that tie a group together and leaves the members' behaviour
completely untouched. Fifty accounts that all order the same way at the same
times are still fifty accounts that order the same way at the same times, even
after every address and promotion they had in common has been severed.

So this adds a relation the attacker's move does not touch: an edge between two
suspicious accounts that behave alike. It is not free — behaviour similarity is
much weaker evidence than a shared delivery record, and ordinary customers
behave alike too — so the question is whether it recovers enough of what
fragmentation removes to be worth the false positives it brings.

Three deliberate constraints:

- **Only among accounts the scorer already flagged.** Behaviour similarity
  across the whole population would connect millions of ordinary customers who
  happen to shop alike. Confining it to accounts above the score cut-off keeps
  it a refinement of a suspicious region rather than a new source of noise.
- **Mutual k-nearest neighbours.** If A is one of B's five closest and B is one
  of A's, they are linked. One-directional nearest-neighbour graphs are
  asymmetric and let a single account acquire hundreds of inbound edges;
  mutual kNN stays sparse and symmetric on its own.
- **Weighted like every other relation.** The twin weight is the fraud–fraud
  lift measured on training accounts, times the median entity edge weight. No
  hand-picked constant, and the same rule that sets the weight of a shared
  location sets this one.

Twins are added *after* the fragmentation cuts, because that is the order the
attack happens in: the attacker severs the entities, and the behaviour is what
is left.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList

K_NEIGHBOURS = 5
# 78,000 candidates squared is a 24 GB similarity matrix, so the search runs
# in blocks. Each block is one matmul against the whole candidate set.
BLOCK = 2048


def standardise(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def mutual_knn(X: np.ndarray, k: int = K_NEIGHBOURS,
               block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Mutual k-nearest neighbours by cosine similarity, in blocks.

    Returns positional index pairs (i < j). Working in blocks keeps peak
    memory at `block × n` rather than `n × n`.
    """
    Z = X.astype(np.float32)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Z = Z / norm

    n = Z.shape[0]
    nbrs = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        # numpy 2.2 on Apple's Accelerate BLAS raises divide-by-zero, overflow
        # and invalid warnings from matmul even on finite, unit-norm input.
        # The results are exact - checked against explicit dot products - so
        # the flags are spurious and only these three are silenced, only here.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim = Z[start:stop] @ Z.T                # (block, n)
        # An account is its own nearest neighbour; remove it before ranking.
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-sim[rows, idx], axis=1)
        nbrs[start:stop] = idx[rows, order]
        del sim

    # Keep a pair only if each is in the other's list.
    src = np.repeat(np.arange(n), k)
    dst = nbrs.ravel()
    lookup = np.zeros((n, k), dtype=np.int64)
    lookup[:] = nbrs
    mutual = np.zeros(src.size, dtype=bool)
    for j in range(k):
        mutual |= (lookup[dst, j] == src)
    src, dst = src[mutual], dst[mutual]
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def twin_weight(cfg: Config, src: np.ndarray, dst: np.ndarray,
                labels: np.ndarray, visible: np.ndarray,
                median_entity_weight: float) -> dict:
    """How much a behaviour edge is worth, measured rather than chosen.

    Exactly the rule every entity relation gets: how much more often does an
    edge of this kind join two known fraudsters than chance predicts, measured
    on training accounts only.
    """
    both = visible[src] & visible[dst]
    n = int(both.sum())
    if n < 200:
        return {"edges_labelled": n, "lift": 1.0, "measured": False,
                "weight": float(median_entity_weight),
                "note": "too few labelled twin edges to measure; neutral weight"}
    a, b = labels[src[both]], labels[dst[both]]
    ff = float(((a == 1) & (b == 1)).sum()) / n
    p = (float((a == 1).sum()) + float((b == 1).sum())) / (2 * n)
    lift = ff / (p * p) if p > 0 else 1.0
    return {"edges_labelled": n, "fraud_fraud_rate": round(ff, 6),
            "expected_if_random": round(p * p, 6), "lift": round(lift, 4),
            "measured": True,
            "weight": round(float(lift * median_entity_weight), 6),
            "note": ("lift measured on training accounts only, times the median "
                     "entity edge weight - no hand-picked constant")}


def build_twins(cfg: Config | None = None) -> dict:
    """Twin edges over the accounts the scorer flagged, with their weight."""
    from orbweaver.scoring.xgb_graph import load_features

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()
    n = labels.size

    scores = np.zeros(n)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

    candidates = np.flatnonzero(scores > cfg.rings.prune_tau_headline)
    X = standardise(load_features(2, cfg, n, "late")[candidates])

    i, j = mutual_knn(X)
    src, dst = candidates[i], candidates[j]

    e = pq.read_table(proc / "edges_week2_late.parquet", columns=["weight"])
    median_w = float(np.median(e["weight"].to_numpy()))

    from eval.split import make_split
    split = make_split(cfg)
    visible = np.zeros(n, dtype=bool)
    visible[split.train_pool] = True

    w = twin_weight(cfg, src, dst, labels, visible, median_w)
    return {
        "candidates": int(candidates.size),
        "twin_edges": int(src.size),
        "k": K_NEIGHBOURS,
        "median_entity_edge_weight": round(median_w, 6),
        "weight": w,
        "src": src, "dst": dst,
    }


def add_twins(edges: EdgeList, twins: dict) -> EdgeList:
    """Twin edges on top of an existing graph, at their measured weight."""
    w = np.full(twins["src"].size, twins["weight"]["weight"], dtype=np.float64)
    return EdgeList(
        np.concatenate([edges.src, twins["src"]]),
        np.concatenate([edges.dst, twins["dst"]]),
        np.concatenate([edges.weight, w]),
        edges.n_nodes,
    )


def run(cfg: Config | None = None) -> dict:
    from eval.metrics import ltv_proxy
    from eval.run_rings import load_edges, prune
    from eval.split import make_split
    from orbweaver.adversarial.fragment import fraud_components, fragment_graph
    from orbweaver.rings.cost import evaluate_rings
    from orbweaver.rings.hostel_test import run_hostel_test
    from orbweaver.rings.peel import extract_rings_batch

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    labels = split.labels
    n = labels.size

    scores = np.zeros(n)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

    f = pq.read_table(proc / "features_week2_early.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    twins = build_twins(cfg)
    print(f"twin edges: {twins['twin_edges']:,} among "
          f"{twins['candidates']:,} flagged accounts")
    w = twins["weight"]
    print(f"their fraud-fraud lift on training accounts: {w.get('lift')} "
          f"-> weight {w['weight']}")

    full = load_edges("late", cfg, n)
    comps = fraud_components(labels, full.src, full.dst)

    def measure(edges: EdgeList) -> dict:
        sub = prune(edges, scores, cfg.rings.prune_tau_headline)
        rings = extract_rings_batch(sub, scores, lambda_=cfg.rings.lambda_headline,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
        block = evaluate_rings(rings, labels, ltv, restrict_to=split.test)
        return {"n_rings": len(rings),
                "accounts_in_rings": block.get("accounts_in_rings"),
                "ring_precision": block.get("ring_precision"),
                "fraud_members": block.get("fraud_members"),
                "normal_flagged_per_fraud_caught":
                    block.get("normal_flagged_per_fraud_caught")}, rings

    rows = {}
    intact_wo, _ = measure(full)
    intact_w, rings_twin_intact = measure(add_twins(full, twins))
    rows["intact"] = {"cell_size": None, "without_twins": intact_wo,
                      "with_twins": intact_w}
    print(f"{'cells':>8s} {'without twins':>14s} {'with twins':>12s} {'change':>8s}")
    print(f"{'intact':>8s} {str(intact_wo['ring_precision']):>14s} "
          f"{str(intact_w['ring_precision']):>12s} "
          f"{intact_w['ring_precision'] - intact_wo['ring_precision']:>+8.4f}")

    for c in (3, 5, 10, 20):
        damaged, _ = fragment_graph(full, comps, c, cfg.seed)
        wo, _ = measure(damaged)
        # Twins go on AFTER the cuts: the attacker severs shared entities, and
        # behaviour is what survives that.
        wi, _ = measure(add_twins(damaged, twins))
        rows[f"cells_of_{c}"] = {"cell_size": c, "without_twins": wo,
                                 "with_twins": wi}
        print(f"{c:>8} {str(wo['ring_precision']):>14s} "
              f"{str(wi['ring_precision']):>12s} "
              f"{(wi['ring_precision'] or 0) - (wo['ring_precision'] or 0):>+8.4f}")

    hostel = run_hostel_test(rings_twin_intact, cfg)

    return {
        "twins": {k: v for k, v in twins.items() if k not in ("src", "dst")},
        "fragmentation": rows,
        "hostel_test_with_twins": {
            "clusters_found": hostel.get("clusters_found"),
            "clusters_with_a_member_in_a_ring":
                hostel.get("clusters_with_a_member_in_a_ring"),
            "share_of_clusters_touched": hostel.get("share_of_clusters_touched"),
        },
        "method": ("Mutual 5-nearest-neighbour edges in standardised feature "
                   "space, among accounts above the score cut-off only, "
                   "weighted by their measured fraud-fraud lift on training "
                   "accounts times the median entity edge weight. Added after "
                   "the fragmentation cuts, because the attacker severs shared "
                   "entities and behaviour is what survives."),
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "twins.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    h = out["hostel_test_with_twins"]
    print(f"\nhostel test with twins present: "
          f"{h['clusters_with_a_member_in_a_ring']} of {h['clusters_found']:,} "
          f"legitimate clusters touched ({h['share_of_clusters_touched']:.2%})")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
