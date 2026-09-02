"""Per-user features, computed independently within a single week.

**PPA ships no node features.** `node.csv`'s `vec_0..vec_7` are all 1.0, and
the authors' own model feeds `torch.ones(n, 52)` into its GNN - it has no node
information whatsoever. Every feature here is engineered from the order
stream, which is squarely the open problem PromoGuardian names in its future
work: *"Designing discriminative node features remains an open challenge."*

**Temporal discipline.** `build_features(week)` reads only that week's orders
and that week's graph. Week-1 features never see a week-2 row. Because the
two weeks are processed by the same code path, a feature that is impossible
to compute for week 1 is impossible for week 2 too - there is no way for an
asymmetry to sneak in.

**What the data does not permit.** `order_time` has day resolution and is
year-shifted (see `docs/data.md`), so there is no hour-of-day histogram and no
sub-day inter-order gap. Temporal features are day-level: active days,
busiest-day concentration, longest silent gap.

**No neighbour-label features.** "Fraction of neighbours labelled fraud" is a
strong and standard feature, and it is deliberately omitted. Labels are static
and user-level, so such a feature would let week-1 training absorb label
information about the exact users being scored in week 2. Excluding it costs
accuracy and buys a metric that means what it says.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

ORDER_FEATURES = [
    "n_orders", "n_distinct_sku", "sku_repeat_rate",
    "n_promo_orders", "promo_order_rate", "n_distinct_promo",
    "n_distinct_coupon", "n_distinct_location", "n_distinct_delivery",
    "n_distinct_stimulation",
    "active_days", "orders_per_active_day", "max_orders_in_day",
    "day_concentration", "longest_gap_days",
]
GRAPH_FEATURES = [
    "degree", "weighted_degree", "mean_edge_weight", "max_edge_weight",
    "mean_relations_per_edge", "max_relations_per_edge",
    "min_entity_size", "rare_edge_fraction",
    "core_number", "two_hop_size", "neighbour_degree_mean", "neighbour_degree_max",
]
FEATURE_NAMES = ORDER_FEATURES + GRAPH_FEATURES

# An edge is "rare" when its rarest shared entity is small enough to be
# genuine evidence rather than a crowd.
RARE_ENTITY_MAX = 10


def _group_nunique(keys: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Distinct non-null `values` per `keys` id, for keys in [0, n)."""
    ok = ~np.isnan(values) if values.dtype.kind == "f" else np.ones(values.size, bool)
    k, v = keys[ok], values[ok].astype(np.int64)
    if k.size == 0:
        return np.zeros(n, dtype=np.int32)
    order = np.lexsort((v, k))
    k, v = k[order], v[order]
    new = np.empty(k.size, dtype=bool)
    new[0] = True
    np.logical_or(k[1:] != k[:-1], v[1:] != v[:-1], out=new[1:])
    return np.bincount(k[new], minlength=n).astype(np.int32)


def order_features(week: int, cfg: Config, n_users: int) -> dict[str, np.ndarray]:
    proc = cfg.abs_path(cfg.paths.processed)
    t = pq.read_table(proc / f"orders_week{week}.parquet")
    uid = t["user_id"].to_numpy().astype(np.int64)
    day = t["day_ordinal"].to_numpy().astype(np.int64)

    f: dict[str, np.ndarray] = {}
    f["n_orders"] = np.bincount(uid, minlength=n_users).astype(np.float32)
    f["n_distinct_sku"] = _group_nunique(uid, t["sku_id"].to_numpy().astype(np.float64), n_users)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["sku_repeat_rate"] = np.where(
            f["n_orders"] > 0, 1.0 - f["n_distinct_sku"] / np.maximum(f["n_orders"], 1), 0.0)

    promo = t["r6"].to_numpy(zero_copy_only=False).astype(np.float64)
    has_promo = ~np.isnan(promo)
    f["n_promo_orders"] = np.bincount(uid[has_promo], minlength=n_users).astype(np.float32)
    f["promo_order_rate"] = np.where(f["n_orders"] > 0,
                                     f["n_promo_orders"] / np.maximum(f["n_orders"], 1), 0.0)
    f["n_distinct_promo"] = _group_nunique(uid, promo, n_users)
    for name, rel in [("n_distinct_coupon", "r7"), ("n_distinct_location", "r1"),
                      ("n_distinct_delivery", "r3"), ("n_distinct_stimulation", "r8")]:
        f[name] = _group_nunique(uid, t[rel].to_numpy(zero_copy_only=False).astype(np.float64), n_users)

    # Day-level temporal shape. No sub-day features exist in this data.
    day0 = day - day.min()
    n_days = int(day0.max()) + 1
    per_day = np.zeros((n_users, n_days), dtype=np.int32)
    np.add.at(per_day, (uid, day0), 1)
    active = (per_day > 0)
    f["active_days"] = active.sum(axis=1).astype(np.float32)
    f["max_orders_in_day"] = per_day.max(axis=1).astype(np.float32)
    f["orders_per_active_day"] = np.where(f["active_days"] > 0,
                                          f["n_orders"] / np.maximum(f["active_days"], 1), 0.0)
    f["day_concentration"] = np.where(f["n_orders"] > 0,
                                      f["max_orders_in_day"] / np.maximum(f["n_orders"], 1), 0.0)
    idx = np.arange(n_days)
    first = np.where(active.any(1), np.argmax(active, 1), 0)
    last = np.where(active.any(1), n_days - 1 - np.argmax(active[:, ::-1], 1), 0)
    f["longest_gap_days"] = ((last - first + 1) - f["active_days"]).astype(np.float32)
    del per_day, active
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def graph_features(week: int, cfg: Config, n_users: int) -> dict[str, np.ndarray]:
    """Graph-aggregated features.

    Every per-node aggregate is a `reduceat` over a once-sorted adjacency
    rather than `np.ufunc.at`. On 138M directed endpoint records `ufunc.at`
    is orders of magnitude slower and the intermediates do not fit in 16 GB.
    """
    import igraph as ig

    proc = cfg.abs_path(cfg.paths.processed)
    e = pq.read_table(proc / f"edges_week{week}.parquet")
    src = e["src"].to_numpy().astype(np.int32)
    dst = e["dst"].to_numpy().astype(np.int32)
    w = e["weight"].to_numpy().astype(np.float32)
    nrel = e["n_relations"].to_numpy().astype(np.float32)
    esz = e["min_entity_size"].to_numpy().astype(np.int32)
    del e

    # Sort every directed endpoint record by its owning node exactly once.
    node = np.concatenate([src, dst])
    order = np.argsort(node, kind="stable")
    node = node[order]
    deg = np.bincount(node, minlength=n_users).astype(np.float64)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    nonempty = deg > 0
    starts = indptr[:-1][nonempty].astype(np.int64)

    def per_node(values: np.ndarray, op) -> np.ndarray:
        """Aggregate a per-endpoint array into a per-node array."""
        out = np.zeros(n_users, dtype=np.float64)
        if starts.size:
            out[nonempty] = op(values[order], starts)
        return out

    f: dict[str, np.ndarray] = {"degree": deg}
    f["weighted_degree"] = per_node(np.concatenate([w, w]).astype(np.float64), np.add.reduceat)
    f["max_edge_weight"] = per_node(np.concatenate([w, w]).astype(np.float64), np.maximum.reduceat)
    f["mean_edge_weight"] = np.where(deg > 0, f["weighted_degree"] / np.maximum(deg, 1), 0.0)

    nrel2 = np.concatenate([nrel, nrel]).astype(np.float64)
    f["mean_relations_per_edge"] = np.where(
        deg > 0, per_node(nrel2, np.add.reduceat) / np.maximum(deg, 1), 0.0)
    f["max_relations_per_edge"] = per_node(nrel2, np.maximum.reduceat)
    del nrel2, nrel

    esz2 = np.concatenate([esz, esz]).astype(np.float64)
    f["min_entity_size"] = per_node(esz2, np.minimum.reduceat)
    rare = (esz2 <= RARE_ENTITY_MAX).astype(np.float64)
    f["rare_edge_fraction"] = np.where(
        deg > 0, per_node(rare, np.add.reduceat) / np.maximum(deg, 1), 0.0)
    del esz2, rare, esz

    # Neighbour degree: the other endpoint of each record.
    other_deg = deg[np.concatenate([dst, src])]
    f["neighbour_degree_mean"] = np.where(
        deg > 0, per_node(other_deg, np.add.reduceat) / np.maximum(deg, 1), 0.0)
    f["neighbour_degree_max"] = per_node(other_deg, np.maximum.reduceat)
    # Sum of neighbour degrees = 2-hop reach counted with multiplicity.
    f["two_hop_size"] = per_node(other_deg, np.add.reduceat)
    del other_deg, node, order

    g = ig.Graph(n=n_users, edges=np.stack([src, dst], axis=1))
    f["core_number"] = np.asarray(g.coreness(), dtype=np.float64)
    del g
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def build_features(week: int, cfg: Config | None = None, *, force: bool = False) -> Path:
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    dest = proc / f"features_week{week}.parquet"
    if dest.exists() and not force:
        return dest

    orders = pq.read_table(proc / f"orders_week{week}.parquet", columns=["user_id"])
    n_users = int(orders["user_id"].to_numpy().max()) + 1
    del orders

    feats = order_features(week, cfg, n_users)
    feats.update(graph_features(week, cfg, n_users))
    missing = set(FEATURE_NAMES) - set(feats)
    if missing:
        raise RuntimeError(f"missing features: {sorted(missing)}")

    table = pa.table({"user_id": pa.array(np.arange(n_users, dtype=np.int32))} |
                     {k: pa.array(feats[k]) for k in FEATURE_NAMES})
    pq.write_table(table, dest, compression="zstd")
    (proc / f"features_week{week}_manifest.json").write_text(json.dumps({
        "week": week, "n_users": n_users, "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES, "bytes": dest.stat().st_size}, indent=2))
    return dest


def main() -> None:
    cfg = load_config()
    for week in (1, 2):
        dest = build_features(week, cfg, force=True)
        m = json.loads((dest.parent / f"features_week{week}_manifest.json").read_text())
        print(f"week {week}: {m['n_users']:,} users x {m['n_features']} features "
              f"({m['bytes']/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
