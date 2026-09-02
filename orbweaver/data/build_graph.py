"""Build the multi-relation user graph with entity-rarity edge weights.

Two users are connected when they share an entity - a geohash, a delivery
record, a promotion, a coupon type, a stimulation id. This is the object the
whole project rests on: promotion-abuse rings are invisible per-order and
visible only as density here.

**Entity capping is not an optimisation, it is the algorithm.** An entity
shared by a huge number of users is not evidence of anything. Measured on the
real week-2 orders (`docs/data.md`):

- uncapped:      5,461,311,086,506 user-pairs   (5.46 trillion)
- `N_max = 500`:       568,161,582 pairs        (will not fit in 16 GB)
- `N_max = 100`:        71,810,711 pairs        (the configured default)

One coupon type is shared by 3,187,247 users - 97.5 % of the entire user base.
Left uncapped it alone contributes 5.09 trillion pairs and drowns every real
signal.

**Edge weight**, following FRAUDAR's camouflage-resistant weighting:

    w_r(e) = 1 / log(rarity_base + |users(e)|)      rarer entity => stronger
    w(u,v) = sum over all (r, e) shared by u and v

Rationale: a fraudster can cheaply add edges through *common* entities as
camouflage. Rarity weighting makes those edges nearly free of value, so
camouflage does not raise a subgraph's density.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config


def rarity_weight(size: np.ndarray, base: float) -> np.ndarray:
    """w_r(e) = 1 / log(base + |users(e)|). Monotone decreasing in size."""
    return (1.0 / np.log(base + size.astype(np.float64))).astype(np.float32)


def pairs_from_groups(users_sorted: np.ndarray, sizes: np.ndarray,
                      starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All within-group user pairs, vectorised.

    `users_sorted` is grouped contiguously; group g occupies
    `users_sorted[starts[g] : starts[g] + sizes[g]]`.

    Because sizes are capped at `N_max`, there are at most `N_max` *distinct*
    group sizes. We compute the upper-triangle index pattern once per distinct
    size and broadcast it over every group of that size - so the work is
    O(N_max) numpy calls, not O(#groups) python iterations.

    Returns (left, right, group_index) with left < right positionally.
    """
    out_l, out_r, out_g = [], [], []
    for n in np.unique(sizes):
        if n < 2:
            continue
        which = np.flatnonzero(sizes == n)
        i, j = np.triu_indices(int(n), k=1)          # (P,) each, P = n(n-1)/2
        base = starts[which][:, None]                 # (m, 1)
        out_l.append(users_sorted[base + i[None, :]].ravel())
        out_r.append(users_sorted[base + j[None, :]].ravel())
        out_g.append(np.repeat(which, i.size))
    if not out_l:
        empty_i = np.empty(0, dtype=users_sorted.dtype)
        return empty_i, empty_i, np.empty(0, dtype=np.int64)
    return np.concatenate(out_l), np.concatenate(out_r), np.concatenate(out_g)


def relation_edges(user_id: np.ndarray, entity: np.ndarray, *, n_max: int,
                   rarity_base: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """User pairs induced by one relation column.

    Returns (src, dst, weight, entity_size) with src < dst.
    Entities with fewer than 2 or more than `n_max` distinct users are dropped.
    """
    keep = ~np.isnan(entity) if entity.dtype.kind == "f" else np.ones(len(entity), bool)
    user_id, entity = user_id[keep], entity[keep].astype(np.int64)

    # One user ordering twice on the same entity is one member, not two.
    key = np.stack([entity, user_id.astype(np.int64)])
    order = np.lexsort(key[::-1])
    entity, user_id = entity[order], user_id[order]
    uniq_mask = np.empty(len(entity), dtype=bool)
    uniq_mask[0] = True
    np.not_equal(entity[1:], entity[:-1], out=uniq_mask[1:])
    np.logical_or(uniq_mask[1:], user_id[1:] != user_id[:-1], out=uniq_mask[1:])
    entity, user_id = entity[uniq_mask], user_id[uniq_mask]

    # Group boundaries over the (now sorted, deduplicated) entity column.
    starts = np.flatnonzero(np.concatenate(([True], entity[1:] != entity[:-1])))
    sizes = np.diff(np.append(starts, len(entity)))

    sel = (sizes >= 2) & (sizes <= n_max)
    starts, sizes = starts[sel], sizes[sel]
    if starts.size == 0:
        e32 = np.empty(0, np.int32)
        return e32, e32, np.empty(0, np.float32), np.empty(0, np.int32)

    left, right, gidx = pairs_from_groups(user_id, sizes, starts)
    w = rarity_weight(sizes, rarity_base)[gidx]
    src = np.minimum(left, right).astype(np.int32)
    dst = np.maximum(left, right).astype(np.int32)
    return src, dst, w, sizes[gidx].astype(np.int32)


def build_graph(week: int, cfg: Config | None = None, *, n_max: int | None = None,
                force: bool = False) -> Path:
    """Build the week's multi-relation graph (View A) and write it to parquet.

    Edges are aggregated across relations: one row per undirected user pair,
    carrying the summed rarity weight, how many relations connect the pair,
    and the size of the rarest entity they share (the strongest evidence).
    """
    cfg = cfg or load_config()
    n_max = n_max or cfg.graph.n_max
    proc = cfg.abs_path(cfg.paths.processed)
    suffix = "" if n_max == cfg.graph.n_max else f"_nmax{n_max}"
    dest = proc / f"edges_week{week}{suffix}.parquet"
    if dest.exists() and not force:
        return dest

    orders = pq.read_table(proc / f"orders_week{week}.parquet")
    n_users = int(orders["user_id"].to_numpy().max()) + 1
    users = orders["user_id"].to_numpy()

    all_src, all_dst, all_w, all_esz, all_rel = [], [], [], [], []
    per_relation = {}
    for ridx, rel in enumerate(cfg.data.buildable_relations, start=1):
        ent = orders[rel].to_numpy(zero_copy_only=False).astype(np.float64)
        s, d, w, esz = relation_edges(users, ent, n_max=n_max,
                                      rarity_base=cfg.graph.rarity_base)
        per_relation[rel] = {"pairs": int(s.size),
                             "mean_weight": float(w.mean()) if s.size else 0.0}
        all_src.append(s); all_dst.append(d); all_w.append(w); all_esz.append(esz)
        all_rel.append(np.full(s.size, ridx, dtype=np.int8))
        del ent, s, d, w, esz
    del orders

    src = np.concatenate(all_src); dst = np.concatenate(all_dst)
    w = np.concatenate(all_w); esz = np.concatenate(all_esz)
    rel = np.concatenate(all_rel)
    del all_src, all_dst, all_w, all_esz, all_rel
    raw_pairs = int(src.size)

    # Aggregate duplicate pairs across relations and entities.
    key = src.astype(np.int64) * n_users + dst
    order = np.argsort(key, kind="stable")
    key, src, dst, w, esz, rel = (a[order] for a in (key, src, dst, w, esz, rel))
    del order
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    n_edges = first.size

    # Rows are sorted by key, so each pair's rows are contiguous and we can
    # use reduceat. np.minimum.at / np.bitwise_or.at would give the same
    # answer but are an order of magnitude slower at 70M+ rows.
    weight = np.add.reduceat(w.astype(np.float64), first).astype(np.float32)
    min_entity = np.minimum.reduceat(esz, first)
    # distinct relations per pair: relations are 1..8, so use a bitmask
    relmask = np.bitwise_or.reduceat((1 << (rel.astype(np.int16) - 1)), first)
    n_rel = np.zeros(n_edges, dtype=np.int8)
    for b in range(len(cfg.data.buildable_relations)):
        n_rel += ((relmask >> b) & 1).astype(np.int8)

    table = pa.table({
        "src": pa.array(src[first], pa.int32()),
        "dst": pa.array(dst[first], pa.int32()),
        "weight": pa.array(weight, pa.float32()),
        "n_relations": pa.array(n_rel, pa.int8()),
        "relation_mask": pa.array(relmask, pa.int16()),
        "min_entity_size": pa.array(min_entity, pa.int32()),
    })
    pq.write_table(table, dest, compression="zstd")

    manifest = {
        "week": week, "n_max": n_max, "rarity_base": cfg.graph.rarity_base,
        "relations": cfg.data.buildable_relations,
        "raw_pairs_before_aggregation": raw_pairs,
        "unique_edges": n_edges,
        "distinct_nodes_in_edges": int(np.unique(np.concatenate([src[first], dst[first]])).size),
        "weight_min": float(weight.min()), "weight_max": float(weight.max()),
        "weight_mean": float(weight.mean()),
        "per_relation": per_relation,
        "bytes": dest.stat().st_size,
    }
    (proc / f"edges_week{week}{suffix}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def main() -> None:
    cfg = load_config()
    for week in (1, 2):
        dest = build_graph(week, cfg, force=True)
        m = json.loads((dest.parent / f"{dest.stem}_manifest.json").read_text())
        print(f"week {week}: {m['raw_pairs_before_aggregation']:>12,} raw pairs -> "
              f"{m['unique_edges']:>11,} unique edges over "
              f"{m['distinct_nodes_in_edges']:>9,} nodes  ({m['bytes']/1e6:.0f} MB)")
        for rel, s in m["per_relation"].items():
            print(f"    {rel}: {s['pairs']:>11,} pairs  mean w={s['mean_weight']:.4f}")


if __name__ == "__main__":
    main()
