"""Regional subsampling for fast iteration.

**Nodes are never sampled at random.** Sampling nodes tears rings apart: it
removes most of every ring's members, so the thing I am trying to detect stops
existing and any metric computed on the sample is meaningless.

I originally wanted a geohash-prefix regional cut, but the released data does
not allow it. `r1` is an anonymised integer id - 2,023,601 distinct values
like `6838556` - not a geohash string, so there is no prefix to slice on. The
anonymisation destroyed exactly the spatial locality a prefix would exploit.

So I cut on **entities, never on users**. A "region" is a set of `r1` location
entities; every user touching a selected entity is kept, and then every edge
among the kept users is kept, including edges from other relations. A ring
anchored on a selected location arrives intact rather than decimated.

That is still not enough on its own - see `close_neighbourhood` and
FAILURES.md. `ring_preservation` measures the damage honestly: it reports how
many labelled-fraud components survive whole and how many were split. This
sample is a development convenience for fast iteration; the reported ring
metrics come from the full week-2 graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config


def select_region_entities(entity: np.ndarray, user: np.ndarray, *,
                           target_nodes: int, n_max: int,
                           seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick whole location-entities until ~`target_nodes` users are covered.

    Entities are shuffled with a fixed seed and taken in that order, so the
    selection is reproducible but not biased toward large or small regions.
    Returns (selected_entity_ids, covered_user_ids).
    """
    keep = ~np.isnan(entity)
    entity, user = entity[keep].astype(np.int64), user[keep]

    order = np.lexsort((user, entity))
    entity, user = entity[order], user[order]
    uniq = np.empty(entity.size, dtype=bool)
    uniq[0] = True
    np.logical_or(entity[1:] != entity[:-1], user[1:] != user[:-1], out=uniq[1:])
    entity, user = entity[uniq], user[uniq]

    starts = np.flatnonzero(np.concatenate(([True], entity[1:] != entity[:-1])))
    sizes = np.diff(np.append(starts, entity.size))
    ent_ids = entity[starts]

    # Only entities that can actually induce edges are useful as anchors.
    ok = (sizes >= 2) & (sizes <= n_max)
    starts, sizes, ent_ids = starts[ok], sizes[ok], ent_ids[ok]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(ent_ids.size)
    cum = np.cumsum(sizes[perm])
    n_take = int(np.searchsorted(cum, target_nodes) + 1)
    n_take = min(n_take, perm.size)
    chosen = perm[:n_take]

    users = np.concatenate([user[starts[c]: starts[c] + sizes[c]] for c in chosen])
    return ent_ids[chosen], np.unique(users)


def close_neighbourhood(seeds: np.ndarray, src: np.ndarray, dst: np.ndarray,
                        n_nodes: int, hops: int = 1) -> np.ndarray:
    """Expand a seed set to include its graph neighbours, `hops` times.

    This is what makes the cut ring-safe. Rings are *dense*: in a near-clique
    every member is one hop from every other, so seeding a single member pulls
    the entire ring in. Anchoring on locations alone splits rings, because PPA
    rings are held together largely by promotion (r6) and stimulation (r8),
    not by location - measured, see the manifest's `ring_preservation`.

    Deliberately **label-free**. Selecting nodes by fraud label would leak the
    structure we are trying to measure into the sample that measures it.
    """
    inside = np.zeros(n_nodes, dtype=bool)
    inside[seeds] = True
    for _ in range(max(hops, 0)):
        touch = inside[src] | inside[dst]
        nxt = inside.copy()
        nxt[src[touch]] = True
        nxt[dst[touch]] = True
        if nxt.sum() == inside.sum():
            break
        inside = nxt
    return np.flatnonzero(inside).astype(np.int32)


def ring_preservation(members: np.ndarray, labels: np.ndarray,
                      src: np.ndarray, dst: np.ndarray) -> dict:
    """How badly did the cut damage labelled-fraud groups?

    Ground-truth "rings" are the connected components of the subgraph induced
    by fraud-labelled users. Reports how many survive whole in the sample.
    """
    import igraph as ig

    fraud = np.flatnonzero(labels == 1)
    in_sample = np.zeros(labels.size, dtype=bool)
    in_sample[members] = True

    m = np.isin(src, fraud) & np.isin(dst, fraud)
    fs, fd = src[m], dst[m]
    if fs.size == 0:
        return {"fraud_components": 0, "fully_preserved": 0, "split": 0, "lost": 0}

    nodes = np.unique(np.concatenate([fs, fd]))
    ei = np.stack([np.searchsorted(nodes, fs), np.searchsorted(nodes, fd)], axis=1)
    g = ig.Graph(n=nodes.size, edges=ei)

    whole = split = lost = 0
    for comp in g.connected_components():
        orig = nodes[np.array(comp, dtype=np.int64)]
        n_in = int(in_sample[orig].sum())
        if n_in == orig.size:
            whole += 1
        elif n_in == 0:
            lost += 1
        else:
            split += 1
    return {"fraud_components": whole + split + lost, "fully_preserved": whole,
            "split": split, "lost": lost,
            "preserved_fraction": round(whole / max(whole + split + lost, 1), 4)}


def build_subsample(week: int = 2, cfg: Config | None = None, *,
                    force: bool = False) -> Path:
    """Write the regional subsample: node list, induced edges, manifest."""
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    dest = proc / f"subsample_week{week}_edges.parquet"
    if dest.exists() and not force:
        return dest

    orders = pq.read_table(proc / f"orders_week{week}.parquet",
                           columns=["user_id", cfg.subsample.region_relation])
    users = orders["user_id"].to_numpy()
    ent = orders[cfg.subsample.region_relation].to_numpy(zero_copy_only=False).astype(np.float64)
    del orders

    regions, seeds = select_region_entities(
        ent, users, target_nodes=cfg.subsample.seed_nodes,
        n_max=cfg.graph.n_max, seed=cfg.seed)
    del ent, users

    edges = pq.read_table(proc / f"edges_week{week}.parquet")
    src = edges["src"].to_numpy(); dst = edges["dst"].to_numpy()
    n_nodes = int(max(src.max(), dst.max())) + 1
    members = close_neighbourhood(seeds, src, dst, n_nodes,
                                  hops=cfg.subsample.closure_hops)
    keep_mask = np.zeros(n_nodes, dtype=bool)
    keep_mask[members] = True
    sel = keep_mask[src] & keep_mask[dst]           # induced subgraph: ALL edges among members
    sub = edges.filter(pa.array(sel))
    pq.write_table(sub, dest, compression="zstd")
    del edges

    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()
    lab_pad = np.full(n_nodes, -1, dtype=np.int8)
    lab_pad[: labels.size] = labels
    in_sample = members[members < labels.size]

    manifest = {
        "week": week, "seed": cfg.seed,
        "region_relation": cfg.subsample.region_relation,
        "seed_nodes_target": cfg.subsample.seed_nodes,
        "closure_hops": cfg.subsample.closure_hops,
        "regions_selected": int(regions.size),
        "seed_nodes": int(seeds.size),
        "nodes": int(members.size),
        "nodes_with_labels": int(in_sample.size),
        "edges": int(sub.num_rows),
        "label_counts": {
            str(int(k)): int(v) for k, v in
            zip(*np.unique(lab_pad[in_sample], return_counts=True))
        },
        "edge_fraction_of_full": round(sub.num_rows / max(src.size, 1), 5),
        "ring_preservation": ring_preservation(members, lab_pad, src, dst),
        "bytes": dest.stat().st_size,
    }
    np.save(proc / f"subsample_week{week}_nodes.npy", members.astype(np.int32))
    (proc / f"subsample_week{week}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def main() -> None:
    cfg = load_config()
    dest = build_subsample(2, cfg, force=True)
    m = json.loads((dest.parent / f"subsample_week2_manifest.json").read_text())
    print(f"regions: {m['regions_selected']:,}  nodes: {m['nodes']:,}  edges: {m['edges']:,}")
    print(f"labels in sample: {m['label_counts']}")
    print(f"ring preservation: {m['ring_preservation']}")


if __name__ == "__main__":
    main()
