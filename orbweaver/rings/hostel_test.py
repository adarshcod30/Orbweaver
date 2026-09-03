"""The hostel test: does this thing flag a building full of ordinary people?

In India a shared delivery address is very often a hostel, a paying-guest
place, an office or a joint family. Fifteen accounts ordering to one address
is completely normal, and a detector that cannot tell that from a fraud ring
is not deployable — the failure is expensive and it lands on real customers
who did nothing wrong.

So this looks for the population that most resembles a ring without being one:
groups of accounts sharing a location entity, large enough to look
coordinated, whose labelled members are overwhelmingly **normal**. Then it
asks what the pipeline does with them.

The interesting output is not only "how many did we wrongly flag" but "what
separates the ones we flagged from the ones we did not" — because that
difference is what an analyst would need in order to trust or override the
system.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

FRAUD, NORMAL, UNLABELLED = 1, 0, -1

# A cluster has to be big enough to look like a ring before it is an
# interesting false-positive risk.
MIN_CLUSTER = 15
# And overwhelmingly normal, so that flagging it really is a mistake.
MIN_NORMAL_SHARE = 0.8
MIN_LABELLED = 3


def find_colocated_clusters(cfg: Config, labels: np.ndarray,
                            relation: str = "r1") -> list[dict]:
    """Groups sharing one location entity that look like rings but are not."""
    proc = cfg.abs_path(cfg.paths.processed)
    t = pq.read_table(proc / "orders_week2.parquet", columns=["user_id", relation])
    ent = t[relation].to_numpy(zero_copy_only=False).astype(np.float64)
    uid = t["user_id"].to_numpy()
    keep = ~np.isnan(ent)
    ent, uid = ent[keep].astype(np.int64), uid[keep]
    if ent.size == 0:
        return []

    order = np.lexsort((uid, ent))
    ent, uid = ent[order], uid[order]
    uniq = np.empty(ent.size, dtype=bool)
    uniq[0] = True
    np.logical_or(ent[1:] != ent[:-1], uid[1:] != uid[:-1], out=uniq[1:])
    ent, uid = ent[uniq], uid[uniq]

    starts = np.flatnonzero(np.concatenate(([True], ent[1:] != ent[:-1])))
    sizes = np.diff(np.append(starts, ent.size))

    # Same cap the graph builder uses: above it, an entity is a crowd rather
    # than an address, and no edges were built from it anyway.
    sel = (sizes >= MIN_CLUSTER) & (sizes <= cfg.graph.n_max)
    out = []
    for s, n in zip(starts[sel], sizes[sel]):
        members = uid[s:s + n]
        lab = labels[members]
        n_lab = int((lab != UNLABELLED).sum())
        if n_lab < MIN_LABELLED:
            continue
        normal_share = float((lab == NORMAL).sum() / n_lab)
        if normal_share >= MIN_NORMAL_SHARE:
            out.append({"entity": int(ent[s]), "members": members,
                        "size": int(n), "labelled": n_lab,
                        "normal_share": round(normal_share, 4)})
    return out


RELATION_LABELS = {"r1": "order location", "r3": "delivery record",
                   "r6": "promotion", "r7": "coupon type", "r8": "sales stimulation"}


def run_hostel_test(rings, cfg: Config | None = None, relation: str = "r1",
                    clusters: list[dict] | None = None) -> dict:
    """How does the pipeline treat legitimate co-located groups?

    `relation` generalises the test beyond location: any of the five relations
    can hold a group that looks coordinated without being one (a promotion
    used by a real group-deal cohort is the r6 analogue of a hostel's address).

    Which co-located clusters exist depends only on `relation`, not on which
    rings are being checked against them - `clusters` lets a caller comparing
    several sets of rings against the same relation (see
    `run_hostel_test_all_relations`) find them once and reuse the list, rather
    than re-scanning the order file per arm.
    """
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()

    if clusters is None:
        clusters = find_colocated_clusters(cfg, labels, relation=relation)
    if not clusters:
        return {"clusters_found": 0, "relation": relation,
                "note": "no co-located normal clusters matched the criteria"}

    in_ring = np.zeros(labels.size, dtype=bool)
    for r in rings:
        in_ring[r.members] = True

    scores = np.zeros(labels.size)
    sp = proc / "scores_week2.parquet"
    if sp.exists():
        s = pq.read_table(sp)
        scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    # Relation diversity: how many different kinds of thing a group shares.
    # A hostel shares an address and little else; a ring tends to share the
    # address AND the promotion AND the coupon.
    #
    # Vectorised as one pass over the edge list rather than one masked scan
    # per cluster: with thousands of qualifying clusters on the denser
    # relations (13,546 for r6), a per-cluster O(edges) scan is a
    # cluster-count multiple of a full edge-array pass each time it runs, and
    # this function runs ten times over from `make lockstep`. Tagging every
    # account with which cluster (if any) it belongs to, then keeping only
    # edges where both ends carry the same cluster tag, does the same
    # selection in one pass regardless of how many clusters there are.
    cluster_of = np.full(labels.size, -1, dtype=np.int32)
    for i, c in enumerate(clusters):
        cluster_of[c["members"]] = i

    edges = pq.read_table(proc / "edges_week2_late.parquet",
                          columns=["src", "dst", "relation_mask"])
    esrc, edst = edges["src"].to_numpy(), edges["dst"].to_numpy()
    emask = edges["relation_mask"].to_numpy()

    cs, cd = cluster_of[esrc], cluster_of[edst]
    same = (cs == cd) & (cs >= 0)
    ce, cmask = cs[same], emask[same]
    order = np.argsort(ce, kind="stable")
    ce, cmask = ce[order], cmask[order]
    diversity_by_cluster = np.zeros(len(clusters), dtype=np.int64)
    edges_by_cluster = np.zeros(len(clusters), dtype=np.int64)
    if ce.size:
        starts = np.flatnonzero(np.concatenate(([True], ce[1:] != ce[:-1])))
        or_by_group = np.bitwise_or.reduceat(cmask.astype(np.int64), starts)
        count_by_group = np.diff(np.append(starts, ce.size))
        diversity_by_cluster[ce[starts]] = or_by_group
        edges_by_cluster[ce[starts]] = count_by_group

    flagged, clean = [], []
    for i, c in enumerate(clusters):
        m = c["members"]
        diversity = bin(int(diversity_by_cluster[i])).count("1")
        rec = {
            "entity": c["entity"], "size": c["size"],
            "labelled": c["labelled"], "normal_share": c["normal_share"],
            "members_in_a_ring": int(in_ring[m].sum()),
            "share_in_a_ring": round(float(in_ring[m].mean()), 4),
            "mean_score": round(float(scores[m].mean()), 4),
            "max_score": round(float(scores[m].max()), 4),
            "internal_edges": int(edges_by_cluster[i]),
            "relation_diversity": diversity,
        }
        (flagged if rec["members_in_a_ring"] > 0 else clean).append(rec)

    def avg(rows, key):
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else None

    return {
        "clusters_found": len(clusters),
        "accounts_in_clusters": int(sum(c["size"] for c in clusters)),
        "clusters_with_a_member_in_a_ring": len(flagged),
        "clusters_untouched": len(clean),
        "share_of_clusters_touched": round(len(flagged) / len(clusters), 4),
        "criteria": {
            "min_cluster_size": MIN_CLUSTER,
            "max_cluster_size": cfg.graph.n_max,
            "min_normal_share_of_labelled": MIN_NORMAL_SHARE,
            "relation": f"{relation} ({RELATION_LABELS.get(relation, relation)})",
        },
        "what_separates_them": {
            "flagged_mean_score": avg(flagged, "mean_score"),
            "untouched_mean_score": avg(clean, "mean_score"),
            "flagged_relation_diversity": avg(flagged, "relation_diversity"),
            "untouched_relation_diversity": avg(clean, "relation_diversity"),
            "flagged_internal_edges": avg(flagged, "internal_edges"),
            "untouched_internal_edges": avg(clean, "internal_edges"),
        },
        "worst_cases": sorted(flagged, key=lambda r: -r["share_in_a_ring"])[:10],
    }


def main() -> None:
    from eval.run_rings import load_edges, prune
    from orbweaver.rings.peel import extract_rings_batch

    cfg = load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()
    n = labels.size

    scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    ring_report = json.loads((proc / "ring_report.json").read_text())
    best = ring_report.get("best_cell", {"tau": 0.3, "lambda": 0.0})

    edges = prune(load_edges("late", cfg, n), scores, best["tau"])
    rings = extract_rings_batch(edges, scores, lambda_=best["lambda"],
                                k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                top_k=ring_report["graph"]["top_k"],
                                g_min=cfg.rings.g_min)
    result = run_hostel_test(rings, cfg)
    result["evaluated_at"] = best
    (proc / "hostel_test.json").write_text(json.dumps(result, indent=2, default=str))

    print(f"co-located normal clusters found: {result['clusters_found']:,} "
          f"({result['accounts_in_clusters']:,} accounts)")
    print(f"clusters with any member in a ring: "
          f"{result['clusters_with_a_member_in_a_ring']} "
          f"({result['share_of_clusters_touched']:.2%})")
    print("what separates them:")
    for k, v in result["what_separates_them"].items():
        print(f"  {k:34s} {v}")


if __name__ == "__main__":
    main()


def run_hostel_test_all_relations(cfg: Config, rings_by_arm: dict[str, list]) -> dict:
    """The crowd test on all five relations, for each graph arm supplied.

    `rings_by_arm` is e.g. `{"standard": [...], "lockstep": [...]}` - whatever
    arms the caller extracted rings for. Every relation is run against every
    arm at the SAME criteria (`MIN_CLUSTER`, `MIN_NORMAL_SHARE`), so the only
    thing that differs between two rows for the same relation is which rings
    were used to decide what counts as "touched". Which clusters exist is
    found once per relation and reused across every arm, since it does not
    depend on the rings at all.
    """
    proc = cfg.abs_path(cfg.paths.processed)
    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()

    out: dict[str, dict] = {}
    for rel in cfg.data.buildable_relations:
        clusters = find_colocated_clusters(cfg, labels, relation=rel)
        out[rel] = {arm: run_hostel_test(rings, cfg, relation=rel, clusters=clusters)
                   for arm, rings in rings_by_arm.items()}
    return out
