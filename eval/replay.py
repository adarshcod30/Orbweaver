"""If this had been running nightly, when would it have seen each ring?

Everything else in this project is forensic. It takes a window that has already
happened, builds the graph, and finds the rings that were in it. That is a fair
way to measure whether the method works, and it is not how the thing would be
used. A team running this would see one more day of orders every night and want
to know, tonight, which groups are worth looking at.

So this replays the scoring window one day at a time. For each day `d` it
rebuilds the graph and the features from only the days up to `d`, applies the
model **that was already fitted on the earlier window** - nothing is refitted,
because a system running on the 25th cannot use a model trained on the 28th -
and peels at the standard operating point.

Two things the replay makes it possible to say, neither of which the static
result can:

- **days to detection**: for each ring the last day finds, the first night on
  which a recognisably similar group was already visible.
- **caught before it was spent**: the share of a ring's promotion spend in the
  window that fell on or after the night it first surfaced, which is the part
  an intervention could have stopped.

Rarity is recomputed for every prefix rather than carried over. Fewer accounts
have been seen on day one, so entities look rarer, and that is genuinely what a
system would have measured that night.

The last prefix is the whole window, so it must reproduce the standard result
exactly - the same rings, the same precision. That is asserted in the tests
rather than hoped for.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import date

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.data.build_graph import build_graph
from orbweaver.data.windows import EARLY, LATE, week2_windows
from orbweaver.features.node_features import build_features
from orbweaver.rings.peel import EdgeList, extract_rings_batch

# Two rings are "the same group seen on a different night" at this overlap.
# Half the members is a deliberately loose bar: a ring that is still growing
# will not match its final self exactly, and insisting that it did would report
# every ring as detected only on the last day.
MATCH_JACCARD = 0.5


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def promo_spend_by_day(cfg: Config, members: np.ndarray,
                       lo: int, hi: int) -> dict[int, float]:
    """Promotion spend per day for one group, on the stated assumption."""
    proc = cfg.abs_path(cfg.paths.processed)
    t = pq.read_table(proc / "orders_week2.parquet",
                      columns=["user_id", "day_ordinal", "r6"])
    uid = t["user_id"].to_numpy()
    day = t["day_ordinal"].to_numpy()
    promo = t["r6"].to_numpy(zero_copy_only=False)
    inside = np.zeros(int(uid.max()) + 1, dtype=bool)
    inside[members] = True
    m = inside[uid] & ~np.isnan(promo) & (day >= lo) & (day <= hi)
    out: dict[int, float] = {}
    for d in range(lo, hi + 1):
        n = int((m & (day == d)).sum())
        out[d] = round(n * cfg.cost.assumed_avg_promo_value_inr, 2)
    return out


def snapshot(cfg: Config, lo: int, d: int, scores_of, top_k: int) -> dict:
    """One night: build what is visible, score it, peel it, then clean up."""
    from orbweaver.scoring.xgb_graph import load_features

    proc = cfg.abs_path(cfg.paths.processed)
    tag = f"late_upto_{d}"
    t0 = time.time()

    build_graph(2, cfg, days=(lo, d), tag=tag, force=True)
    build_features(2, cfg, days=(lo, d), tag=tag, force=True)
    t_build = time.time() - t0

    n = int(pq.read_table(proc / "nodes.parquet").num_rows)
    t1 = time.time()
    scores = scores_of(load_features(2, cfg, n, tag))
    t_score = time.time() - t1

    e = pq.read_table(proc / f"edges_week2_{tag}.parquet",
                      columns=["src", "dst", "weight"])
    edges = EdgeList(e["src"].to_numpy().astype(np.int64),
                     e["dst"].to_numpy().astype(np.int64),
                     e["weight"].to_numpy().astype(np.float64), n)
    keep = scores > cfg.rings.prune_tau_headline
    m = keep[edges.src] & keep[edges.dst]
    pruned = EdgeList(edges.src[m], edges.dst[m], edges.weight[m], n)

    t2 = time.time()
    rings = extract_rings_batch(pruned, scores.astype(np.float64),
                                lambda_=cfg.rings.lambda_headline,
                                k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                top_k=top_k, g_min=cfg.rings.g_min)
    t_peel = time.time() - t2

    return {
        "day": d,
        "edges": int(edges.src.size),
        "edges_after_prune": int(pruned.src.size),
        "accounts_scored": int((scores > 0).sum()),
        "rings": rings,
        "scores": scores,
        "seconds": {"build": round(t_build, 1), "score": round(t_score, 1),
                    "peel": round(t_peel, 1),
                    "total": round(time.time() - t0, 1)},
    }


def cleanup(cfg: Config, d: int) -> int:
    """Delete a night's graph and features; keep the manifests. Returns bytes."""
    proc = cfg.abs_path(cfg.paths.processed)
    freed = 0
    for name in (f"edges_week2_late_upto_{d}.parquet",
                 f"features_week2_late_upto_{d}.parquet"):
        p = proc / name
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
    return freed


def run(cfg: Config | None = None, top_k: int | None = None) -> dict:
    from eval.split import make_split
    from orbweaver.rings.cost import evaluate_rings
    from eval.metrics import ltv_proxy
    from orbweaver.scoring.xgb_graph import load_scorer, score_features

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    top_k = top_k or cfg.rings.top_k
    split = make_split(cfg)
    labels = split.labels
    n = labels.size

    model, calibrate = load_scorer(cfg)
    scores_of = lambda X: score_features(model, calibrate, X)

    lo, hi = week2_windows(cfg)[LATE]
    days = list(range(lo, hi + 1))

    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    free_before = shutil.disk_usage(proc).free
    snapshots, by_day = [], {}
    print(f"{'night':>18s} {'edges':>12s} {'rings':>6s} {'accounts':>9s} "
          f"{'precision':>10s} {'per catch':>10s} {'build':>7s} {'peel':>6s}")
    for i, d in enumerate(days, start=1):
        snap = snapshot(cfg, lo, d, scores_of, top_k)
        rings = snap["rings"]
        block = evaluate_rings(rings, labels, ltv, restrict_to=split.test) if rings else {}
        row = {
            "day": d,
            "date": date.fromordinal(d).isoformat(),
            "days_of_data": i,
            "edges": snap["edges"],
            "edges_after_prune": snap["edges_after_prune"],
            "n_rings": len(rings),
            "accounts_in_rings": block.get("accounts_in_rings", 0),
            "ring_precision": block.get("ring_precision"),
            "precision_lift_over_base": block.get("precision_lift_over_base"),
            "normal_flagged_per_fraud_caught": block.get("normal_flagged_per_fraud_caught"),
            "fraud_members": block.get("fraud_members"),
            "seconds": snap["seconds"],
        }
        by_day[d] = {"rings": [r.members for r in rings], "row": row}
        snapshots.append(row)
        print(f"{row['date']:>18s} {row['edges']:>12,} {row['n_rings']:>6} "
              f"{row['accounts_in_rings']:>9,} {str(row['ring_precision']):>10} "
              f"{str(row['normal_flagged_per_fraud_caught']):>10} "
              f"{snap['seconds']['build']:>7} {snap['seconds']['peel']:>6}")
        freed = cleanup(cfg, d)
        row["freed_bytes"] = freed

    # Two ways of asking when a ring was first visible, because the first one
    # alone turned out to say almost nothing.
    #
    # Ring matching asks whether a recognisably similar *group* existed on an
    # earlier night. That is the natural question and it is brittle: a ring
    # that is still accumulating members does not resemble its final self, so a
    # fixed overlap threshold can report that nothing was ever seen early even
    # when most of the members were already being surfaced.
    #
    # So the second measure drops group identity entirely and asks, of the
    # accounts in a final ring, how many were already inside *some* ring on an
    # earlier night. That is what an analyst working the queue would actually
    # have had in front of them, and it needs no matching at all.
    final_day = days[-1]
    final_rings = by_day[final_day]["rings"]
    surfaced_by_day = {d: (np.unique(np.concatenate(by_day[d]["rings"]))
                           if by_day[d]["rings"] else np.empty(0, dtype=np.int64))
                       for d in days}

    detection = []
    for rank, members in enumerate(final_rings, start=1):
        overlap_by_day, member_share_by_day = {}, {}
        for d in days:
            best = max((jaccard(members, m) for m in by_day[d]["rings"]), default=0.0)
            overlap_by_day[str(d)] = round(best, 3)
            seen = np.isin(members, surfaced_by_day[d])
            member_share_by_day[str(d)] = round(float(seen.mean()), 4)

        matched = [d for d in days if overlap_by_day[str(d)] >= MATCH_JACCARD]
        first_match = matched[0] if matched else final_day
        # First night on which at least half this ring's members were already
        # inside some surfaced ring.
        half = [d for d in days if member_share_by_day[str(d)] >= 0.5]
        first_half = half[0] if half else final_day

        spend = promo_spend_by_day(cfg, members, lo, hi)
        total = sum(spend.values())
        after_match = sum(v for k, v in spend.items() if k >= first_match)
        after_half = sum(v for k, v in spend.items() if k >= first_half)
        lab = labels[members]
        detection.append({
            "rank": rank,
            "size": int(members.size),
            "fraud": int((lab == 1).sum()),
            "normal": int((lab == 0).sum()),
            "best_ring_overlap_by_day": overlap_by_day,
            "member_share_already_surfaced_by_day": member_share_by_day,
            "first_matched_day": int(first_match),
            "days_to_detection": int(first_match - lo) + 1,
            "matched_before_last_night": bool(matched and matched[0] < final_day),
            "first_day_half_the_members_surfaced": int(first_half),
            "days_to_half_the_members": int(first_half - lo) + 1,
            "window_spend_inr": round(total, 2),
            "spend_on_or_after_detection_inr": round(after_match, 2),
            "spend_on_or_after_half_surfaced_inr": round(after_half, 2),
            "share_still_ahead_at_detection": round(after_match / total, 4) if total else None,
            "share_still_ahead_at_half_surfaced": round(after_half / total, 4) if total else None,
        })

    dtd = [r["days_to_detection"] for r in detection]
    dth = [r["days_to_half_the_members"] for r in detection]
    total_spend = sum(r["window_spend_inr"] for r in detection)
    total_after = sum(r["spend_on_or_after_detection_inr"] for r in detection)
    total_after_half = sum(r["spend_on_or_after_half_surfaced_inr"] for r in detection)
    best_overlap_before_last = [
        max((v for k, v in r["best_ring_overlap_by_day"].items()
             if int(k) < days[-1]), default=0.0) for r in detection]

    free_after = shutil.disk_usage(proc).free
    return {
        "operating_point": {"tau": cfg.rings.prune_tau_headline,
                            "lambda": cfg.rings.lambda_headline,
                            "top_k": top_k},
        "window": {"first_day": date.fromordinal(lo).isoformat(),
                   "last_day": date.fromordinal(hi).isoformat(),
                   "days": len(days)},
        "method": ("Each night rebuilds the graph and features from the days up "
                   "to that night only, applies the model already fitted on the "
                   "earlier window - nothing is refitted - and peels at the "
                   "standard operating point. Rarity is recomputed per night, "
                   "because fewer accounts have been seen."),
        "match_jaccard": MATCH_JACCARD,
        "snapshots": snapshots,
        "detection": detection,
        "summary": {
            "rings_on_the_last_night": len(detection),
            "days_to_detection": {
                "min": int(min(dtd)) if dtd else None,
                "median": float(np.median(dtd)) if dtd else None,
                "max": int(max(dtd)) if dtd else None,
                "share_seen_before_the_last_night": round(
                    float(np.mean([d < len(days) for d in dtd])), 4) if dtd else None,
            },
            "best_ring_overlap_before_the_last_night": {
                "median": round(float(np.median(best_overlap_before_last)), 3)
                if best_overlap_before_last else None,
                "max": round(float(max(best_overlap_before_last)), 3)
                if best_overlap_before_last else None,
                "threshold_required": MATCH_JACCARD,
            },
            "days_to_half_the_members_surfaced": {
                "median": float(np.median(dth)) if dth else None,
                "min": int(min(dth)) if dth else None,
                "share_before_the_last_night": round(
                    float(np.mean([d < len(days) for d in dth])), 4) if dth else None,
            },
            "share_of_ring_spend_still_ahead_at_detection": round(
                total_after / total_spend, 4) if total_spend else None,
            "share_of_ring_spend_still_ahead_when_half_surfaced": round(
                total_after_half / total_spend, 4) if total_spend else None,
            "total_window_spend_inr": round(total_spend, 2),
            "spend_on_or_after_detection_inr": round(total_after, 2),
            "spend_on_or_after_half_surfaced_inr": round(total_after_half, 2),
            "seconds_per_night": {
                "median_total": float(np.median([s["seconds"]["total"] for s in snapshots])),
                "max_total": float(max(s["seconds"]["total"] for s in snapshots)),
            },
        },
        "disk": {"free_before_bytes": free_before, "free_after_bytes": free_after,
                 "note": "each night's graph and features are deleted once its "
                         "numbers are written; manifests are kept"},
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "replay.json"
    dest.write_text(json.dumps(out, indent=2))
    s = out["summary"]
    print()
    print(f"rings on the last night: {s['rings_on_the_last_night']}")
    print(f"days to detection: median {s['days_to_detection']['median']}, "
          f"range {s['days_to_detection']['min']}-{s['days_to_detection']['max']}")
    ov = s["best_ring_overlap_before_the_last_night"]
    print(f"ring matched before the last night: "
          f"{s['days_to_detection']['share_seen_before_the_last_night']:.1%} "
          f"(best overlap reached: median {ov['median']}, max {ov['max']}, "
          f"needed {ov['threshold_required']})")
    h = s["days_to_half_the_members_surfaced"]
    print(f"half the members already surfaced: median night {h['median']}, "
          f"{h['share_before_the_last_night']:.1%} before the last night")
    print(f"ring spend still ahead at that point: "
          f"{s['share_of_ring_spend_still_ahead_when_half_surfaced']:.1%}")
    print(f"ring spend still ahead when first seen: "
          f"{s['share_of_ring_spend_still_ahead_at_detection']:.1%} "
          f"(Rs {s['spend_on_or_after_detection_inr']:,.0f} of "
          f"Rs {s['total_window_spend_inr']:,.0f})")
    print(f"median seconds per night: {s['seconds_per_night']['median_total']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
