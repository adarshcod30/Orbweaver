"""Which offers are being farmed.

Every other output in this project is account-shaped: a score per account, a
ring of accounts. The person who owns the promotion budget thinks in
campaigns - which offer is leaking, how much, and since when - and Razorpay
issues promotions itself through its rewards marketplace, so that is the view
its own business would open first. Forter's own reporting frames coupon abuse
the same way: 81% of abuse attempts are serial abusers of *particular*
promotions, not a random cross-section of customers.

There is a second, structural reason this view matters. Ring recall is
0.0036 by construction - twenty-five rings surface a few hundred accounts out
of a base rate that is itself only 22% of a small labelled minority. One
farmed offer surfaces every account that redeemed it, which can be thousands.
The offer view does not replace the ring view; it is a review surface that
scales where rings cannot.

**The leakage score never sees a label.** It is built only from what a team
would know before checking anyone's fraud status - who redeemed the offer,
how many of them a graph-based ring already flagged, what the account scorer
thinks of them. The labelled fraud share is the evaluation target this is
checked against, computed separately and never fed back in; a test asserts
the scoring function's signature carries no label-derived argument at all.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

PROMO_RELATIONS = ("r6", "r7", "r8")
RELATION_LABELS = {"r6": "promotion", "r7": "coupon type", "r8": "sales stimulation"}
# A floor and a ceiling on what counts as a reportable offer, for the same
# reason build_graph.py caps entity size before it becomes an edge: a
# one-redeemer entity is noise (r8 alone has 5.2 million distinct codes, most
# used once), and at the other end r7 has a single value covering 96.4% of
# every account active in the scoring window - the coupon-type analogue of
# the "one coupon type shared by 97.5% of the user base" default the graph
# already documents, not a campaign anyone farmed.
MIN_REDEEMERS_TO_REPORT = 5
MAX_REDEEMER_SHARE = 0.10
BUDGETS = (10, 25, 50)
RANDOM_DRAWS = 200
# A confirmed-bad offer for the early-warning check: majority of its labelled
# redeemers are fraud by the window's end, the same 0.5 rule used everywhere
# else in this project an offer/ring is called "bad" rather than "mixed".
CONFIRMED_BAD_FRAUD_SHARE = 0.5
MIN_LABELLED_TO_CONFIRM = 3
# How far above the platform's own rate an offer's high-score share must sit
# before it counts as a warning rather than noise. Ten points was chosen
# because the day-one platform baseline itself moves by roughly that much
# night to night on this data (see `results.md`'s replay section); anything
# smaller is not distinguishable from ordinary day-to-day drift.
EARLY_WARNING_MARGIN = 0.10
MIN_REDEEMERS_FOR_WARNING = 3


# ------------------------------------------------------------- the table --

def entity_groups(uid: np.ndarray, ent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every (entity, account) pair once, grouped by entity. Uncapped - an
    offer used by five thousand accounts is exactly the case this view
    exists to surface, unlike the graph, which caps entity size because an
    uncapped entity would otherwise induce a near-complete subgraph."""
    keep = ~np.isnan(ent)
    e = ent[keep].astype(np.int64)
    u = uid[keep].astype(np.int64)
    order = np.lexsort((u, e))
    e, u = e[order], u[order]
    uniq = np.empty(e.size, dtype=bool)
    if e.size:
        uniq[0] = True
        np.logical_or(e[1:] != e[:-1], u[1:] != u[:-1], out=uniq[1:])
    e, u = e[uniq], u[uniq]
    starts = np.flatnonzero(np.concatenate(([True], e[1:] != e[:-1]))) if e.size else np.empty(0, np.int64)
    sizes = np.diff(np.append(starts, e.size)) if starts.size else np.empty(0, np.int64)
    return e[starts], starts, sizes, u


def build_offer_table(cfg: Config, split, scores: np.ndarray, ring_of: np.ndarray,
                      ring_ids_by_account: dict[int, set[int]]) -> tuple[list[dict], dict]:
    """One row per (relation, entity) surviving in the late scoring window,
    plus how many were excluded as too small to be a campaign or too large to
    be anything but a platform default, and why."""
    proc = cfg.abs_path(cfg.paths.processed)
    labels = split.labels
    orders = pq.read_table(proc / "orders_week2.parquet")
    day = orders["day_ordinal"].to_numpy()
    uid_all = orders["user_id"].to_numpy()

    from orbweaver.data.windows import LATE, week2_windows
    lo, hi = week2_windows(cfg)[LATE]
    inwin = (day >= lo) & (day <= hi)
    uid = uid_all[inwin]
    n_active = int(np.unique(uid).size)
    max_redeemers = int(MAX_REDEEMER_SHARE * n_active)

    basis = (f"redeemers x an assumed Rs.{cfg.cost.assumed_avg_promo_value_inr:.0f} per "
            "redemption - one order on the entity assumed per redeemer, the same "
            "convention every other rupee figure in this project uses.")

    # burst_z from the lockstep burstiness measure, when the entity is small
    # enough to have one - lockstep only defines burst_z for entities inside
    # the graph's own size cap, so a bigger offer's burst_z is honestly
    # absent rather than approximated.
    from orbweaver.data.lockstep import burst_z_table, first_arrival_groups
    burst_by_relation: dict[str, dict[int, float]] = {}
    for rel in PROMO_RELATIONS:
        ent_for_burst = orders[rel].to_numpy(zero_copy_only=False).astype(np.float64)[inwin]
        e_b, u_b, b_b, _ = first_arrival_groups(uid, ent_for_burst, day[inwin], n_max=cfg.graph.n_max)
        burst_by_relation[rel] = {}
        if e_b.size:
            zt = burst_z_table(e_b, b_b, lo, hi, seed=cfg.seed)
            burst_by_relation[rel] = dict(zip(zt["entity"].tolist(), zt["z"].tolist()))

    excluded_as_platform_default = {rel: 0 for rel in PROMO_RELATIONS}
    excluded_as_too_small = {rel: 0 for rel in PROMO_RELATIONS}
    rows = []
    for rel in PROMO_RELATIONS:
        ent_full = orders[rel].to_numpy(zero_copy_only=False).astype(np.float64)[inwin]
        ent_ids, starts, sizes, u_sorted = entity_groups(uid, ent_full)
        for entity, s, n in zip(ent_ids, starts, sizes):
            if n < MIN_REDEEMERS_TO_REPORT:
                excluded_as_too_small[rel] += 1
                continue
            if n > max_redeemers:
                excluded_as_platform_default[rel] += 1
                continue
            members = u_sorted[s:s + n]
            member_scores = scores[members]
            lab = labels[members]
            n_lab = int((lab != -1).sum())
            fraud_lab = int((lab == 1).sum())
            in_ring = ring_of[members] >= 0
            distinct_rings: set[int] = set()
            for m in members[in_ring]:
                distinct_rings |= ring_ids_by_account.get(int(m), set())
            rows.append({
                "relation": rel, "relation_label": RELATION_LABELS[rel],
                "entity": int(entity),
                "redeemers": int(n),
                "redeemers_in_a_ring": int(in_ring.sum()),
                "share_in_a_ring": round(float(in_ring.mean()), 6),
                "distinct_rings": len(distinct_rings),
                # Ring ranks, not member ids - lets the console link an offer
                # to the ring cards that touch it without persisting any
                # account's membership in this file.
                "ring_ranks": sorted(distinct_rings),
                "mean_score": round(float(member_scores.mean()), 6),
                "p90_score": round(float(np.percentile(member_scores, 90)), 6),
                "rupees_at_stake": round(n * cfg.cost.assumed_avg_promo_value_inr, 2),
                "rupees_at_stake_basis": basis,
                "burst_z": (round(float(burst_by_relation[rel][int(entity)]), 4)
                           if int(entity) in burst_by_relation[rel] else None),
                "labelled_redeemers": n_lab,
                "fraud_redeemers": fraud_lab,
                "fraud_share_among_labelled": round(fraud_lab / n_lab, 4) if n_lab else None,
                "members": members.tolist(),
            })
    exclusions = {
        "min_redeemers_to_report": MIN_REDEEMERS_TO_REPORT,
        "max_redeemer_share": MAX_REDEEMER_SHARE,
        "max_redeemers_this_window": max_redeemers,
        "active_accounts_in_window": n_active,
        "excluded_as_too_small": excluded_as_too_small,
        "excluded_as_platform_default": excluded_as_platform_default,
        "note": ("A one-redeemer entity is noise, not a campaign - r8 alone has "
                 "millions of distinct codes, most used once. At the other end, "
                 "an entity above 10% of every account active in the window is "
                 "the coupon-type analogue of the platform-wide default "
                 "build_graph.py already excludes from the graph, not something "
                 "anyone farmed."),
    }
    return rows, exclusions


# ---------------------------------------------------------- leakage score --

def leakage_score(n_redeemers: int, n_in_a_ring: int, mean_member_score: float) -> dict:
    """Two label-free rankings of an offer: how much of it a ring already
    caught, and what the account scorer thinks of it on average. Neither
    argument here can be derived from a label - `n_in_a_ring` comes from the
    deterministic peeling objective over scores, and `mean_member_score` is
    the scorer's own output - so this function has nothing to leak."""
    ring_share = n_in_a_ring / n_redeemers if n_redeemers else 0.0
    return {"ring_share": round(ring_share, 6), "mean_score": round(mean_member_score, 6)}


def rank_offers(offers: list[dict], by: str) -> list[dict]:
    if by == "ring_share":
        key = lambda o: (-o["_leak"]["ring_share"], -o["redeemers"], o["entity"])
    else:
        key = lambda o: (-o["_leak"]["mean_score"], -o["redeemers"], o["entity"])
    return sorted(offers, key=key)


# -------------------------------------------------------------- precision --

def _pooled_precision(offers: list[dict], labels: np.ndarray) -> dict:
    """Precision over the UNIQUE accounts across a set of offers, not offers
    averaged one at a time - an account redeeming two of the top-k offers
    must not be counted twice."""
    members = np.unique(np.concatenate([np.asarray(o["members"]) for o in offers])) \
        if offers else np.empty(0, dtype=np.int64)
    lab = labels[members]
    n_lab = int((lab != -1).sum())
    n_fraud = int((lab == 1).sum())
    return {"accounts": int(members.size), "labelled": n_lab, "fraud": n_fraud,
           "precision": round(n_fraud / n_lab, 4) if n_lab else None}


def precision_at_k(ranked: list[dict], labels: np.ndarray, base_rate: float,
                   budgets=BUDGETS, seed: int = 0) -> dict:
    """precision@k of a ranking, against the labelled base rate and against
    drawing k offers at random (RANDOM_DRAWS seeded draws, mean and range) -
    the same-size, same-shape comparison a raw base rate cannot give, since a
    random set of offers pools a different number of accounts than the
    top-k does."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in budgets:
        top = ranked[:k]
        pooled = _pooled_precision(top, labels)
        randoms = []
        if len(ranked) > k:
            for _ in range(RANDOM_DRAWS):
                idx = rng.choice(len(ranked), size=k, replace=False)
                randoms.append(_pooled_precision([ranked[i] for i in idx], labels))
        rp = [r["precision"] for r in randoms if r["precision"] is not None]
        out[str(k)] = {
            "leakage_ranked": pooled,
            "vs_base_rate": round(pooled["precision"] - base_rate, 4) if pooled["precision"] is not None else None,
            "random_offers": {"mean": round(float(np.mean(rp)), 4) if rp else None,
                              "min": round(float(np.min(rp)), 4) if rp else None,
                              "max": round(float(np.max(rp)), 4) if rp else None,
                              "draws": len(rp)},
        }
    return out


# ---------------------------------------------------------------- coverage --

def coverage_curve(ranked: list[dict], labels: np.ndarray, ring_recall: float | None,
                   max_k: int = 50) -> dict:
    """Cumulative share of ALL labelled fraud accounts covered by the top-k
    offers, and how many legitimate accounts arrive with them - monotone in
    k by construction, since it is a running union rather than k independent
    samples."""
    total_fraud = int((labels == 1).sum())
    seen = np.zeros(labels.size, dtype=bool)
    points = []
    for k, o in enumerate(ranked[:max_k], start=1):
        seen[np.asarray(o["members"])] = True
        covered = seen & (labels == 1)
        legit = seen & (labels == 0)
        points.append({"k": k, "fraud_covered": int(covered.sum()),
                       "fraud_coverage": round(covered.sum() / total_fraud, 4) if total_fraud else None,
                       "legitimate_accounts_swept_in": int(legit.sum()),
                       "accounts_reviewed": int(seen.sum())})
    return {"total_labelled_fraud": total_fraud, "ring_recall_reference": ring_recall,
           "points": points}


# ------------------------------------------------------------ early warning --

def _night_scores(cfg: Config, lo: int, d: int, hi: int, scores_of) -> tuple[np.ndarray, int]:
    """Scores as of night `d`. Night `hi` (the standard late window) is
    already on disk and permanent; earlier nights are rebuilt the same way
    `eval.replay` does, since replay deletes its own intermediates once its
    numbers are written."""
    proc = cfg.abs_path(cfg.paths.processed)
    if d == hi:
        n = int(pq.read_table(proc / "nodes.parquet").num_rows)
        scores = np.zeros(n)
        t = pq.read_table(proc / "scores_week2.parquet")
        scores[t["user_id"].to_numpy()] = t["score"].to_numpy()
        return scores, n

    from orbweaver.data.build_graph import build_graph
    from orbweaver.features.node_features import build_features
    from orbweaver.scoring.xgb_graph import load_features

    tag = f"late_upto_{d}"
    build_graph(2, cfg, days=(lo, d), tag=tag, force=True)
    build_features(2, cfg, days=(lo, d), tag=tag, force=True)
    n = int(pq.read_table(proc / "nodes.parquet").num_rows)
    scores = scores_of(load_features(2, cfg, n, tag))
    for name in (f"edges_week2_{tag}.parquet", f"features_week2_{tag}.parquet"):
        p = proc / name
        if p.exists():
            p.unlink()
    return scores, n


def early_warning(cfg: Config, offers: list[dict], labels: np.ndarray) -> dict:
    """For each confirmed-bad offer, the first night its redeemers' high-score
    share clears the platform's own baseline by EARLY_WARNING_MARGIN."""
    from orbweaver.data.windows import LATE, week2_windows
    from orbweaver.scoring.xgb_graph import load_scorer, score_features

    lo, hi = week2_windows(cfg)[LATE]
    days = list(range(lo, hi + 1))
    model, calibrate = load_scorer(cfg)
    scores_of = lambda X: score_features(model, calibrate, X)
    tau = cfg.rings.prune_tau_headline

    orders = pq.read_table(cfg.abs_path(cfg.paths.processed) / "orders_week2.parquet")
    day = orders["day_ordinal"].to_numpy()
    uid_all = orders["user_id"].to_numpy()

    confirmed = [o for o in offers if o["labelled_redeemers"] >= MIN_LABELLED_TO_CONFIRM
                and (o["fraud_share_among_labelled"] or 0) >= CONFIRMED_BAD_FRAUD_SHARE]

    by_night = []
    warned_by_offer: dict[tuple[str, int], dict] = {(o["relation"], o["entity"]): {"crossed_on_night": None}
                                                     for o in confirmed}
    for i, d in enumerate(days, start=1):
        scores, n = _night_scores(cfg, lo, d, hi, scores_of)
        active = scores > 0
        baseline = float((scores[active] > tau).mean()) if active.any() else 0.0
        inwin = (day >= lo) & (day <= d)
        uid_so_far = uid_all[inwin]
        cols = {rel: orders[rel].to_numpy(zero_copy_only=False)[inwin] for rel in PROMO_RELATIONS}

        night_rows = []
        for o in confirmed:
            key = (o["relation"], o["entity"])
            ent_col = cols[o["relation"]]
            redeemers = np.unique(uid_so_far[ent_col == o["entity"]])
            share_above = float((scores[redeemers] > tau).mean()) if redeemers.size else 0.0
            crossed = (redeemers.size >= MIN_REDEEMERS_FOR_WARNING
                      and (share_above - baseline) >= EARLY_WARNING_MARGIN)
            if crossed and warned_by_offer[key]["crossed_on_night"] is None:
                warned_by_offer[key]["crossed_on_night"] = i
            night_rows.append({"relation": o["relation"], "entity": o["entity"],
                               "redeemers_so_far": int(redeemers.size),
                               "share_above_tau": round(share_above, 4)})
        by_night.append({"night": i, "day": d, "platform_baseline_above_tau": round(baseline, 4),
                         "offers": night_rows})

    lead = []
    for o in confirmed:
        key = (o["relation"], o["entity"])
        night = warned_by_offer[key]["crossed_on_night"]
        lead.append({"relation": o["relation"], "entity": o["entity"],
                    "labelled_redeemers": o["labelled_redeemers"],
                    "fraud_share_among_labelled": o["fraud_share_among_labelled"],
                    "crossed_on_night": night,
                    "nights_of_lead_before_the_window_closed": (len(days) - night) if night else 0})

    warned = [l for l in lead if l["crossed_on_night"] is not None]
    return {
        "margin": EARLY_WARNING_MARGIN, "tau": tau,
        "confirmed_bad_offers": len(confirmed),
        "ever_warned": len(warned),
        "share_ever_warned": round(len(warned) / len(confirmed), 4) if confirmed else None,
        "median_night_of_first_warning": float(np.median([l["crossed_on_night"] for l in warned])) if warned else None,
        "by_night": by_night, "per_offer": lead,
    }


# --------------------------------------------------------------- the run --

def run(cfg: Config | None = None, with_early_warning: bool = True) -> dict:
    from eval.split import make_split
    from orbweaver.config import load_config as _lc

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    labels = split.labels
    n = labels.size

    scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    ring_report = json.loads((proc / "ring_report.json").read_text())
    ring_of = np.full(n, -1, dtype=np.int32)
    ring_ids_by_account: dict[int, set[int]] = {}
    for c in ring_report.get("case_files", []):
        for m in c["members"]:
            ring_of[m] = c["rank"]
            ring_ids_by_account.setdefault(m, set()).add(c["rank"])

    print("building the offer table over r6/r7/r8 in the scoring window ...", flush=True)
    offers, exclusions = build_offer_table(cfg, split, scores, ring_of, ring_ids_by_account)
    for o in offers:
        o["_leak"] = leakage_score(o["redeemers"], o["redeemers_in_a_ring"], o["mean_score"])
    print(f"  {len(offers):,} offers across r6/r7/r8")
    print(f"  excluded as too small (<{exclusions['min_redeemers_to_report']} redeemers): "
          f"{exclusions['excluded_as_too_small']}")
    print(f"  excluded as a platform default (>{exclusions['max_redeemers_this_window']:,} "
          f"redeemers): {exclusions['excluded_as_platform_default']}")

    base_rate = float((labels[labels != -1] == 1).mean())
    ranked_ring = rank_offers(offers, "ring_share")
    ranked_mean = rank_offers(offers, "mean_score")

    prec_ring = precision_at_k(ranked_ring, labels, base_rate, seed=cfg.seed)
    prec_mean = precision_at_k(ranked_mean, labels, base_rate, seed=cfg.seed + 1)
    for k, row in prec_ring.items():
        print(f"  precision@{k} (ring-share ranking): {row['leakage_ranked']['precision']}  "
              f"vs base {base_rate:.4f}  vs random {row['random_offers']['mean']}")

    ring_recall = ring_report.get("best_cell", {})
    cell = ring_report.get("grid", {}).get(f"tau={ring_recall.get('tau')},lambda={ring_recall.get('lambda')}", {})
    ring_recall_ref = cell.get("ring_recall")
    coverage_by_leakage = coverage_curve(ranked_mean, labels, ring_recall_ref)
    # Leakage ranks small, concentrated offers first - exactly what makes it
    # precise also caps how much of total fraud it can ever touch at a given
    # k. Size answers a different question: how much of the recall ceiling
    # can be beaten at all, at the cost of reviewing far more accounts.
    ranked_by_size = sorted(offers, key=lambda o: (-o["redeemers"], o["entity"]))
    coverage_by_size = coverage_curve(ranked_by_size, labels, ring_recall_ref)
    coverage = {"total_labelled_fraud": coverage_by_leakage["total_labelled_fraud"],
               "ring_recall_reference": ring_recall_ref,
               "by_leakage_mean_score": coverage_by_leakage["points"],
               "by_redeemer_count": coverage_by_size["points"]}

    warn = None
    if with_early_warning:
        print("replaying nights 1-3 for early warning (night 4 already on disk) ...", flush=True)
        warn = early_warning(cfg, offers, labels)
        print(f"  {warn['ever_warned']} of {warn['confirmed_bad_offers']} confirmed-bad offers "
              f"warned before the window closed")

    # Persist enough of each ranking that the console's own re-sort by
    # leakage is correct, not just the biggest campaigns. The highest-leakage
    # offers are typically small (a handful of redeemers, nearly all in a
    # ring) - keeping only the top offers by redeemer count would silently
    # drop exactly the offers a leakage-sorted queue should lead with.
    PERSIST_TOP_N = 300
    by_redeemers = sorted(offers, key=lambda o: -o["redeemers"])[:PERSIST_TOP_N]
    by_ring_share = ranked_ring[:PERSIST_TOP_N]
    by_mean = ranked_mean[:PERSIST_TOP_N]
    seen_keys = set()
    persisted = []
    for o in by_redeemers + by_ring_share + by_mean:
        key = (o["relation"], o["entity"])
        if key not in seen_keys:
            seen_keys.add(key)
            persisted.append(o)
    persisted.sort(key=lambda o: -o["redeemers"])

    for o in offers:
        o.pop("members", None)  # kept only for the ranking/coverage computation above

    return {
        "method": ("Per-entity aggregates over the promotion-like relations "
                  "(r6, r7, r8) in the late scoring window, uncapped - unlike "
                  "the graph, an offer used by thousands of accounts is exactly "
                  "the case this view exists to surface. The leakage score uses "
                  "no label: the share of redeemers a ring already flagged, and "
                  "the account scorer's mean opinion of them."),
        "base_rate": round(base_rate, 4),
        "n_offers": len(offers),
        "exclusions": exclusions,
        "offers": persisted,
        "precision_at_k": {"by_ring_share": prec_ring, "by_mean_score": prec_mean},
        "coverage": coverage,
        "early_warning": warn,
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "offers.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
