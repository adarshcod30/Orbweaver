"""Telling a crowd from a ring by when it formed.

On the processor graph the billing address is at once the strongest relation
(5.42x fraud lift) and the thing that legitimately ties every card in a
building together; `docs/results.md` says the two "cannot be separated by
weighting, because the weighting is what discovered the address was
informative in the first place", and 4 of 7 apartment clusters were touched.
On PPA the same shape appears in the case files: rings held together by one
sales-stimulation entity shared by dozens of accounts, most unreviewed -
indistinguishable from a legitimate group-deal cohort. Rarity cannot separate
them: a hostel's address and a ring's address are equally rare.

The published discriminator is time, not rarity. CopyCatch (Beutel, Xu,
Guruswami, Palow, Faloutsos, WWW 2013, deployed at Facebook) defines lockstep
behaviour as a group acting on the same objects inside a narrow window: natural
popularity spreads across time, coordination clusters in it. HoloScope (Liu,
Hooi, Faloutsos, CIKM 2017) folds temporal spikes into a topology-based
suspiciousness score. SliceNDice (Nilforoshan and Shah, ICDM 2019 - 89%
precision on Snapchat advertiser rings) scores multi-attribute groups against
what non-coordinated behaviour would produce. This module measures, per
entity, how concentrated in time its members' first arrivals are, corrects
that for the entity's size against a simulated null, and turns the excess into
a second, separate edge weight - never the one every other number here rests
on.

**The null is the part that has to be right.** A two-account entity is
trivially "bursty" - both members can only have arrived on at most two
distinct days, so a high raw concentration is guaranteed by size alone and
means nothing. So burstiness is measured against what independent arrivals of
the *same size*, drawn from the platform's own daily activity, would produce:
`burst_z(e) = (observed - null_mean) / null_sd`, simulated once per size
bucket. An entity's burstiness is its excess over what its size alone would
produce, not its raw concentration.

**Fitted, not chosen**, exactly like `relation_weights.py`: bin entities by
`(relation, burst_z quartile)`, and for each bin measure fraud-fraud lift on
training-pool accounts only. A bin whose lift comes out near 1.0 is reported
as such - the honest outcome may be that time adds nothing on PPA, where a
burst is a day, the finest resolution the released data has. IEEE-CIS carries
the sharper claim: `TransactionDT` is seconds over six months, so hour- and
day-scale windows are meaningful there in a way a day-resolution dataset
cannot support.

**Default off, and structurally incapable of moving anything.** This module
never touches `orbweaver/data/build_graph.py`. `make lockstep` builds its own
second parquet, `edges_week2_late_lockstep.parquet`, and runs the standard
extractor on it. `config/default.yaml` carries `graph.time_weighting: false`
as the stated intent this module reports against; no code path in the
existing pipeline reads it, which is the strongest form of "today's numbers
cannot move" available - not a flag that is checked and happens to be off, but
a second file nothing else imports.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.data.build_graph import pairs_from_groups, rarity_weight

FRAUD, NORMAL = 1, 0

# |W| = 1 day is primary on PPA (day resolution is all the data has); |W| = 2
# is reported beside it. IEEE-CIS's arm uses its own, finer windows (see
# `ieee_burst_windows`).
BURST_WINDOWS = (1, 2)
PRIMARY_WINDOW = 1

# (lo, hi) inclusive. An entity below 2 or above n_max induces no edge at all,
# so buckets start at 2 and stop at n_max (100).
SIZE_BUCKETS = [(2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
N_NULL_DRAWS = 10_000
N_QUARTILES = 4
# Bins are finer than a whole relation (quartile of a relation's entities
# rather than the relation itself), so there is less data per bin than
# `relation_weights.MIN_LABELLED_EDGES` assumes. Documented, not copied.
MIN_LABELLED_PAIRS_PER_BIN = 300


def size_bucket(size: np.ndarray) -> np.ndarray:
    """Bucket index 0..5 for each size, or -1 outside [2, 100]."""
    out = np.full(size.shape, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(SIZE_BUCKETS):
        out[(size >= lo) & (size <= hi)] = i
    return out


def daily_share(bin_idx: np.ndarray, bin_lo: int, bin_hi: int) -> np.ndarray:
    """p_b = orders in bin b / orders in the window, over every bin in range.

    This is "the platform's daily activity" the spec asks for: every order in
    the window counts, not only orders that carry the relation being tested,
    because the null is a statement about when the platform is busy, not about
    any one relation.
    """
    n_bins = bin_hi - bin_lo + 1
    counts = np.bincount(bin_idx - bin_lo, minlength=n_bins)[:n_bins].astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.full(n_bins, 1.0 / n_bins)


# --------------------------------------------------------- entity arrivals --

def first_arrival_groups(user_id: np.ndarray, entity: np.ndarray,
                         bin_idx: np.ndarray, *, n_max: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One row per (entity, account): that account's first bin on that entity.

    Entities outside [2, n_max] are dropped before the caller ever sees them -
    they induce no edge, so their burstiness is not evidence of anything a
    graph could act on.

    Returns (entity_ids, account_ids, first_bin, group starts/sizes are
    recoverable from entity_ids being sorted and contiguous).
    """
    keep = ~np.isnan(entity) if entity.dtype.kind == "f" else np.ones(entity.shape, bool)
    e = entity[keep].astype(np.int64)
    u = user_id[keep].astype(np.int64)
    b = bin_idx[keep].astype(np.int64)

    # Sort by (entity, account, bin) so the first row of each (entity,account)
    # run is that account's minimum - i.e. first - bin on that entity.
    order = np.lexsort((b, u, e))
    e, u, b = e[order], u[order], b[order]
    first = np.empty(e.size, dtype=bool)
    first[0] = True
    np.logical_or(e[1:] != e[:-1], u[1:] != u[:-1], out=first[1:])
    e, u, b = e[first], u[first], b[first]

    starts = np.flatnonzero(np.concatenate(([True], e[1:] != e[:-1])))
    sizes = np.diff(np.append(starts, e.size))
    sel = (sizes >= 2) & (sizes <= n_max)
    keep_rows = np.zeros(e.size, dtype=bool)
    for s, n in zip(starts[sel], sizes[sel]):
        keep_rows[s:s + n] = True
    return e[keep_rows], u[keep_rows], b[keep_rows], sel


# -------------------------------------------------------------- burst(e) --

def burst_for_entities(entity: np.ndarray, first_bin: np.ndarray,
                       bin_lo: int, bin_hi: int) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Per-entity, per-window burst statistics, vectorised over a sparse table.

    For |W| = 1 the answer is the busiest single bin's share of the entity.
    For |W| = 2, the true maximum over every integer window [b, b+1] is
    achieved at some window whose *first* bin has a nonzero count - a window
    starting at an empty bin can only tie a window starting one bin later,
    never beat it - so only nonzero rows need to be evaluated, and the whole
    computation stays sparse even when the bin range is thousands wide (as it
    is for IEEE-CIS at hourly resolution).
    """
    order = np.lexsort((first_bin, entity))
    e, b = entity[order], first_bin[order]
    starts = np.flatnonzero(np.concatenate(([True], e[1:] != e[:-1])))
    ent_ids = e[starts]
    sizes = np.diff(np.append(starts, e.size))

    # Per (entity, bin) arrival counts - one row per distinct pair, already
    # sorted by (entity, bin) since b was the secondary sort key within each
    # entity run.
    key = np.empty(e.size, dtype=bool)
    key[0] = True
    np.logical_or(e[1:] != e[:-1], b[1:] != b[:-1], out=key[1:])
    pe, pb = e[key], b[key]
    pcount = np.diff(np.flatnonzero(np.append(key, True)))

    burst1 = np.zeros(ent_ids.size)
    np.maximum.at(burst1, np.searchsorted(ent_ids, pe), pcount)
    burst1 = burst1 / sizes.astype(np.float64)

    # w=2: for every (entity,bin) row, does (entity,bin+1) also exist? Look it
    # up by exact match on the sorted key, since a plain +1 search would find
    # the wrong entity's row once we cross an entity boundary.
    big = int(bin_hi - bin_lo + 2)
    pk = pe * big + (pb - bin_lo)
    nxt = pe * big + (pb - bin_lo + 1)
    pos = np.searchsorted(pk, nxt)
    safe = np.minimum(pos, pk.size - 1)
    hit = (pos < pk.size) & (pk[safe] == nxt)
    nxt_count = np.where(hit, pcount[safe], 0)
    w2_cand = pcount + nxt_count
    burst2 = np.zeros(ent_ids.size)
    np.maximum.at(burst2, np.searchsorted(ent_ids, pe), w2_cand)
    burst2 = burst2 / sizes.astype(np.float64)

    return {"entity": ent_ids, "size": sizes, 1: burst1, 2: burst2}


# ------------------------------------------------------------- null model --

def simulate_null_by_size(sizes: np.ndarray, bin_lo: int, bin_hi: int,
                          p: np.ndarray, *, draws_per_size: int, seed: int
                          ) -> dict[int, dict[int, tuple[float, float]]]:
    """Null mean/sd of burst(w), one distribution **per exact size**.

    Bucketing sizes and drawing a null size uniformly across the bucket looks
    like it satisfies "10,000 draws per size bucket", but it is biased: a
    size-2 entity and a size-3 entity have genuinely different null means
    (measured on this module's own null: 0.561 vs 0.453 at 8 bins), so a null
    built by mixing them sits between the two, and a real size-2 entity then
    reads as falsely bursty against a null that is, on average, less bursty
    than size-2 alone would be. Comparing every entity against a null of its
    own exact size removes this by construction. All distinct sizes present
    are simulated in one batched call rather than one Python loop per size,
    so this costs about the same as the bucketed version despite simulating
    up to `n_max` sizes instead of six.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(sizes)
    n_bins = bin_hi - bin_lo + 1
    max_size = int(uniq.max())

    reps = np.repeat(uniq, draws_per_size)
    n_draws = reps.size
    bins = rng.choice(n_bins, size=(n_draws, max_size), p=p)
    valid = np.arange(max_size)[None, :] < reps[:, None]

    hist = np.zeros((n_draws, n_bins), dtype=np.int32)
    rows = np.repeat(np.arange(n_draws), max_size)[valid.ravel()]
    cols = bins.ravel()[valid.ravel()]
    np.add.at(hist, (rows, cols), 1)

    b1 = hist.max(axis=1).astype(np.float64) / reps
    b2 = ((hist[:, :-1] + hist[:, 1:]).max(axis=1).astype(np.float64) / reps
         if n_bins > 1 else b1.copy())

    out: dict[int, dict[int, tuple[float, float]]] = {}
    order = np.argsort(reps, kind="stable")
    reps_s, b1_s, b2_s = reps[order], b1[order], b2[order]
    starts = np.flatnonzero(np.concatenate(([True], reps_s[1:] != reps_s[:-1])))
    for i, s in enumerate(reps_s[starts]):
        lo = starts[i]
        hi = starts[i + 1] if i + 1 < starts.size else reps_s.size
        m1, m2 = b1_s[lo:hi], b2_s[lo:hi]
        out[int(s)] = {1: (float(m1.mean()), float(m1.std()) or 1e-9),
                       2: (float(m2.mean()), float(m2.std()) or 1e-9)}
    return out


def burst_z_table(entity: np.ndarray, first_bin: np.ndarray, bin_lo: int,
                  bin_hi: int, *, seed: int, window: int = PRIMARY_WINDOW,
                  draws_per_size: int = 400) -> dict:
    """`burst_z(e)` for every surviving entity, each against its own exact
    size's null. `draws_per_size x (typically tens of distinct sizes) lands
    near `N_NULL_DRAWS` in total simulated entities while giving every size
    its own unbiased comparison; size buckets are kept only as a reporting
    grouping (`bucket`), never as what an entity is actually compared to.
    """
    b = burst_for_entities(entity, first_bin, bin_lo, bin_hi)
    p = daily_share(first_bin, bin_lo, bin_hi)
    buckets = size_bucket(b["size"])

    null_by_size = simulate_null_by_size(b["size"], bin_lo, bin_hi, p,
                                         draws_per_size=draws_per_size, seed=seed)
    mean = np.array([null_by_size[int(s)][window][0] for s in b["size"]])
    sd = np.array([null_by_size[int(s)][window][1] for s in b["size"]])
    z = (b[window] - mean) / sd

    return {"entity": b["entity"], "size": b["size"], "bucket": buckets,
           "burst": b[window], "z": z, "window": window,
           "null_by_size": null_by_size}


# ------------------------------------------------------- multiplier fit --

def quartile_cutoffs(z: np.ndarray) -> np.ndarray:
    """Three cut points splitting z into four bins, empirical over this
    relation's own entities."""
    return np.quantile(z, [0.25, 0.5, 0.75])


def assign_bin(z: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return np.searchsorted(cutoffs, z, side="right")


def fit_lockstep_multipliers(cfg: Config | None = None, split=None,
                             graph_tag: str = "early") -> dict:
    """beta_{r, burst quartile}, fitted on training-pool accounts only.

    Mirrors `relation_weights.fit_relation_weights` exactly: measured from how
    much more often an edge of a given (relation, burst-quartile) bin joins
    two known fraudsters than chance predicts, on the early window, with
    training accounts the only ones visible. The quartile cutoffs themselves
    are fitted here too and are the thing later applied unchanged to the late
    window, the same way alpha is a training-time constant applied at scoring
    time.
    """
    from eval.split import make_split
    from orbweaver.data.windows import week2_windows

    cfg = cfg or load_config()
    split = split or make_split(cfg)
    proc = cfg.abs_path(cfg.paths.processed)

    labels = split.labels
    visible = np.zeros(labels.size, dtype=bool)
    visible[split.train] = True
    visible[split.val] = True
    if visible[split.test].any():
        raise RuntimeError("held-out accounts are visible to the lockstep fit")

    lo, hi = week2_windows(cfg)[graph_tag]
    orders = pq.read_table(proc / "orders_week2.parquet")
    day = orders["day_ordinal"].to_numpy()
    inwin = (day >= lo) & (day <= hi)
    uid = orders["user_id"].to_numpy()[inwin]
    day_w = day[inwin]
    p = daily_share(day_w, lo, hi)

    out: dict[str, dict] = {}
    for rel in cfg.data.buildable_relations:
        ent = orders[rel].to_numpy(zero_copy_only=False).astype(np.float64)[inwin]
        e, u, b, _ = first_arrival_groups(uid, ent, day_w, n_max=cfg.graph.n_max)
        if e.size == 0:
            out[rel] = {"entities": 0, "cutoffs": [], "bins": [],
                       "note": "no surviving entities in this window"}
            continue
        zt = burst_z_table(e, b, lo, hi, seed=cfg.seed, window=PRIMARY_WINDOW)
        cutoffs = quartile_cutoffs(zt["z"])
        bin_of_entity = assign_bin(zt["z"], cutoffs)

        # This entity's induced pairs, tagged with its (relation) bin. Groups
        # are rebuilt directly from (e, u) sorted by entity, matching
        # `first_arrival_groups`'s own contiguous-by-entity ordering.
        gorder = np.argsort(e, kind="stable")
        eg, ug = e[gorder], u[gorder]
        gstarts = np.flatnonzero(np.concatenate(([True], eg[1:] != eg[:-1])))
        gsizes = np.diff(np.append(gstarts, eg.size))
        pos_in_zt = np.searchsorted(zt["entity"], eg[gstarts])
        entity_bin = bin_of_entity[pos_in_zt]

        left, right, gidx = pairs_from_groups(ug, gsizes, gstarts)
        pair_bin = entity_bin[gidx]

        ls, ld = labels[left], labels[right]
        both = visible[left] & visible[right]
        bins_out = []
        for k in range(N_QUARTILES):
            m = (pair_bin == k) & both
            n = int(m.sum())
            pop = int((entity_bin == k).sum())
            if n < MIN_LABELLED_PAIRS_PER_BIN:
                bins_out.append({"bin": k, "entities": pop, "edges_labelled": n,
                                 "lift": 1.0, "beta": 1.0,
                                 "note": "too few labelled pairs; neutral"})
                continue
            a, bb = ls[m], ld[m]
            ff = float(((a == FRAUD) & (bb == FRAUD)).sum()) / n
            p_ = (float((a == FRAUD).sum()) + float((bb == FRAUD).sum())) / (2 * n)
            lift = ff / (p_ * p_) if p_ > 0 else 1.0
            bins_out.append({"bin": k, "entities": pop, "edges_labelled": n,
                             "fraud_fraud_rate": round(ff, 6),
                             "expected_if_random": round(p_ * p_, 6),
                             "lift": round(lift, 4)})

        measured = [r for r in bins_out if "note" not in r]
        mean_lift = float(np.mean([r["lift"] for r in measured])) if measured else 1.0
        for r in bins_out:
            if "note" not in r:
                r["beta"] = round(r["lift"] / mean_lift, 4)

        out[rel] = {"entities": int(np.unique(e).size),
                    "window_days": [lo, hi], "cutoffs": [round(float(c), 4) for c in cutoffs],
                    "bins": bins_out}

    manifest = {"fitted_on": f"week2_{graph_tag}", "primary_window": PRIMARY_WINDOW,
               "accounts_visible": int(visible.sum()), "heldout_excluded": int(split.test.size),
               "size_buckets": SIZE_BUCKETS, "n_null_draws": N_NULL_DRAWS,
               "relations": out,
               "note": ("beta multiplies the entity-rarity, relation-weighted edge "
                        "weight by how bursty the entity's arrivals were, relative to "
                        "a simulated null of the same size. Fitted on training and "
                        "validation accounts only. A bin near 1.0 means burstiness did "
                        "not add information for that relation.")}
    return manifest


def load_lockstep_multipliers(cfg: Config) -> dict | None:
    path = cfg.abs_path(cfg.paths.processed) / "lockstep_weights.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------- the second graph --

def build_lockstep_edges(cfg: Config, days: tuple[int, int], multipliers: dict,
                         alphas: dict) -> tuple:
    """The late-window graph, with an extra per-entity burst multiplier.

    Structurally parallel to `build_graph.build_graph`, not a call into it:
    this needs per-entity resolution to look up each entity's own burst bin,
    which the standard build already throws away by the time it aggregates
    across entities into one row per account pair. Nothing here is imported by
    anything the standard pipeline runs.
    """
    proc = cfg.abs_path(cfg.paths.processed)
    orders = pq.read_table(proc / "orders_week2.parquet")
    n_users = int(orders["user_id"].to_numpy().max()) + 1
    day = orders["day_ordinal"].to_numpy()
    inwin = (day >= days[0]) & (day <= days[1])
    uid = orders["user_id"].to_numpy()[inwin]
    day_w = day[inwin]

    all_src, all_dst, all_w = [], [], []
    per_relation = {}
    for rel in cfg.data.buildable_relations:
        ent = orders[rel].to_numpy(zero_copy_only=False).astype(np.float64)[inwin]
        e, u, b, _ = first_arrival_groups(uid, ent, day_w, n_max=cfg.graph.n_max)
        if e.size == 0:
            per_relation[rel] = {"pairs": 0}
            continue

        cutoffs = np.asarray(multipliers.get(rel, {}).get("cutoffs") or [], dtype=np.float64)
        bins_cfg = multipliers.get(rel, {}).get("bins") or []
        beta_by_bin = {b_["bin"]: float(b_["beta"]) for b_ in bins_cfg}

        zt = burst_z_table(e, b, days[0], days[1], seed=cfg.seed, window=PRIMARY_WINDOW)
        entity_beta = (np.array([beta_by_bin.get(int(k), 1.0) for k in assign_bin(zt["z"], cutoffs)])
                      if cutoffs.size else np.ones(zt["entity"].size))

        gorder = np.argsort(e, kind="stable")
        eg, ug = e[gorder], u[gorder]
        gstarts = np.flatnonzero(np.concatenate(([True], eg[1:] != eg[:-1])))
        gsizes = np.diff(np.append(gstarts, eg.size))
        pos = np.searchsorted(zt["entity"], eg[gstarts])
        group_beta = entity_beta[pos]
        group_w = rarity_weight(gsizes, cfg.graph.rarity_base).astype(np.float64) * group_beta
        alpha = float(alphas.get(rel, 1.0))
        group_w = group_w * alpha

        left, right, gidx = pairs_from_groups(ug, gsizes, gstarts)
        w = group_w[gidx].astype(np.float32)
        per_relation[rel] = {"pairs": int(left.size), "alpha": alpha,
                             "mean_beta": float(group_beta.mean()) if gsizes.size else 1.0}
        all_src.append(np.minimum(left, right).astype(np.int64))
        all_dst.append(np.maximum(left, right).astype(np.int64))
        all_w.append(w)

    src = np.concatenate(all_src) if all_src else np.empty(0, np.int64)
    dst = np.concatenate(all_dst) if all_dst else np.empty(0, np.int64)
    w = np.concatenate(all_w) if all_w else np.empty(0, np.float32)

    key = src * n_users + dst
    order = np.argsort(key, kind="stable")
    key, src, dst, w = key[order], src[order], dst[order], w[order]
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    weight = np.add.reduceat(w.astype(np.float64), first).astype(np.float32)

    return src[first].astype(np.int32), dst[first].astype(np.int32), weight, n_users, per_relation


def run(cfg: Config | None = None) -> dict:
    """Fit on the early window, build the late-window lockstep graph, compare
    it against the standard graph's ring metrics at the headline operating
    point, generalise the hostel test to all five relations, and run the
    IEEE-CIS arm."""
    import time as _time

    from eval.split import make_split
    from orbweaver.data.build_graph import build_graph
    from orbweaver.data.relation_weights import load_relation_weights
    from orbweaver.data.windows import LATE, week2_windows
    from orbweaver.rings.hostel_test import run_hostel_test_all_relations
    from orbweaver.rings.peel import EdgeList, extract_rings_batch

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    lo, hi = week2_windows(cfg)[LATE]

    t0 = _time.time()
    print("fitting burst multipliers on the early window ...", flush=True)
    fit = fit_lockstep_multipliers(cfg, graph_tag="early")
    (proc / "lockstep_weights.json").write_text(json.dumps(fit, indent=2, default=str))
    print(f"  done in {_time.time() - t0:.0f}s")
    for rel, v in fit["relations"].items():
        for b in v.get("bins", []):
            tag = b.get("note", "")
            print(f"    {rel} bin{b['bin']:>2}  entities={b.get('entities', 0):>7,}  "
                  f"edges={b.get('edges_labelled', 0):>7,}  lift={b.get('lift')!s:>7}  "
                  f"beta={b.get('beta')!s:>7}  {tag}")

    alphas = load_relation_weights(cfg) or {}
    t1 = _time.time()
    print("building the late-window lockstep graph ...", flush=True)
    src, dst, weight, n_users, per_rel = build_lockstep_edges(
        cfg, (lo, hi), fit["relations"], alphas)
    build_time = _time.time() - t1
    print(f"  {src.size:,} edges in {build_time:.0f}s")

    n = int(pq.read_table(proc / "nodes.parquet").num_rows)
    lockstep_edges = EdgeList(src.astype(np.int64), dst.astype(np.int64),
                              weight.astype(np.float64), n)

    from eval.metrics import ltv_proxy
    from eval.run_rings import load_edges, prune
    from orbweaver.rings.cost import evaluate_rings

    split = make_split(cfg)
    labels = split.labels
    scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()
    f = pq.read_table(proc / "features_week2_early.parquet", columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    fu = f["user_id"].to_numpy(); keep = fu < n
    orders_n[fu[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    def measure(edges: EdgeList) -> tuple[dict, list]:
        sub = prune(edges, scores, cfg.rings.prune_tau_headline)
        rings = extract_rings_batch(sub, scores, lambda_=cfg.rings.lambda_headline,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
        block = evaluate_rings(rings, labels, ltv, restrict_to=split.test) if rings else {}
        return {"n_rings": len(rings), "accounts_in_rings": block.get("accounts_in_rings"),
               "ring_precision": block.get("ring_precision"),
               "precision_lift_over_base": block.get("precision_lift_over_base"),
               "fraud_members": block.get("fraud_members"),
               "normal_flagged_per_fraud_caught": block.get("normal_flagged_per_fraud_caught")}, rings

    standard_edges = load_edges(LATE, cfg, n)
    std_metrics, std_rings = measure(standard_edges)
    ls_metrics, ls_rings = measure(lockstep_edges)
    print(f"\nstandard graph:  precision {std_metrics['ring_precision']}  "
          f"per catch {std_metrics['normal_flagged_per_fraud_caught']}")
    print(f"lockstep graph:  precision {ls_metrics['ring_precision']}  "
          f"per catch {ls_metrics['normal_flagged_per_fraud_caught']}")

    print("\ngeneralising the crowd test to all five relations, both graphs ...", flush=True)
    crowd = run_hostel_test_all_relations(cfg, {"standard": std_rings, "lockstep": ls_rings})

    ieee_arm = None
    ieee_flag = proc / "ieee_cis.json"
    if (cfg.abs_path(".") / "data/raw/ieee_cis/train_transaction.csv").exists():
        print("\nrunning the IEEE-CIS arm (second-resolution windows) ...", flush=True)
        from orbweaver.data.ieee_lockstep import run_ieee_lockstep
        ieee_arm = run_ieee_lockstep(cfg)
    elif ieee_flag.exists():
        print("\nIEEE-CIS raw files absent (only the prior run's summary is), "
              "skipping the strong arm this run.")

    out = {
        "method": ("A second, separate account graph. Edge weight is the same "
                  "alpha_r x rarity(e) as the standard graph, times an extra "
                  "beta_{relation, burst quartile}, fitted on training-pool "
                  "accounts on the early window and applied unchanged to the "
                  "late window. `graph.time_weighting` in config/default.yaml "
                  "stays false; nothing in the standard pipeline reads this "
                  "module or that flag."),
        "primary_window_days": PRIMARY_WINDOW,
        "burst_windows_reported": list(BURST_WINDOWS),
        "fit": fit,
        "late_graph": {"edges": int(src.size), "per_relation": per_rel,
                       "build_seconds": round(build_time, 1)},
        "rings_at_headline": {"standard": std_metrics, "lockstep": ls_metrics},
        "crowd_test_all_relations": crowd,
        "ieee_cis_arm": ieee_arm,
    }
    return out


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "lockstep.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
