"""Lockstep burstiness on a payment processor's graph, at second resolution.

PPA's lockstep arm is weak by construction: a burst is a day, the finest
resolution the released order files carry, so CopyCatch's whole argument -
that coordination clusters in time while natural activity spreads across it -
barely has room to show itself. IEEE-CIS's `TransactionDT` is seconds over six
months, so hour, six-hour and day windows are all meaningful, and this is
where the idea gets a fair test.

The question that motivates it: the billing-address relation is at once the
strongest signal on this graph (5.42x fraud lift) and the thing that
legitimately ties together every card billed to one building, and
`docs/results.md` already says weighting cannot separate the two because
weighting is what found the address informative in the first place. Time is a
genuinely different axis - a family or an office accrues cards over months; a
ring provisions and uses them inside a much narrower window - so it is worth
testing whether it fixes what rarity and relation-lift weighting could not: 4
of 7 apartment clusters touched.

Same design as `orbweaver.data.lockstep`, reapplied: burst(e) against a
simulated size-corrected null, a multiplier fitted on training accounts only
at each resolution, a second graph, the address-cluster test run both ways.
Nothing here is imported by `eval.ieee_cis_run` or by anything the standard
pipeline runs; the committed IEEE-CIS numbers cannot move because this module
never touches the code that produces them.
"""
from __future__ import annotations

import numpy as np

from orbweaver.config import Config, load_config
from orbweaver.data.build_graph import pairs_from_groups, rarity_weight
from orbweaver.data.lockstep import (MIN_LABELLED_PAIRS_PER_BIN, N_QUARTILES,
                                     assign_bin, burst_z_table, daily_share,
                                     first_arrival_groups, quartile_cutoffs)
from orbweaver.rings.peel import EdgeList, extract_rings_batch

FRAUD = 1
RESOLUTIONS = {"1_hour": 3600, "6_hour": 21600, "1_day": 86400}


def fit_ieee_multipliers(cfg: Config, st: dict, resolution_seconds: int) -> dict:
    """beta_{relation, burst quartile} on the early (train) half, one
    resolution. Identical logic to `lockstep.fit_lockstep_multipliers`; the
    only difference is the bin unit and that entities come from IEEE-CIS's
    five payment-side relations instead of PPA's five order relations."""
    df, account, n = st["df"], st["account"], st["n"]
    early, cols, labels = st["early"], st["cols"], st["labels"]
    train = st["train"]

    dt = df["TransactionDT"].to_numpy()[early]
    bin_idx = (dt // resolution_seconds).astype(np.int64)
    bin_lo, bin_hi = int(bin_idx.min()), int(bin_idx.max())
    p = daily_share(bin_idx, bin_lo, bin_hi)

    visible = np.zeros(n, dtype=bool)
    visible[train] = True
    acc_early = account[early]

    out: dict[str, dict] = {}
    for rel, ent_full in cols.items():
        ent = ent_full[early]
        e, u, b, _ = first_arrival_groups(acc_early, ent, bin_idx, n_max=cfg.graph.n_max)
        if e.size == 0:
            out[rel] = {"entities": 0, "cutoffs": [], "bins": []}
            continue
        zt = burst_z_table(e, b, bin_lo, bin_hi, seed=cfg.seed, window=1)
        cutoffs = quartile_cutoffs(zt["z"])
        bin_of_entity = assign_bin(zt["z"], cutoffs)

        gorder = np.argsort(e, kind="stable")
        eg, ug = e[gorder], u[gorder]
        gstarts = np.flatnonzero(np.concatenate(([True], eg[1:] != eg[:-1])))
        gsizes = np.diff(np.append(gstarts, eg.size))
        pos = np.searchsorted(zt["entity"], eg[gstarts])
        entity_bin = bin_of_entity[pos]

        left, right, gidx = pairs_from_groups(ug, gsizes, gstarts)
        pair_bin = entity_bin[gidx]
        ls, ld = labels[left], labels[right]
        both = visible[left] & visible[right]

        bins_out = []
        for k in range(N_QUARTILES):
            m = (pair_bin == k) & both
            cnt = int(m.sum())
            pop = int((entity_bin == k).sum())
            if cnt < MIN_LABELLED_PAIRS_PER_BIN:
                bins_out.append({"bin": k, "entities": pop, "edges_labelled": cnt,
                                 "lift": 1.0, "beta": 1.0, "note": "too few labelled pairs; neutral"})
                continue
            a, bb = ls[m], ld[m]
            ff = float(((a == FRAUD) & (bb == FRAUD)).sum()) / cnt
            p_ = (float((a == FRAUD).sum()) + float((bb == FRAUD).sum())) / (2 * cnt)
            lift = ff / (p_ * p_) if p_ > 0 else 1.0
            bins_out.append({"bin": k, "entities": pop, "edges_labelled": cnt,
                             "fraud_fraud_rate": round(ff, 6), "lift": round(lift, 4)})
        measured = [r for r in bins_out if "note" not in r]
        mean_lift = float(np.mean([r["lift"] for r in measured])) if measured else 1.0
        for r in bins_out:
            if "note" not in r:
                r["beta"] = round(r["lift"] / mean_lift, 4)
        out[rel] = {"entities": int(np.unique(e).size),
                   "cutoffs": [round(float(c), 4) for c in cutoffs], "bins": bins_out}
    return {"resolution_seconds": resolution_seconds, "bin_range": [bin_lo, bin_hi],
           "relations": out}


def build_ieee_lockstep_edges(cfg: Config, st: dict, resolution_seconds: int,
                              multipliers: dict, mask: np.ndarray) -> EdgeList:
    """The late (score) half's graph, entity weight carrying the extra beta."""
    df, account, n = st["df"], st["account"], st["n"]
    cols, alphas = st["cols"], st["alphas"]
    dt = df["TransactionDT"].to_numpy()[mask]
    bin_idx = (dt // resolution_seconds).astype(np.int64)
    bin_lo, bin_hi = int(bin_idx.min()), int(bin_idx.max()) if bin_idx.size else (0, 0)
    acc = account[mask]

    all_src, all_dst, all_w = [], [], []
    for bit, (rel, ent_full) in enumerate(cols.items()):
        ent = ent_full[mask]
        e, u, b, _ = first_arrival_groups(acc, ent, bin_idx, n_max=cfg.graph.n_max)
        if e.size == 0:
            continue
        rel_fit = multipliers.get("relations", {}).get(rel, {})
        cutoffs = np.asarray(rel_fit.get("cutoffs") or [], dtype=np.float64)
        beta_by_bin = {r["bin"]: float(r["beta"]) for r in rel_fit.get("bins", [])}

        zt = burst_z_table(e, b, bin_lo, bin_hi, seed=cfg.seed, window=1)
        entity_beta = (np.array([beta_by_bin.get(int(k), 1.0) for k in assign_bin(zt["z"], cutoffs)])
                      if cutoffs.size else np.ones(zt["entity"].size))

        gorder = np.argsort(e, kind="stable")
        eg, ug = e[gorder], u[gorder]
        gstarts = np.flatnonzero(np.concatenate(([True], eg[1:] != eg[:-1])))
        gsizes = np.diff(np.append(gstarts, eg.size))
        pos = np.searchsorted(zt["entity"], eg[gstarts])
        group_beta = entity_beta[pos]
        group_w = (rarity_weight(gsizes, cfg.graph.rarity_base).astype(np.float64)
                  * group_beta * float(alphas.get(rel, 1.0)))

        left, right, gidx = pairs_from_groups(ug, gsizes, gstarts)
        w = group_w[gidx]
        all_src.append(np.minimum(left, right).astype(np.int64))
        all_dst.append(np.maximum(left, right).astype(np.int64))
        all_w.append(w)

    if not all_src:
        return EdgeList(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64), n)
    src = np.concatenate(all_src); dst = np.concatenate(all_dst); w = np.concatenate(all_w)
    key = src * n + dst
    order = np.argsort(key, kind="stable")
    key, src, dst, w = key[order], src[order], dst[order], w[order]
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    return EdgeList(src[first], dst[first], np.add.reduceat(w, first), n)


def run_ieee_lockstep(cfg: Config | None = None) -> dict:
    """Fit and build at each of the three resolutions; run the address-cluster
    test both ways at each; compare against the standard (no time weighting)
    result already reported in `ieee_cis.json`."""
    from eval.ieee_cis_run import address_cluster_test, prepare_ieee
    from orbweaver.rings.cost import evaluate_rings

    cfg = cfg or load_config()
    st = prepare_ieee(cfg)
    df, account, n, labels = st["df"], st["account"], st["n"], st["labels"]
    late, test, scores, e_late = st["late"], st["test"], st["scores"], st["e_late"]

    keep = scores > cfg.rings.prune_tau_headline
    m_std = keep[e_late.src] & keep[e_late.dst]
    std_edges = EdgeList(e_late.src[m_std], e_late.dst[m_std], e_late.weight[m_std], n)
    std_rings = extract_rings_batch(std_edges, scores, lambda_=cfg.rings.lambda_headline,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
    ltv = np.zeros(n)
    std_block = evaluate_rings(std_rings, labels, ltv, restrict_to=test) if std_rings else {}
    std_addr = address_cluster_test(cfg, df[late], account[late], labels, std_rings, n)

    arms = {}
    for name, res_s in RESOLUTIONS.items():
        print(f"  fitting at {name} resolution ...", flush=True)
        fit = fit_ieee_multipliers(cfg, st, res_s)
        edges = build_ieee_lockstep_edges(cfg, st, res_s, fit, late)
        keep_ls = scores > cfg.rings.prune_tau_headline
        m = keep_ls[edges.src] & keep_ls[edges.dst]
        sub = EdgeList(edges.src[m], edges.dst[m], edges.weight[m], n)
        rings = extract_rings_batch(sub, scores, lambda_=cfg.rings.lambda_headline,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
        block = evaluate_rings(rings, labels, ltv, restrict_to=test) if rings else {}
        addr = address_cluster_test(cfg, df[late], account[late], labels, rings, n)
        arms[name] = {
            "resolution_seconds": res_s, "fit": fit, "edges": int(edges.src.size),
            "ring_precision": block.get("ring_precision"),
            "normal_flagged_per_fraud_caught": block.get("normal_flagged_per_fraud_caught"),
            "fraud_members": block.get("fraud_members"),
            "address_cluster_test": addr,
        }
        print(f"    precision {block.get('ring_precision')}  "
              f"address clusters {addr['clusters_touched']}/{addr['clusters_found']}")

    return {
        "standard": {"ring_precision": std_block.get("ring_precision"),
                    "normal_flagged_per_fraud_caught": std_block.get("normal_flagged_per_fraud_caught"),
                    "fraud_members": std_block.get("fraud_members"),
                    "address_cluster_test": std_addr},
        "resolutions": arms,
        "method": ("Same design as the PPA arm - burst(e) against a simulated "
                  "size-corrected null, multipliers fitted on training accounts, "
                  "a second graph - reapplied at three second-resolution windows "
                  "IEEE-CIS's timestamps actually support."),
    }
