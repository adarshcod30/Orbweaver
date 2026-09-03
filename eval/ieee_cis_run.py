"""Run the standard pipeline on a payment processor's transactions.

Same graph construction, same relation weighting, same scorer, same score
cut-off, same peeling. The only things that change are where the entities come
from and what an account is.

The split is stronger here than on PPA. IEEE-CIS has real timestamps over six
months, so the early window genuinely precedes the late one in calendar time
*and* the held-out accounts are absent from training — both guarantees at once,
rather than the within-week arrangement PPA forces.
"""
from __future__ import annotations

import json

import numpy as np

from orbweaver.config import Config, load_config
from orbweaver.data.build_graph import relation_edges
from orbweaver.data.ieee_cis import (
    MAX_TX_PER_ACCOUNT, RELATIONS, account_labels, account_proxy, load_raw,
    relation_columns, window_features,
)
from orbweaver.rings.peel import EdgeList, extract_rings_batch


def build_edges(cfg: Config, cols: dict, account: np.ndarray, n: int,
                mask: np.ndarray, alphas: dict | None = None):
    """Account-pair edges from shared entities, exactly as on the order data."""
    alphas = alphas or {}
    src_all, dst_all, w_all, mask_all, per_rel = [], [], [], [], {}
    for bit, (rel, ent) in enumerate(cols.items()):
        s, d, w, esz = relation_edges(account[mask], ent[mask],
                                      n_max=cfg.graph.n_max,
                                      rarity_base=cfg.graph.rarity_base)
        alpha = float(alphas.get(rel, 1.0))
        if alpha != 1.0:
            w = (w * alpha).astype(np.float32)
        per_rel[rel] = {"pairs": int(s.size), "alpha": alpha}
        src_all.append(s.astype(np.int64)); dst_all.append(d.astype(np.int64))
        w_all.append(w.astype(np.float64))
        mask_all.append(np.full(s.size, 1 << bit, dtype=np.int32))

    src = np.concatenate(src_all); dst = np.concatenate(dst_all)
    w = np.concatenate(w_all); rmask = np.concatenate(mask_all)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = lo * n + hi
    order = np.argsort(key, kind="stable")
    key, lo, hi, w, rmask = key[order], lo[order], hi[order], w[order], rmask[order]
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    return (EdgeList(lo[first], hi[first], np.add.reduceat(w, first), n),
            np.bitwise_or.reduceat(rmask, first), per_rel)


def fit_alphas(edges: EdgeList, rmask: np.ndarray, labels: np.ndarray,
               visible: np.ndarray, cols: dict) -> dict:
    """Fraud-fraud lift per relation, on training accounts only."""
    both = visible[edges.src] & visible[edges.dst]
    ls, ld = labels[edges.src], labels[edges.dst]
    out, lifts = {}, {}
    for bit, rel in enumerate(cols):
        m = (((rmask >> bit) & 1) == 1) & both
        n = int(m.sum())
        if n < 200:
            out[rel] = {"edges_labelled": n, "lift": 1.0, "measured": False}
            continue
        a, b = ls[m], ld[m]
        ff = float(((a == 1) & (b == 1)).sum()) / n
        p = (float((a == 1).sum()) + float((b == 1).sum())) / (2 * n)
        lift = ff / (p * p) if p > 0 else 1.0
        out[rel] = {"edges_labelled": n, "lift": round(lift, 4), "measured": True}
        lifts[rel] = lift
    mean = float(np.mean(list(lifts.values()))) if lifts else 1.0
    for rel in out:
        out[rel]["alpha"] = round(out[rel]["lift"] / mean, 4) if out[rel]["measured"] else 1.0
        out[rel]["meaning"] = RELATIONS.get(rel, rel)
    return out


def graph_features(edges: EdgeList, rmask: np.ndarray, n_rel: int, n: int) -> np.ndarray:
    both = np.concatenate([edges.src, edges.dst])
    w2 = np.concatenate([edges.weight, edges.weight])
    m2 = np.concatenate([rmask, rmask])
    deg = np.bincount(both, minlength=n).astype(np.float64)
    wdeg = np.bincount(both, weights=w2, minlength=n)
    cols = [deg, wdeg, np.where(deg > 0, wdeg / np.maximum(deg, 1), 0.0)]
    for bit in range(n_rel):
        cols.append(np.bincount(both, weights=((m2 >> bit) & 1).astype(np.float64),
                                minlength=n))
    other = np.concatenate([edges.dst, edges.src])
    cols.append(np.where(deg > 0,
                         np.bincount(both, weights=deg[other], minlength=n)
                         / np.maximum(deg, 1), 0.0))
    return np.column_stack(cols).astype(np.float32)


def address_cluster_test(cfg: Config, df, account, labels, rings, n) -> dict:
    """The apartment-building analogue of the hostel test.

    Many cards billed to one address is ordinary - a building, a family, a
    small business - so this is the population the method must not sweep up.
    """
    addr = df["addr1"].astype("string")
    codes, _ = __import__("pandas").factorize(addr, sort=True)
    v = codes.astype(np.float64); v[codes < 0] = np.nan
    ok = ~np.isnan(v)
    pair = np.stack([v[ok], account[ok]])
    o = np.lexsort(pair[::-1])
    e, a = pair[0][o], pair[1][o]
    new = np.empty(e.size, dtype=bool)
    new[0] = True
    np.logical_or(e[1:] != e[:-1], a[1:] != a[:-1], out=new[1:])
    e, a = e[new], a[new].astype(np.int64)

    starts = np.flatnonzero(np.concatenate(([True], e[1:] != e[:-1])))
    sizes = np.diff(np.append(starts, e.size))
    in_ring = np.zeros(n, dtype=bool)
    for r in rings:
        in_ring[r.members] = True

    found = touched = 0
    for s, sz in zip(starts, sizes):
        if not (15 <= sz <= cfg.graph.n_max):
            continue
        members = a[s:s + sz]
        lab = labels[members]
        known = int((lab == 1).sum() + (lab == 0).sum())
        if known < 3 or (lab == 0).sum() / max(known, 1) < 0.8:
            continue
        found += 1
        touched += int(in_ring[members].any())
    return {"clusters_found": found, "clusters_touched": touched,
            "share_touched": round(touched / found, 4) if found else None,
            "criteria": "15+ cards billed to one address, 80%+ of labelled ones good"}


def prepare_ieee(cfg: Config) -> dict:
    """Load, dedupe, label, split, score and build the late graph once.

    Factored out of `run()` so the lockstep arm can share exactly this state
    - same accounts, same labels, same split, same scores - rather than
    re-deriving it and risking a second version that quietly disagrees with
    the committed numbers. `run()` below calls this and its own output is
    unchanged by the refactor; a test checks that.
    """
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    print("loading transactions ...", flush=True)
    df = load_raw(cfg)
    account, fingerprints = account_proxy(df)
    n = int(account.max()) + 1

    counts = np.bincount(account, minlength=n)
    too_common = counts > MAX_TX_PER_ACCOUNT
    keep_tx = ~too_common[account]
    print(f"accounts: {n:,}  transactions: {len(df):,}  "
          f"dropped {int((~keep_tx).sum()):,} rows on "
          f"{int(too_common.sum())} over-common fingerprints")

    labels, sens = account_labels(df[keep_tx], account[keep_tx], n)
    days = df["day"].to_numpy()
    mid = int(np.median(days))
    early = keep_tx & (days <= mid)
    late = keep_tx & (days > mid)
    print(f"forward split: days {int(days.min())}-{mid} train, "
          f"{mid + 1}-{int(days.max())} score")

    cols = relation_columns(df)
    rng = np.random.default_rng(cfg.seed)

    labelled = np.flatnonzero(labels != -1)
    tr_parts, te_parts = [], []
    for cls in (0, 1):
        members = labelled[labels[labelled] == cls]
        perm = rng.permutation(members.size)
        cut = int(round(cfg.scoring.heldout_fraction * members.size))
        te_parts.append(members[perm[:cut]]); tr_parts.append(members[perm[cut:]])
    test = np.sort(np.concatenate(te_parts))
    train = np.sort(np.concatenate(tr_parts))
    visible = np.zeros(n, dtype=bool); visible[train] = True

    print("building the early graph and fitting relation weights ...", flush=True)
    e0, m0, _ = build_edges(cfg, cols, account, n, early)
    alphas_info = fit_alphas(e0, m0, labels, visible, cols)
    alphas = {r: v["alpha"] for r, v in alphas_info.items()}
    for r, v in alphas_info.items():
        print(f"    {r:18s} {v['edges_labelled']:>9,} labelled edges  "
              f"lift {v['lift']:>7}  alpha {v['alpha']}")

    e_early, m_early, per_rel_early = build_edges(cfg, cols, account, n, early, alphas)
    e_late, m_late, per_rel_late = build_edges(cfg, cols, account, n, late, alphas)

    Xe = np.hstack([window_features(df[early], account[early], n,
                                    int(days[early].min()), mid)[0],
                    graph_features(e_early, m_early, len(cols), n)])
    Xl = np.hstack([window_features(df[late], account[late], n,
                                    mid + 1, int(days[late].max()))[0],
                    graph_features(e_late, m_late, len(cols), n)])

    y = (labels == 1).astype(np.int8)
    params = cfg.scoring.xgb.model_dump()
    n_est = params.pop("n_estimators")
    pos, neg = int(y[train].sum()), int((y[train] == 0).sum())
    print(f"training on {train.size:,} accounts ({pos:,} fraud) ...", flush=True)
    model = xgb.XGBClassifier(**params, n_estimators=n_est,
                              scale_pos_weight=neg / max(pos, 1),
                              objective="binary:logistic", eval_metric="aucpr",
                              random_state=cfg.seed)
    model.fit(Xe[train], y[train], verbose=False)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.predict_proba(Xe[train])[:, 1], y[train])
    scores = iso.predict(model.predict_proba(Xl)[:, 1]).astype(np.float64)

    return {"df": df, "account": account, "n": n, "labels": labels, "sens": sens,
           "days": days, "mid": mid, "early": early, "late": late, "cols": cols,
           "train": train, "test": test, "alphas": alphas, "alphas_info": alphas_info,
           "e_early": e_early, "e_late": e_late, "m_late": m_late,
           "per_rel_late": per_rel_late, "scores": scores}


def run(cfg: Config | None = None) -> dict:
    from eval.metrics import evaluate
    from orbweaver.rings.cost import evaluate_rings

    cfg = cfg or load_config()
    st = prepare_ieee(cfg)
    df, account, n, labels = st["df"], st["account"], st["n"], st["labels"]
    late, test, scores = st["late"], st["test"], st["scores"]
    days, mid, train = st["days"], st["mid"], st["train"]
    e_late = st["e_late"]

    y = (labels == 1).astype(np.int8)
    node = evaluate(y[test], scores[test])
    base = float(y[test].mean())

    keep = scores > cfg.rings.prune_tau_headline
    m = keep[e_late.src] & keep[e_late.dst]
    sub = EdgeList(e_late.src[m], e_late.dst[m], e_late.weight[m], n)
    rings = extract_rings_batch(sub, scores, lambda_=cfg.rings.lambda_headline,
                                k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
    ltv = np.zeros(n)
    ring_block = evaluate_rings(rings, labels, ltv, restrict_to=test) if rings else {}
    addr = address_cluster_test(cfg, df[late], account[late], labels, rings, n)

    return {
        "dataset": "IEEE-CIS (Vesta), train_transaction + train_identity",
        "caveats": [
            "This is card fraud, not promotion abuse. A stolen card used across "
            "merchants is a different mechanism from many accounts farming one "
            "offer; only the graph shape is shared.",
            "The account is a card fingerprint "
            "(card1|card2|card3|card5|addr1|addr2), not a person. One person "
            "with two cards is two accounts here.",
            "The identity file covers 144,233 of 590,540 transactions (24.4%), "
            "so device and browser edges exist for a quarter of the data and "
            "are missing, not zero, for the rest.",
        ],
        "transactions": int(len(df)),
        "accounts": int(n),
        "split": {
            "kind": "forward in time AND account-disjoint",
            "train_days": [int(days.min()), mid],
            "score_days": [mid + 1, int(days.max())],
            "train_accounts": int(train.size),
            "heldout_accounts": int(test.size),
            "heldout_base_rate": round(base, 4),
        },
        "label_rule": {"headline": "at least half an account's transactions are fraud",
                       "sensitivity": st["sens"]},
        "relation_weights": st["alphas_info"],
        "edges": {"early": int(st["e_early"].src.size), "late": int(e_late.src.size),
                  "per_relation_late": st["per_rel_late"]},
        "node_scoring_heldout": node,
        "rings": {
            "tau": cfg.rings.prune_tau_headline,
            "lambda": cfg.rings.lambda_headline,
            "n_rings": len(rings),
            "accounts_in_rings": ring_block.get("accounts_in_rings"),
            "ring_precision": ring_block.get("ring_precision"),
            "precision_lift_over_base": round(
                (ring_block.get("ring_precision") or 0) / base, 3) if base else None,
            "fraud_members": ring_block.get("fraud_members"),
            "normal_flagged_per_fraud_caught":
                ring_block.get("normal_flagged_per_fraud_caught"),
            "heldout_only": ring_block.get("heldout_only"),
        },
        "address_cluster_test": addr,
    }


def main() -> None:
    cfg = load_config()
    base = cfg.abs_path(".") / "data/raw/ieee_cis/train_transaction.csv"
    if not base.exists():
        print("IEEE-CIS not present; run `make download-ieee-cis`. Skipping.")
        return
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "ieee_cis.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    r, s = out["rings"], out["split"]
    print(f"\nheld-out base rate {s['heldout_base_rate']}  "
          f"node AUPRC {out['node_scoring_heldout']['auprc']} "
          f"({out['node_scoring_heldout']['auprc_lift_over_random']}x random)")
    print(f"rings: {r['n_rings']} over {r['accounts_in_rings']} accounts, "
          f"precision {r['ring_precision']} ({r['precision_lift_over_base']}x base), "
          f"{r['normal_flagged_per_fraud_caught']} good cards per catch")
    a = out["address_cluster_test"]
    print(f"address clusters: {a['clusters_touched']} of {a['clusters_found']} touched")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
