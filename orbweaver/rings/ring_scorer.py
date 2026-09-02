"""A confidence for each ring, learned from rings rather than from accounts.

The queue is ordered by density today, which is the crudest thing it could be.
Density says how tightly a group is connected; it does not say how likely the
group is to be fraudulent, and those are not the same question — the whole
reason the score cut-off exists is that the densest groups in this graph are
mostly ordinary people.

So this learns a ring-level model. It is a small one, and the care is almost
entirely in where the training data comes from.

**Out-of-fold account scores.** Candidate rings are built on the *early*
window, and if I peel that window using the account scores the model produced
after training on it, every ring is built from scores that have already seen
their own members' labels. The rings would look better than they are and the
ring model would learn from a fantasy. So the account scores used to build
candidates are five-fold out-of-fold: each fold's accounts are scored by a
model trained on the other four and calibrated on them too. Held-out accounts
are never touched by any of it.

**Label-free ring features.** `ring_features` takes no labels, by signature,
so it cannot accidentally read one. Everything it computes — size, density,
what the members share, how their scores are distributed, how concentrated
their ordering is — is available at the moment a ring is extracted, before
anyone knows whether it is fraud.

**Three rankings, reported together.** Learned confidence is compared against
density (what the queue does today) and against mean member score, which is
the obvious baseline anyone would try first. Reporting only the first two
would make the model look better than the honest comparison does.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList, Ring, extract_rings_batch

# Candidate rings are generated across a spread of operating points, so the
# ring model sees loose groups as well as tight ones and has to tell them
# apart, rather than only ever seeing the output of one good setting.
CANDIDATE_CELLS = [(tau, lam) for tau in (0.3, 0.5) for lam in (0.0, 1.0, 5.0)]
# 300 per cell across six cells was 1,800 rings per window and ran for over an
# hour without reaching a checkpoint. 120 gives 720 candidates per window,
# which is far more than a 21-feature model needs, and finishes in minutes.
CANDIDATE_TOP_K = 120
# Two candidates overlapping more than this are the same group found twice.
DEDUPE_JACCARD = 0.8
# A candidate needs this many labelled members before its fraud share means
# anything; below it the label is noise.
MIN_LABELLED = 3
POSITIVE_FRAUD_SHARE = 0.5

FEATURE_NAMES = [
    "size", "density", "internal_weight_per_member", "score_mass_per_member",
    "relation_kinds", "top_entity_coverage", "top_entity_platform_count",
    "top_entity_rarity", "rare_edge_share",
    "score_mean", "score_median", "score_min", "score_p10", "score_std",
    "orders_per_member", "promo_share", "busiest_day_share", "day_span",
    "active_days_per_member", "tau", "lambda",
]


def account_aggregates(cfg: Config, week: int, days: tuple[int, int] | None,
                       n: int) -> dict[str, np.ndarray]:
    """Per-account order shape, for the ring-level features."""
    proc = cfg.abs_path(cfg.paths.processed)
    t = pq.read_table(proc / f"orders_week{week}.parquet",
                      columns=["user_id", "day_ordinal", "r6"])
    uid = t["user_id"].to_numpy()
    day = t["day_ordinal"].to_numpy()
    promo = ~np.isnan(t["r6"].to_numpy(zero_copy_only=False))
    if days is not None:
        m = (day >= days[0]) & (day <= days[1])
        uid, day, promo = uid[m], day[m], promo[m]
    orders = np.bincount(uid, minlength=n).astype(np.float64)
    promo_orders = np.bincount(uid[promo], minlength=n).astype(np.float64)
    return {"orders": orders, "promo_orders": promo_orders,
            "uid": uid, "day": day}


def ring_features(ring: Ring, scores: np.ndarray, agg: dict,
                  edge_index: dict, tau: float, lam: float) -> np.ndarray:
    """Everything a reviewer could know about a ring before opening it.

    Takes no labels. That is enforced by the signature rather than by
    discipline: there is no argument here that could carry one.
    """
    m = ring.members
    n_m = max(m.size, 1)
    s = scores[m]

    inside = edge_index["inside"]
    inside[:] = False
    inside[m] = True
    src, dst = edge_index["src"], edge_index["dst"]
    live = inside[src] & inside[dst]
    kinds = 0
    if live.any():
        kinds = bin(int(np.bitwise_or.reduce(edge_index["mask"][live]))).count("1")
    esz = edge_index["entity"][live] if live.any() else np.array([np.inf])
    rare_share = float((esz <= 10).mean()) if live.any() else 0.0
    top_count = float(np.min(esz)) if live.any() and np.isfinite(esz).any() else 0.0
    coverage = float(live.sum()) / max(n_m * (n_m - 1) / 2, 1)

    uid, day = agg["uid"], agg["day"]
    sel = inside[uid]
    if sel.any():
        d = day[sel]
        counts = np.bincount(d - d.min())
        busiest = float(counts.max() / counts.sum())
        span = float(d.max() - d.min() + 1)
        active = float((counts > 0).sum()) / n_m
    else:
        busiest, span, active = 0.0, 0.0, 0.0

    orders = float(agg["orders"][m].sum())
    promo = float(agg["promo_orders"][m].sum())

    return np.array([
        float(n_m),
        float(ring.density),
        float(ring.internal_weight) / n_m,
        float(ring.score_mass) / n_m,
        float(kinds),
        coverage,
        top_count,
        1.0 / np.log(2.0 + top_count) if top_count > 0 else 0.0,
        rare_share,
        float(s.mean()), float(np.median(s)), float(s.min()),
        float(np.percentile(s, 10)), float(s.std()),
        orders / n_m,
        promo / max(orders, 1.0),
        busiest, span, active,
        float(tau), float(lam),
    ], dtype=np.float64)


def oof_account_scores(cfg: Config, split, X: np.ndarray,
                       n_folds: int = 5) -> np.ndarray:
    """Account scores where nobody was scored by a model that saw them.

    The training pool is scored fold-wise; everything else — unlabelled
    accounts, and the held-out set, which no ring model ever trains on — is
    scored by the full model. Without this the candidate rings would be built
    from scores that already knew their members' labels.
    """
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(cfg.seed)
    pool = split.train_pool
    y_pool = split.y(pool)
    order = rng.permutation(pool.size)
    folds = np.array_split(order, n_folds)

    params = cfg.scoring.xgb.model_dump()
    n_est = params.pop("n_estimators")
    out = np.zeros(X.shape[0], dtype=np.float32)
    assigned = np.zeros(X.shape[0], dtype=bool)

    for k in range(n_folds):
        held = pool[folds[k]]
        rest = pool[np.concatenate([folds[j] for j in range(n_folds) if j != k])]
        y_rest = split.y(rest)
        pos, neg = int(y_rest.sum()), int((y_rest == 0).sum())
        model = xgb.XGBClassifier(**params, n_estimators=n_est,
                                  scale_pos_weight=neg / max(pos, 1),
                                  objective="binary:logistic",
                                  eval_metric="aucpr", random_state=cfg.seed)
        model.fit(X[rest], y_rest, verbose=False)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(model.predict_proba(X[rest])[:, 1], y_rest)
        out[held] = iso.predict(model.predict_proba(X[held])[:, 1])
        assigned[held] = True

    # Everyone else gets the full model; none of them is in the ring model's
    # training data, so there is nothing to leak into.
    from orbweaver.scoring.xgb_graph import load_scorer, score_features
    model, calibrate = load_scorer(cfg)
    rest = ~assigned
    out[rest] = score_features(model, calibrate, X[rest])
    return out


def dedupe(rings: list[tuple[Ring, float, float]]) -> list[tuple[Ring, float, float]]:
    """Drop candidates that are the same group found at a different setting."""
    kept: list[tuple[Ring, float, float]] = []
    seen: list[set] = []
    for r, tau, lam in sorted(rings, key=lambda t: -t[0].density):
        ms = set(r.members.tolist())
        if any(len(ms & s) / len(ms | s) > DEDUPE_JACCARD for s in seen):
            continue
        kept.append((r, tau, lam))
        seen.append(ms)
    return kept


def candidates(cfg: Config, edges: EdgeList, scores: np.ndarray,
               top_k: int = CANDIDATE_TOP_K,
               label: str = "") -> list[tuple[Ring, float, float]]:
    import sys
    import time

    out = []
    for tau, lam in CANDIDATE_CELLS:
        t0 = time.time()
        keep = scores > tau
        m = keep[edges.src] & keep[edges.dst]
        if not m.any():
            continue
        sub = EdgeList(edges.src[m], edges.dst[m], edges.weight[m], edges.n_nodes)
        for r in extract_rings_batch(sub, scores.astype(np.float64), lambda_=lam,
                                     k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                     top_k=top_k, g_min=cfg.rings.g_min):
            out.append((r, tau, lam))
        print(f"    {label} tau={tau} lambda={lam}: {len(out):>5,} candidates so far "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return dedupe(out)


def label_candidates(rings, labels: np.ndarray, visible: np.ndarray):
    """Fraud share among each candidate's labelled, *visible* members."""
    keep, y, shares = [], [], []
    for item in rings:
        m = item[0].members
        lab = labels[m][visible[m]]
        known = int((lab == 1).sum() + (lab == 0).sum())
        if known < MIN_LABELLED:
            continue
        share = float((lab == 1).sum() / known)
        keep.append(item)
        y.append(1 if share >= POSITIVE_FRAUD_SHARE else 0)
        shares.append(round(share, 4))
    return keep, np.array(y, dtype=np.int8), shares


def build_edge_index(cfg: Config, tag: str, n: int) -> dict:
    proc = cfg.abs_path(cfg.paths.processed)
    e = pq.read_table(proc / f"edges_week2_{tag}.parquet",
                      columns=["src", "dst", "relation_mask", "min_entity_size"])
    return {
        "src": e["src"].to_numpy().astype(np.int64),
        "dst": e["dst"].to_numpy().astype(np.int64),
        "mask": e["relation_mask"].to_numpy().astype(np.int32),
        "entity": e["min_entity_size"].to_numpy().astype(np.float64),
        "inside": np.zeros(n, dtype=bool),
    }
