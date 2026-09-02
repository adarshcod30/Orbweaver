"""Train the ring-level model and compare it against the two obvious baselines.

Three ways to order a review queue:

- **density**, which is what the queue does today;
- **mean member score**, the first thing anyone would try;
- **learned confidence**, the model in `orbweaver.rings.ring_scorer`.

All three are reported at 25, 100 and 200 rings, on all labelled members and on
held-out members separately. Leaving out the mean-score baseline would make the
model look better than the honest comparison does, which is the whole reason it
is here.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.data.windows import EARLY, LATE, week2_windows
from orbweaver.rings.peel import EdgeList
from orbweaver.rings.ring_scorer import (
    FEATURE_NAMES, account_aggregates, build_edge_index, candidates,
    label_candidates, oof_account_scores, ring_features,
)

DEPTHS = (25, 100, 200)


def load_edges(cfg: Config, tag: str, n: int) -> EdgeList:
    proc = cfg.abs_path(cfg.paths.processed)
    e = pq.read_table(proc / f"edges_week2_{tag}.parquet",
                      columns=["src", "dst", "weight"])
    return EdgeList(e["src"].to_numpy().astype(np.int64),
                    e["dst"].to_numpy().astype(np.int64),
                    e["weight"].to_numpy().astype(np.float64), n)


def precision_at_depth(ordered, labels: np.ndarray,
                       restrict: np.ndarray | None = None) -> dict:
    """Fraud share among the labelled members of the first N rings."""
    allowed = None
    if restrict is not None:
        allowed = np.zeros(labels.size, dtype=bool)
        allowed[restrict] = True
    out = {}
    seen: set[int] = set()
    fraud = normal = 0
    for i, r in enumerate(ordered, start=1):
        for a in r.members.tolist():
            if a in seen:
                continue
            seen.add(a)
            if allowed is not None and not allowed[a]:
                continue
            if labels[a] == 1:
                fraud += 1
            elif labels[a] == 0:
                normal += 1
        if i in DEPTHS:
            known = fraud + normal
            out[str(i)] = {
                "rings": i, "labelled": known, "fraud": fraud,
                "precision": round(fraud / known, 4) if known else None,
            }
    return out


def run(cfg: Config | None = None) -> dict:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    from eval.split import make_split
    from orbweaver.scoring.xgb_graph import load_features

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    labels = split.labels
    n = labels.size

    # --- candidate rings on the early window, from out-of-fold scores --------
    X_early = load_features(2, cfg, n, EARLY)
    oof = oof_account_scores(cfg, split, X_early)
    visible = np.zeros(n, dtype=bool)
    visible[split.train_pool] = True

    early_edges = load_edges(cfg, EARLY, n)
    early_idx = build_edge_index(cfg, EARLY, n)
    lo_e, hi_e = week2_windows(cfg)[EARLY]
    agg_e = account_aggregates(cfg, 2, (lo_e, hi_e), n)

    print("generating early-window candidates...", flush=True)
    cands = candidates(cfg, early_edges, oof, label="early")
    kept, y, shares = label_candidates(cands, labels, visible)
    print(f"candidate rings: {len(cands):,}  labelled enough to train on: "
          f"{len(kept):,}  positives: {int(y.sum()):,}")
    if len(kept) < 40 or y.sum() < 8:
        return {"trained": False,
                "reason": (f"only {len(kept)} usable candidates with "
                           f"{int(y.sum())} positives - too few to fit a ring "
                           "model on")}

    F = np.vstack([ring_features(r, oof, agg_e, early_idx, tau, lam)
                   for r, tau, lam in kept])

    # --- the ring model -----------------------------------------------------
    # Shallow and small on purpose: a few hundred candidates is not enough data
    # to support anything larger, and a deep model here would memorise the
    # candidate set rather than learn what a ring looks like.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    oof_conf = np.zeros(len(kept))
    for tr, te in skf.split(F, y):
        m = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                              subsample=0.9, colsample_bytree=0.9,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=cfg.seed)
        m.fit(F[tr], y[tr], verbose=False)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(m.predict_proba(F[tr])[:, 1], y[tr])
        oof_conf[te] = iso.predict(m.predict_proba(F[te])[:, 1])

    final = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                              subsample=0.9, colsample_bytree=0.9,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=cfg.seed)
    final.fit(F, y, verbose=False)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(final.predict_proba(F)[:, 1], y)

    # --- apply to the late window ------------------------------------------
    late_edges = load_edges(cfg, LATE, n)
    late_idx = build_edge_index(cfg, LATE, n)
    lo_l, hi_l = week2_windows(cfg)[LATE]
    agg_l = account_aggregates(cfg, 2, (lo_l, hi_l), n)

    scores = np.zeros(n)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

    print("generating late-window candidates...", flush=True)
    late_cands = candidates(cfg, late_edges, scores, label="late")
    FL = np.vstack([ring_features(r, scores, agg_l, late_idx, tau, lam)
                    for r, tau, lam in late_cands])
    conf = cal.predict(final.predict_proba(FL)[:, 1])
    contribs = final.get_booster().predict(
        xgb.DMatrix(FL), pred_contribs=True)[:, :-1]

    rings_only = [r for r, _, _ in late_cands]
    mean_score = np.array([scores[r.members].mean() for r in rings_only])
    density = np.array([r.density for r in rings_only])

    rankings = {
        "density": np.argsort(-density),
        "mean_member_score": np.argsort(-mean_score),
        "learned_confidence": np.argsort(-conf),
    }
    results = {}
    for name, order in rankings.items():
        ordered = [rings_only[i] for i in order]
        results[name] = {
            "all_labelled": precision_at_depth(ordered, labels),
            "heldout_only": precision_at_depth(ordered, labels, split.test),
        }

    # --- is the confidence honest? -----------------------------------------
    deciles = []
    q = np.quantile(oof_conf, np.linspace(0, 1, 11))
    for i in range(10):
        m = (oof_conf >= q[i]) & (oof_conf <= q[i + 1] if i == 9 else oof_conf < q[i + 1])
        if m.sum() < 3:
            continue
        deciles.append({"decile": i + 1,
                        "predicted": round(float(oof_conf[m].mean()), 4),
                        "realised": round(float(y[m].mean()), 4),
                        "n": int(m.sum())})

    # --- what the hostel clusters score ------------------------------------
    hostel_conf = None
    hp = proc / "hostel_test.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        touched = {c["entity"] for c in h.get("worst_cases", [])}
        if touched:
            # Confidence of the late rings that overlap a legitimate cluster.
            from orbweaver.rings.hostel_test import find_colocated_clusters
            clusters = find_colocated_clusters(cfg, labels)
            legit = np.concatenate([c["members"] for c in clusters]) if clusters else np.array([])
            legit_set = np.zeros(n, dtype=bool)
            legit_set[legit.astype(np.int64)] = True
            overlap = np.array([float(legit_set[r.members].mean()) for r in rings_only])
            mostly_legit = overlap > 0.5
            hostel_conf = {
                "rings_mostly_inside_a_legitimate_cluster": int(mostly_legit.sum()),
                "their_median_confidence": round(float(np.median(conf[mostly_legit])), 4)
                if mostly_legit.any() else None,
                "median_confidence_of_all_rings": round(float(np.median(conf)), 4),
            }

    top_drivers = []
    for i in np.argsort(-conf)[:10]:
        c = contribs[i]
        idx = np.argsort(-np.abs(c))[:3]
        top_drivers.append({
            "rank": int(i),
            "confidence": round(float(conf[i]), 4),
            "size": int(rings_only[i].size),
            "drivers": [{"feature": FEATURE_NAMES[j],
                         "value": round(float(FL[i, j]), 4),
                         "contribution": round(float(c[j]), 4)} for j in idx],
        })

    return {
        "trained": True,
        "candidates": {"generated": len(cands), "usable": len(kept),
                       "positives": int(y.sum()),
                       "positive_share": round(float(y.mean()), 4),
                       "cells": [{"tau": t_, "lambda": l_} for t_, l_ in
                                 sorted({(t_, l_) for _, t_, l_ in kept})]},
        "late_candidates": len(late_cands),
        "rankings": results,
        "calibration": deciles,
        "hostel_clusters": hostel_conf,
        "top_drivers": top_drivers,
        "feature_names": FEATURE_NAMES,
        "note": ("Candidate rings come from the early window using five-fold "
                 "out-of-fold account scores, so no ring is built from scores "
                 "that had already seen its own members' labels. Held-out "
                 "accounts are used for nothing here except the final "
                 "held-out-only measurement."),
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "ring_scorer.json"
    dest.write_text(json.dumps(out, indent=2))
    if not out.get("trained"):
        print(f"ring model not trained: {out['reason']}")
        print(f"wrote {dest}")
        return
    print(f"late candidates: {out['late_candidates']:,}")
    print(f"\n{'ranking':>20s} " + " ".join(f"{'@' + str(d):>9s}" for d in DEPTHS)
          + "   (held-out in brackets)")
    for name, r in out["rankings"].items():
        cells = []
        for d in DEPTHS:
            a = r["all_labelled"].get(str(d), {})
            h = r["heldout_only"].get(str(d), {})
            cells.append(f"{str(a.get('precision')):>9s}")
        print(f"{name:>20s} " + " ".join(cells))
    if out["hostel_clusters"]:
        h = out["hostel_clusters"]
        print(f"\nrings mostly inside a legitimate cluster: "
              f"{h['rings_mostly_inside_a_legitimate_cluster']}, "
              f"median confidence {h['their_median_confidence']} "
              f"against {h['median_confidence_of_all_rings']} overall")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
