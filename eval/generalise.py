"""Does any of this work on a graph that is not PPA?

Every number elsewhere in this project comes from one dataset, from one
platform, in one country. That is a real limit, and the cheapest way to test it
is to run the whole pipeline unchanged on fraud graphs that share the shape but
nothing else.

**Amazon** (11,944 reviewers, 9.5% fraudulent) and **YelpChi** (45,954 reviews,
14.5% fraudulent) come from Dou et al., CIKM 2020, and are two of the ten
datasets in GADBench. They are multi-relation graphs with node labels, which is
the structure Orbweaver assumes:

    Amazon    net_upu  reviewed at least one same product
              net_usu  same star rating within a week
              net_uvu  top-5% mutual review-text similarity

    YelpChi   net_rur  posted by the same user
              net_rtr  same product in the same month
              net_rsr  same product with the same star rating

**I took the .mat files from the original CARE-GNN release rather than the
GADBench bundle.** GADBench distributes its copies in DGL's serialisation
format, and DGL has no wheel for this machine — the same wall the authors'
checkpoint hit. The underlying data is identical and scipy reads it directly.

**Two honest differences from the PPA evaluation.**

*No temporal dimension.* These files expose adjacency and labels, no
timestamps. The split here is account-disjoint and stratified, but it cannot be
forward in time the way the PPA evaluation is. It is a weaker guarantee and the
numbers should be read as such.

*No entity sizes.* PPA gives raw entity ids, so rarity is measured directly -
how many accounts share this exact promotion. Here only the induced adjacency
survives, so rarity is approximated by the endpoint degrees within a relation:
an edge between two nodes that each have hundreds of neighbours under that
relation is connecting them through something common. Same intent, coarser
instrument, and it is an adaptation rather than the same measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList, extract_rings_batch

DATASETS = {
    "amazon": ("Amazon.mat", ["net_upu", "net_usu", "net_uvu"]),
    "yelpchi": ("YelpChi.mat", ["net_rur", "net_rtr", "net_rsr"]),
}
RELATION_MEANING = {
    "net_upu": "reviewed the same product", "net_usu": "same rating that week",
    "net_uvu": "near-identical review text", "net_rur": "posted by the same user",
    "net_rtr": "same product that month", "net_rsr": "same product, same rating",
}


def load_mat(name: str, cfg: Config):
    import scipy.io as sio
    import scipy.sparse as sp

    fname, relations = DATASETS[name]
    path = cfg.abs_path(cfg.paths.raw).parent / "gadbench" / fname
    mat = sio.loadmat(str(path))
    labels = np.asarray(mat["label"]).ravel().astype(np.int8)
    feats = np.asarray(mat["features"].todense(), dtype=np.float32)

    per_rel = {}
    for r in relations:
        a = sp.triu(sp.coo_matrix(mat[r]), k=1).tocoo()   # undirected, no self-loops
        per_rel[r] = (a.row.astype(np.int64), a.col.astype(np.int64))
    return labels, feats, per_rel, relations


def build_edges(per_rel: dict, alphas: dict, n_nodes: int,
                rarity_base: float) -> tuple[EdgeList, np.ndarray]:
    """Weighted edge list, aggregated over relations.

    Rarity is approximated from endpoint degree within the relation, because
    the entity ids that PPA exposes are not recoverable from an adjacency.
    """
    src_all, dst_all, w_all, mask_all = [], [], [], []
    for bit, (rel, (r, c)) in enumerate(per_rel.items()):
        deg = np.bincount(np.concatenate([r, c]), minlength=n_nodes).astype(np.float64)
        common = np.minimum(deg[r], deg[c])
        w = float(alphas.get(rel, 1.0)) / np.log(rarity_base + common)
        src_all.append(r); dst_all.append(c); w_all.append(w)
        mask_all.append(np.full(r.size, 1 << bit, dtype=np.int32))

    src = np.concatenate(src_all); dst = np.concatenate(dst_all)
    w = np.concatenate(w_all); mask = np.concatenate(mask_all)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = lo * n_nodes + hi
    order = np.argsort(key, kind="stable")
    key, lo, hi, w, mask = key[order], lo[order], hi[order], w[order], mask[order]
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    agg_w = np.add.reduceat(w, first)
    agg_mask = np.bitwise_or.reduceat(mask, first)
    return EdgeList(lo[first], hi[first], agg_w, n_nodes), agg_mask


def fit_alphas(per_rel: dict, labels: np.ndarray, visible: np.ndarray) -> dict:
    """Same relation-weighting idea as PPA: how much more often does an edge of
    this relation join two fraudsters than chance predicts. Training nodes only."""
    out, lifts = {}, {}
    for rel, (r, c) in per_rel.items():
        m = visible[r] & visible[c]
        n = int(m.sum())
        if n < 200:
            out[rel] = {"edges_labelled": n, "lift": 1.0, "measured": False}
            continue
        a, b = labels[r[m]], labels[c[m]]
        ff = float(((a == 1) & (b == 1)).sum()) / n
        p = (float((a == 1).sum()) + float((b == 1).sum())) / (2 * n)
        lift = ff / (p * p) if p > 0 else 1.0
        out[rel] = {"edges_labelled": n, "lift": round(lift, 4), "measured": True}
        lifts[rel] = lift
    mean = float(np.mean(list(lifts.values()))) if lifts else 1.0
    for rel in out:
        out[rel]["alpha"] = round(out[rel]["lift"] / mean, 4) if out[rel]["measured"] else 1.0
    return out


def graph_features(edges: EdgeList, mask: np.ndarray, n_relations: int,
                   n_nodes: int) -> np.ndarray:
    """The same family of graph aggregates the PPA scorer uses."""
    both = np.concatenate([edges.src, edges.dst])
    w2 = np.concatenate([edges.weight, edges.weight])
    m2 = np.concatenate([mask, mask])
    deg = np.bincount(both, minlength=n_nodes).astype(np.float64)
    wdeg = np.bincount(both, weights=w2, minlength=n_nodes)
    cols = [deg, wdeg, np.where(deg > 0, wdeg / np.maximum(deg, 1), 0.0)]
    for bit in range(n_relations):
        present = ((m2 >> bit) & 1).astype(np.float64)
        cols.append(np.bincount(both, weights=present, minlength=n_nodes))
    other = np.concatenate([edges.dst, edges.src])
    nbr_deg = deg[other]
    cols.append(np.where(deg > 0, np.bincount(both, weights=nbr_deg,
                                              minlength=n_nodes) / np.maximum(deg, 1), 0.0))
    return np.column_stack(cols).astype(np.float32)


def run_one(name: str, cfg: Config) -> dict:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    from eval.metrics import evaluate

    labels, feats, per_rel, relations = load_mat(name, cfg)
    n = labels.size
    rng = np.random.default_rng(cfg.seed)

    # Account-disjoint, stratified. No temporal axis exists in these files.
    train, test = [], []
    for cls in (0, 1):
        members = np.flatnonzero(labels == cls)
        perm = rng.permutation(members.size)
        cut = int(round(cfg.scoring.heldout_fraction * members.size))
        test.append(members[perm[:cut]]); train.append(members[perm[cut:]])
    test = np.sort(np.concatenate(test)); train = np.sort(np.concatenate(train))
    visible = np.zeros(n, dtype=bool); visible[train] = True

    alphas_info = fit_alphas(per_rel, labels, visible)
    alphas = {r: v["alpha"] for r, v in alphas_info.items()}
    edges, mask = build_edges(per_rel, alphas, n, cfg.graph.rarity_base)

    X = np.hstack([feats, graph_features(edges, mask, len(relations), n)])
    y = (labels == 1).astype(np.int8)

    params = cfg.scoring.xgb.model_dump()
    n_est = params.pop("n_estimators")
    pos, neg = int(y[train].sum()), int((y[train] == 0).sum())
    model = xgb.XGBClassifier(**params, n_estimators=n_est,
                              scale_pos_weight=neg / max(pos, 1),
                              objective="binary:logistic", eval_metric="aucpr",
                              random_state=cfg.seed)
    model.fit(X[train], y[train], verbose=False)
    raw = model.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw[train], y[train])
    scores = iso.predict(raw)

    node_metrics = evaluate(y[test], scores[test])
    base = float(y[test].mean())

    # Rings, at the same operating points as the PPA run.
    rings_out = {}
    for tau in (0.0, 0.3, 0.5):
        keep = scores > tau if tau > 0 else np.ones(n, dtype=bool)
        m = keep[edges.src] & keep[edges.dst]
        sub = EdgeList(edges.src[m], edges.dst[m], edges.weight[m], n)
        if sub.src.size == 0:
            continue
        rings = extract_rings_batch(sub, scores, lambda_=cfg.rings.lambda_default,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
        if not rings:
            rings_out[str(tau)] = {"tau": tau, "n_rings": 0}
            continue
        members = np.unique(np.concatenate([r.members for r in rings]))
        in_test = members[np.isin(members, test)]
        lab = y[members]
        prec = float(lab.mean())
        rings_out[str(tau)] = {
            "tau": tau, "n_rings": len(rings),
            "accounts_in_rings": int(members.size),
            "ring_precision": round(prec, 4),
            "precision_lift_over_base": round(prec / base, 3) if base else None,
            "heldout_members": int(in_test.size),
            "heldout_precision": round(float(y[in_test].mean()), 4) if in_test.size else None,
            "normal_flagged_per_fraud_caught": round(
                float((lab == 0).sum() / max((lab == 1).sum(), 1)), 3),
        }

    return {
        "dataset": name,
        "nodes": int(n),
        "edges": int(edges.src.size),
        "relations": {r: RELATION_MEANING.get(r, r) for r in relations},
        "anomaly_rate": round(float(y.mean()), 4),
        "heldout_base_rate": round(base, 4),
        "relation_weights": alphas_info,
        "node_scoring_heldout": node_metrics,
        "rings": rings_out,
    }


def available(cfg: Config) -> list[str]:
    base = cfg.abs_path(cfg.paths.raw).parent / "gadbench"
    return [n for n, (f, _) in DATASETS.items() if (base / f).exists()]


def main() -> None:
    cfg = load_config()
    have = available(cfg)
    if not have:
        # Optional stage: reproduce should not fail on a machine that has not
        # fetched these. `make download-gadbench` gets them.
        print("no generalisation datasets found; run `make download-gadbench` "
              "to include this stage. Skipping.")
        return
    out = {"note": ("Account-disjoint and stratified, but NOT forward in time: "
                    "these files carry no timestamps. Rarity is approximated "
                    "from endpoint degree because the entity ids are not "
                    "recoverable from an adjacency."),
           "datasets": {}}
    for name in have:
        print(f"=== {name} ===")
        r = run_one(name, cfg)
        out["datasets"][name] = r
        nm = r["node_scoring_heldout"]
        print(f"  {r['nodes']:,} nodes, {r['edges']:,} edges, "
              f"anomaly rate {r['anomaly_rate']}")
        print(f"  relation lifts: " + ", ".join(
            f"{k}={v['lift']}" for k, v in r["relation_weights"].items()))
        print(f"  node scoring (held out): AUPRC {nm['auprc']} "
              f"({nm['auprc_lift_over_random']}x random)")
        for t, b in r["rings"].items():
            if b.get("n_rings"):
                print(f"  tau={b['tau']}: {b['n_rings']} rings, "
                      f"{b['accounts_in_rings']} accounts, precision "
                      f"{b['ring_precision']} ({b['precision_lift_over_base']}x base)")
    dest = cfg.abs_path(cfg.paths.processed) / "generalisation.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
