"""Extract rings from the week-2 late-window graph and measure them.

Sweeps two knobs and reports the whole grid rather than a chosen cell.

**tau** prunes the graph to accounts the scorer finds suspicious before any
peeling happens. This matters more than anything else in the pipeline: on the
unpruned graph the densest subgraphs are large ordinary communities - people
who happened to use the same promotion - and ring precision lands *below* the
base rate. Pruning first, then looking for dense structure inside the
suspicious region, is what makes the output useful.

**lambda** trades structure against model suspicion inside the peeling
objective. At lambda = 0 no model output enters the objective at all: the
model has chosen the candidate set, and the deterministic density objective
alone decides who is in the ring.

Ring quality is reported over all labelled members (a usable sample) and over
held-out members only (strict, but often too few to estimate from).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from eval.metrics import ltv_proxy
from eval.split import make_split
from orbweaver.config import load_config
from orbweaver.data.windows import EARLY, LATE
from orbweaver.rings.cost import evaluate_rings
from orbweaver.rings.evidence import build_size_lookup, load_ring_orders, ring_evidence
from orbweaver.rings.peel import EdgeList, extract_rings_batch

# tau = 0 means no pruning: the full graph, which is the honest baseline
# showing why pruning is needed.
TAU_SWEEP = [0.0, 0.3, 0.5]


def load_edges(tag: str, cfg, n_nodes: int) -> EdgeList:
    proc = cfg.abs_path(cfg.paths.processed)
    e = pq.read_table(proc / f"edges_week2_{tag}.parquet",
                      columns=["src", "dst", "weight"])
    return EdgeList(e["src"].to_numpy().astype(np.int64),
                    e["dst"].to_numpy().astype(np.int64),
                    e["weight"].to_numpy().astype(np.float64), n_nodes)


def prune(edges: EdgeList, scores: np.ndarray, tau: float) -> EdgeList:
    """Restrict the graph to accounts the scorer finds suspicious."""
    if tau <= 0:
        return edges
    keep = scores > tau
    m = keep[edges.src] & keep[edges.dst]
    return EdgeList(edges.src[m], edges.dst[m], edges.weight[m], edges.n_nodes)


def main() -> None:
    cfg = load_config()
    top_k = int(os.environ.get("ORBWEAVER_TOP_K", cfg.rings.top_k))
    # Restrict the grid to one cell, for a deep run at a chosen operating point.
    only_tau = os.environ.get("ORBWEAVER_TAU")
    only_lam = os.environ.get("ORBWEAVER_LAMBDA")
    out_name = os.environ.get("ORBWEAVER_OUT", "ring_report.json")
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size

    scores = np.zeros(n, dtype=np.float64)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    edges = load_edges(LATE, cfg, n)

    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n, dtype=np.float64)
    uid = f["user_id"].to_numpy()
    keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    labelled = split.labels != -1
    base_rate = float((split.labels[labelled] == 1).mean())

    report = {
        "seed": cfg.seed,
        "graph": {"window": LATE, "edges": int(edges.src.size),
                  "n_max": cfg.graph.n_max, "k_min": cfg.rings.k_min,
                  "k_max": cfg.rings.k_max, "top_k": top_k},
        "base_rate_among_labelled": round(base_rate, 4),
        "assumptions": {
            "assumed_avg_order_value_inr": cfg.cost.assumed_avg_order_value_inr,
            "assumed_avg_promo_value_inr": cfg.cost.assumed_avg_promo_value_inr,
            "note": "PPA ships no monetary amounts; both figures are assumptions.",
        },
        "grid": {},
    }

    print(f"base rate among labelled accounts: {base_rate:.4f}")
    print(f"{'tau':>5} {'lam':>5} {'rings':>6} {'accts':>7} {'lab':>6} "
          f"{'fraud':>6} {'prec':>7} {'lift':>6} {'norm/fraud':>11} {'fp_cost':>12} {'s':>5}")

    best = None
    taus = [float(only_tau)] if only_tau else TAU_SWEEP
    lams = [float(only_lam)] if only_lam else cfg.rings.lambda_sweep
    for tau in taus:
        sub = prune(edges, scores, tau)
        if sub.src.size == 0:
            continue
        for lam in lams:
            t0 = time.time()
            rings = extract_rings_batch(sub, scores, lambda_=lam,
                                        k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                        top_k=top_k, g_min=cfg.rings.g_min)
            elapsed = time.time() - t0
            block = evaluate_rings(rings, split.labels, ltv, restrict_to=split.test)
            block.update({"tau": tau, "lambda": lam,
                          "edges_after_prune": int(sub.src.size),
                          "runtime_seconds": round(elapsed, 1)})
            p = block.get("ring_precision")
            block["precision_lift_over_base"] = (
                round(p / base_rate, 3) if p and base_rate else None)
            key = f"tau={tau},lambda={lam}"
            report["grid"][key] = block
            print(f"{tau:>5} {lam:>5} {block.get('n_rings', 0):>6} "
                  f"{block.get('accounts_in_rings', 0):>7,} "
                  f"{block.get('labelled_members', 0):>6} "
                  f"{block.get('fraud_members', 0):>6} "
                  f"{str(p):>7} {str(block['precision_lift_over_base']):>6} "
                  f"{str(block.get('normal_flagged_per_fraud_caught')):>11} "
                  f"{block.get('fp_cost_inr', 0):>12,.0f} {elapsed:>5.0f}")
            if p is not None and (best is None or p > best[0]):
                best = (p, tau, lam, rings)

    if best:
        p, tau, lam, rings = best
        report["best_cell"] = {"tau": tau, "lambda": lam, "ring_precision": p}
        orders = load_ring_orders(2, cfg)
        lookup = build_size_lookup(2, cfg)
        cases = []
        # Build evidence for every ring, then rank by how *readable* the case
        # is, not by size. The largest ring is the loosest: a 233-account ring
        # shares nothing with more than 12% of its members, while a 38-account
        # one has 60% sharing a stimulation only 35 accounts on the platform
        # have ever used. The second is a case an analyst can act on.
        for r in rings:
            ev = ring_evidence(r.members, orders, cfg, lookup)
            ev["rank"] = r.rank
            ev["density"] = round(float(r.density), 4)
            lab = split.labels[r.members]
            ev["labels"] = {"fraud": int((lab == 1).sum()),
                            "normal": int((lab == 0).sum()),
                            "unlabelled": int((lab == -1).sum())}
            ev["members_sample"] = r.members[:25].tolist()
            top = ev["shared_entities"][0] if ev["shared_entities"] else None
            ev["evidence_strength"] = round(
                top["coverage"] * (top["rarity_weight"] or 0), 4) if top else 0.0
            cases.append(ev)
        cases.sort(key=lambda c: (c["labels"]["fraud"] > 0,
                                  c["evidence_strength"],
                                  c["labels"]["fraud"]), reverse=True)
        report["case_files"] = cases[:10]
        print(f"\nbest cell: tau={tau} lambda={lam} ring precision {p:.4f} "
              f"({p / base_rate:.2f}x the base rate)")

    dest = proc / out_name
    dest.write_text(json.dumps(report, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
