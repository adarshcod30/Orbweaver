"""Extract rings from the week-2 late-window graph and measure them.

Sweeps lambda, which trades graph structure against model suspicion in the
peeling objective. lambda = 0 uses no model output at all - the rings come out
of the graph alone - so it is both a baseline and a demonstration that the
decision path does not require a learned model.

Ring quality is measured on the held-out accounts, which the scorer never saw
in training or calibration.
"""
from __future__ import annotations

import json
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


def load_edges(tag: str, cfg, n_nodes: int) -> EdgeList:
    proc = cfg.abs_path(cfg.paths.processed)
    e = pq.read_table(proc / f"edges_week2_{tag}.parquet",
                      columns=["src", "dst", "weight"])
    return EdgeList(e["src"].to_numpy().astype(np.int64),
                    e["dst"].to_numpy().astype(np.int64),
                    e["weight"].to_numpy().astype(np.float64), n_nodes)


def main() -> None:
    cfg = load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size

    scores = np.zeros(n, dtype=np.float64)
    score_path = proc / "scores_week2.parquet"
    if score_path.exists():
        s = pq.read_table(score_path)
        scores[s["user_id"].to_numpy()] = s["score"].to_numpy()
    else:
        print("no scores found; running structure-only (lambda has no effect)")

    edges = load_edges(LATE, cfg, n)

    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n, dtype=np.float64)
    uid = f["user_id"].to_numpy()
    keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    report = {
        "seed": cfg.seed,
        "graph": {"window": LATE, "edges": int(edges.src.size),
                  "n_max": cfg.graph.n_max},
        "k_min": cfg.rings.k_min, "k_max": cfg.rings.k_max,
        "top_k": cfg.rings.top_k,
        "assumptions": {
            "assumed_avg_order_value_inr": cfg.cost.assumed_avg_order_value_inr,
            "assumed_avg_promo_value_inr": cfg.cost.assumed_avg_promo_value_inr,
            "note": "PPA ships no monetary amounts; both figures are assumptions.",
        },
        "lambda_sweep": {},
    }

    for lam in cfg.rings.lambda_sweep:
        t0 = time.time()
        rings = extract_rings_batch(edges, scores, lambda_=lam,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
        elapsed = time.time() - t0
        block = evaluate_rings(rings, split.labels, ltv, restrict_to=split.test)
        block["runtime_seconds"] = round(elapsed, 1)
        block["lambda"] = lam
        report["lambda_sweep"][str(lam)] = block
        print(f"lambda={lam:<4} rings={block.get('n_rings',0):>4} "
              f"accounts={block.get('accounts_in_rings',0):>6,} "
              f"ring_precision={block.get('ring_precision')} "
              f"ring_recall={block.get('ring_recall')} "
              f"normal_per_fraud={block.get('normal_flagged_per_fraud_caught')} "
              f"({elapsed:.0f}s)")

        if lam == cfg.rings.lambda_default and rings:
            orders = load_ring_orders(2, cfg)
            lookup = build_size_lookup(2, cfg)
            cases = []
            for r in sorted(rings, key=lambda x: x.size, reverse=True)[:10]:
                ev = ring_evidence(r.members, orders, cfg, lookup)
                ev["rank"] = r.rank
                ev["density"] = round(float(r.density), 4)
                ev["members_sample"] = r.members[:25].tolist()
                cases.append(ev)
            report["case_files"] = cases
            del orders, lookup

    dest = proc / "ring_report.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
