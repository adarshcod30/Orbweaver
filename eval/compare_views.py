"""What are the three relations I cannot rebuild actually worth?

`docs/data.md` records that `r2`, `r4` and `r5` have no values at all in the
released order files, while `r4` alone is 38.4% of the edges in the authors'
shipped `edge.csv`. So I can build five of the eight relations, and they can
build eight.

Rather than note that as a caveat and move on, this measures it. The same
extractor, the same scores, the same operating point, run over two graphs:

- **View A — mine.** Built from week-2 orders, five relations, my own
  entity-rarity weights with the fitted per-relation multipliers, entities
  capped at 100 accounts.
- **View B — theirs.** `edge.csv` exactly as shipped: eight relations, their
  weights, already deduplicated.

The gap between the two is the value of the three relations I cannot
reconstruct — and, since the weighting schemes also differ, of their edge
weighting against mine. Those two effects are not separable here, which is
stated rather than glossed.

It is also a generalisation check for the extractor: View B is a graph built
by someone else, with different construction rules and a different weight
distribution. If ring extraction only works on graphs I built myself, that is
worth knowing.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList


def load_view_b(cfg: Config, n_nodes: int) -> EdgeList:
    """The authors' edge.csv. Their per-relation scores are summed into a
    single weight, which is the closest analogue to my aggregated weight."""
    proc = cfg.abs_path(cfg.paths.processed)
    cols = ["src", "dst"] + [f"r{i}_score" for i in range(1, 9)]
    t = pq.read_table(proc / "edges_authors.parquet", columns=cols)
    w = np.zeros(t.num_rows, dtype=np.float64)
    for i in range(1, 9):
        w += t[f"r{i}_score"].to_numpy().astype(np.float64)
    return EdgeList(t["src"].to_numpy().astype(np.int64),
                    t["dst"].to_numpy().astype(np.int64), w, n_nodes)


def run(cfg: Config | None = None) -> dict:
    from eval.metrics import ltv_proxy
    from eval.run_rings import load_edges, prune
    from eval.split import make_split
    from orbweaver.rings.cost import evaluate_rings
    from orbweaver.rings.peel import extract_rings_batch

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size

    scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    f = pq.read_table(proc / "features_week2_early.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    report = json.loads((proc / "ring_report.json").read_text())
    best = report.get("best_cell", {"tau": 0.5, "lambda": 5.0})
    top_k = report["graph"]["top_k"]

    views = {
        "A_mine_5_relations": load_edges("late", cfg, n),
        "B_authors_8_relations": load_view_b(cfg, n),
    }

    out = {"operating_point": best, "views": {}}
    print(f"{'view':<24} {'edges':>12} {'after prune':>12} {'rings':>6} "
          f"{'precision':>10} {'recall':>9} {'fraud':>7} {'norm/fraud':>11}")
    for name, edges in views.items():
        sub = prune(edges, scores, best["tau"])
        rings = extract_rings_batch(sub, scores, lambda_=best["lambda"],
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=top_k, g_min=cfg.rings.g_min)
        block = evaluate_rings(rings, split.labels, ltv, restrict_to=split.test)
        block["edges"] = int(edges.src.size)
        block["edges_after_prune"] = int(sub.src.size)
        out["views"][name] = block
        print(f"{name:<24} {edges.src.size:>12,} {sub.src.size:>12,} "
              f"{block.get('n_rings', 0):>6} {str(block.get('ring_precision')):>10} "
              f"{str(block.get('ring_recall')):>9} "
              f"{block.get('fraud_members', 0):>7} "
              f"{str(block.get('normal_flagged_per_fraud_caught')):>11}")

    a = out["views"]["A_mine_5_relations"]
    b = out["views"]["B_authors_8_relations"]
    if a.get("ring_precision") and b.get("ring_precision"):
        out["delta"] = {
            "precision": round(b["ring_precision"] - a["ring_precision"], 4),
            "recall": round((b.get("ring_recall") or 0) - (a.get("ring_recall") or 0), 6),
            "fraud_members": b.get("fraud_members", 0) - a.get("fraud_members", 0),
        }
    out["caveat"] = (
        "The two views differ in both relation coverage and edge weighting, and "
        "View B covers the whole of week 2 while View A covers its later half. "
        "Those effects are not separable from this comparison; it bounds the "
        "combined difference rather than isolating the three missing relations.")
    return out


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "view_comparison.json"
    dest.write_text(json.dumps(out, indent=2))
    if "delta" in out:
        d = out["delta"]
        print(f"\nEight relations against five: precision {d['precision']:+.4f}, "
              f"{d['fraud_members']:+d} fraud accounts found.")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
