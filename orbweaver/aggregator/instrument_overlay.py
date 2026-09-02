"""Simulated relation: what a shared payment instrument would be worth.

**Everything this module produces is a simulation and is labelled as one.**
PPA has no payment-level relation at all — its eight relations are entirely
platform-native (location, links, delivery, store, group, promotion, coupon,
stimulation). No card token, no UPI VPA, no bank account. That absence is not
an oversight in the dataset; it is a structural fact about who collected it. A
single merchant sees only its own payments.

A payment aggregator does not. It sees the same instrument across every
merchant it serves, which makes an edge nobody else can build: *these two
accounts, on two different platforms, paid with the same card.*

I cannot test that on this data, because the data cannot contain it. What I
can do is a sensitivity analysis: overlay a synthetic instrument relation on
the **real** graph structure, sweep how strongly it correlates with the real
fraud labels, and measure what the pipeline gains at each setting. That
answers a bounded question — *if* an aggregator's payment edge linked ring
members at rate p_f while linking ordinary households at rate p_n, how much
would ring detection improve? — and nothing more.

**What this does not show.** It does not show that Orbweaver performs better
than the numbers in `docs/results.md`. It does not estimate p_f or p_n for any
real system; those are swept precisely because they are unknown. The generated
edges are conditioned on the labels, so a high p_f *builds in* the signal it
then measures — the experiment quantifies the value of a relation given an
assumed strength, it does not discover that the relation is valuable. Read the
gradient across the sweep, never a single cell as a result.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList

FRAUD, NORMAL = 1, 0
SIMULATED_LABEL = "Simulated relation — sensitivity analysis"

# Households genuinely share cards, so instrument groups among ordinary
# accounts are small. Fraud operators reuse a few instruments across many
# accounts, which is the whole reason the edge would be informative.
NORMAL_GROUP = 4
FRAUD_GROUP = 12


def build_instrument_edges(labels: np.ndarray, components: list[np.ndarray],
                           p_fraud: float, p_normal: float, seed: int,
                           rarity_base: float = 2.0) -> EdgeList:
    """Synthesise a shared-payment-instrument relation.

    Fraud-group members are linked into instrument groups with probability
    `p_fraud`; ordinary accounts with probability `p_normal`, in much smaller
    groups. Edges are weighted by the same rarity rule as every real relation,
    so the overlay competes on equal terms rather than being handed extra mass.
    """
    rng = np.random.default_rng(seed)
    n = labels.size
    src, dst, w = [], [], []

    def link(members: np.ndarray, group_size: int) -> None:
        if members.size < 2:
            return
        rng.shuffle(members)
        for i in range(0, members.size, group_size):
            grp = members[i:i + group_size]
            if grp.size < 2:
                continue
            a, b = np.triu_indices(grp.size, k=1)
            src.append(grp[a]); dst.append(grp[b])
            w.append(np.full(a.size, 1.0 / np.log(rarity_base + grp.size)))

    # Fraud groups: some share of each real group is put on one instrument.
    for comp in components:
        chosen = comp[rng.random(comp.size) < p_fraud]
        link(chosen.copy(), FRAUD_GROUP)

    # Ordinary accounts: household-scale sharing, at a much lower rate.
    normals = np.flatnonzero(labels == NORMAL)
    chosen = normals[rng.random(normals.size) < p_normal]
    link(chosen.copy(), NORMAL_GROUP)

    if not src:
        empty = np.empty(0, dtype=np.int64)
        return EdgeList(empty, empty, np.empty(0), n)
    s = np.concatenate(src).astype(np.int64)
    d = np.concatenate(dst).astype(np.int64)
    lo, hi = np.minimum(s, d), np.maximum(s, d)
    return EdgeList(lo, hi, np.concatenate(w), n)


def run_overlay(cfg: Config | None = None) -> dict:
    from eval.run_rings import load_edges, prune
    from eval.split import make_split
    from orbweaver.adversarial.fragment import fraud_components
    from orbweaver.rings.cost import evaluate_rings
    from orbweaver.rings.peel import extract_rings_batch
    from eval.metrics import ltv_proxy

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size
    labels = split.labels

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

    base_edges = load_edges("late", cfg, n)
    comps = fraud_components(labels, base_edges.src, base_edges.dst)

    def measure(edges: EdgeList) -> dict:
        sub = prune(edges, scores, best["tau"])
        rings = extract_rings_batch(sub, scores, lambda_=best["lambda"],
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=top_k, g_min=cfg.rings.g_min)
        return evaluate_rings(rings, labels, ltv, restrict_to=split.test)

    baseline = measure(base_edges)
    print(f"baseline (no simulated relation): precision "
          f"{baseline['ring_precision']}  recall {baseline['ring_recall']}  "
          f"fraud {baseline['fraud_members']}")

    grid = {}
    print(f"\n{SIMULATED_LABEL}")
    print(f"{'p_fraud':>8} {'p_normal':>9} {'sim edges':>11} {'precision':>10} "
          f"{'d prec':>8} {'recall':>9} {'d recall':>9} {'fraud':>7}")
    for p_f in (0.3, 0.5, 0.7):
        for p_n in (0.02, 0.05, 0.10):
            sim = build_instrument_edges(labels, comps, p_f, p_n, cfg.seed,
                                         cfg.graph.rarity_base)
            merged = EdgeList(
                np.concatenate([base_edges.src, sim.src]),
                np.concatenate([base_edges.dst, sim.dst]),
                np.concatenate([base_edges.weight, sim.weight]), n)
            m = measure(merged)
            d_prec = (round(m["ring_precision"] - baseline["ring_precision"], 4)
                      if m.get("ring_precision") and baseline.get("ring_precision")
                      else None)
            d_rec = (round(m["ring_recall"] - baseline["ring_recall"], 6)
                     if m.get("ring_recall") is not None
                     and baseline.get("ring_recall") is not None else None)
            grid[f"p_fraud={p_f},p_normal={p_n}"] = {
                "p_fraud": p_f, "p_normal": p_n,
                "simulated_edges": int(sim.src.size),
                "ring_precision": m.get("ring_precision"),
                "delta_precision": d_prec,
                "ring_recall": m.get("ring_recall"),
                "delta_recall": d_rec,
                "fraud_members": m.get("fraud_members"),
                "fp_cost_inr": m.get("fp_cost_inr"),
                "label": SIMULATED_LABEL,
            }
            print(f"{p_f:>8} {p_n:>9} {sim.src.size:>11,} "
                  f"{str(m.get('ring_precision')):>10} {str(d_prec):>8} "
                  f"{str(m.get('ring_recall')):>9} {str(d_rec):>9} "
                  f"{m.get('fraud_members', 0):>7}")

    return {
        "label": SIMULATED_LABEL,
        "operating_point": best,
        "baseline": baseline,
        "grid": grid,
        "caveat": (
            "The simulated edges are generated conditional on the fraud labels, "
            "so a high p_fraud builds in the signal it then measures. This "
            "quantifies what a payment-instrument relation would be worth at an "
            "assumed strength; it does not discover that it is valuable, and no "
            "cell is a claim about Orbweaver's real performance."),
    }


def main() -> None:
    cfg = load_config()
    out = run_overlay(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "aggregator_overlay.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
