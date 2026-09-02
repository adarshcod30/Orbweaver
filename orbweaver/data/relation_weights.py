"""Fit a per-relation weight from how much each relation actually predicts fraud.

The entity-rarity weight `1/log(2 + |users(e)|)` asks how many people share an
entity. It has no way of knowing that sharing a delivery record is more
incriminating than sharing a promotion. On this graph that omission dominates
everything else:

    relation   edges       fraud-fraud lift
    r1         5,738,406   3.68x     location
    r7         1,143,574   2.75x     coupon type
    r8         4,247,934   2.55x     sales stimulation
    r3             7,533   2.46x     delivery record
    r6        24,864,095   1.76x     promotion

**r6 is 70% of all edges and carries the weakest signal.** So the densest
subgraphs are promotion cliques, most of which are ordinary customers who
happened to use the same offer, and ring precision lands below the base rate.

This module measures each relation's fraud assortativity - how much more often
an edge of that relation joins two fraudsters than chance would predict - and
turns it into a multiplier on the edge weight:

    w_r(e) = alpha_r / log(rarity_base + |users(e)|)

**Fitted on training accounts only.** The held-out accounts are excluded, so
this is model fitting on the training split like any other parameter, not a
peek at the test set. `fit_relation_weights` takes the split and refuses to
look outside `split.train` and `split.val`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

FRAUD, NORMAL = 1, 0
# Below this many labelled edges a lift estimate is noise, so the relation
# falls back to a neutral multiplier rather than an over-fitted one.
MIN_LABELLED_EDGES = 500


def fit_relation_weights(cfg: Config | None = None, split=None,
                         graph_tag: str = "early") -> dict:
    """Measure each relation's fraud-assortativity lift on training accounts.

    Uses the EARLY window graph by default: the weights are a training-time
    parameter, so they must be fitted on the same window the model trains on,
    not on the window used to score.
    """
    from eval.split import make_split

    cfg = cfg or load_config()
    split = split or make_split(cfg)
    proc = cfg.abs_path(cfg.paths.processed)

    labels = split.labels
    # Only accounts the model is allowed to learn from.
    visible = np.zeros(labels.size, dtype=bool)
    visible[split.train] = True
    visible[split.val] = True
    if visible[split.test].any():
        raise RuntimeError("held-out accounts are visible to the weight fit")

    e = pq.read_table(proc / f"edges_week2_{graph_tag}.parquet",
                      columns=["src", "dst", "relation_mask"])
    src, dst = e["src"].to_numpy(), e["dst"].to_numpy()
    rmask = e["relation_mask"].to_numpy()
    del e

    both = visible[src] & visible[dst]
    ls, ld = labels[src], labels[dst]

    out: dict[str, dict] = {}
    for bit, rel in enumerate(cfg.data.buildable_relations):
        m = (((rmask >> bit) & 1) == 1) & both
        n = int(m.sum())
        if n < MIN_LABELLED_EDGES:
            out[rel] = {"edges_labelled": n, "lift": 1.0, "alpha": 1.0,
                        "note": "too few labelled edges to estimate; neutral weight"}
            continue
        a, b = ls[m], ld[m]
        ff = float(((a == FRAUD) & (b == FRAUD)).sum()) / n
        p = (float((a == FRAUD).sum()) + float((b == FRAUD).sum())) / (2 * n)
        expected = p * p
        lift = ff / expected if expected > 0 else 1.0
        out[rel] = {"edges_labelled": n,
                    "fraud_fraud_rate": round(ff, 6),
                    "expected_if_random": round(expected, 6),
                    "lift": round(lift, 4)}

    # Normalise so the mean multiplier is 1: this re-balances relations
    # against each other without inflating densities overall, which keeps
    # g_min and k_max comparable across runs.
    #
    # Relations with too little labelled data are normalised *out* and pinned
    # at a neutral 1.0. Including them at a placeholder lift of 1.0 would drag
    # the mean and then scale them below neutral - which is how r3, the rarest
    # and most incriminating relation in the data (never more than seven users
    # per entity), first came out at alpha 0.465 and was penalised for being
    # too rare to measure.
    measured = [r for r in cfg.data.buildable_relations
                if out[r].get("edges_labelled", 0) >= MIN_LABELLED_EDGES]
    if measured:
        mean_lift = float(np.mean([out[r]["lift"] for r in measured]))
        for rel in cfg.data.buildable_relations:
            out[rel]["alpha"] = (round(out[rel]["lift"] / mean_lift, 4)
                                 if rel in measured else 1.0)
    else:
        for rel in cfg.data.buildable_relations:
            out[rel]["alpha"] = 1.0

    manifest = {
        "fitted_on": f"week2_{graph_tag}",
        "accounts_visible": int(visible.sum()),
        "heldout_excluded": int(split.test.size),
        "relations": out,
        "note": ("alpha multiplies the entity-rarity edge weight. Fitted on "
                 "training and validation accounts only."),
    }
    (proc / "relation_weights.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_relation_weights(cfg: Config) -> dict[str, float] | None:
    """Fitted multipliers, or None when they have not been fitted yet."""
    path = cfg.abs_path(cfg.paths.processed) / "relation_weights.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    return {rel: float(v["alpha"]) for rel, v in m["relations"].items()}


def main() -> None:
    m = fit_relation_weights()
    print(f"fitted on {m['accounts_visible']:,} training accounts "
          f"({m['heldout_excluded']:,} held out and excluded)")
    print(f"{'relation':10s} {'labelled edges':>15s} {'lift':>8s} {'alpha':>8s}")
    for rel, v in m["relations"].items():
        print(f"{rel:10s} {v['edges_labelled']:>15,} {v['lift']:>8.3f} {v['alpha']:>8.3f}")


if __name__ == "__main__":
    main()
