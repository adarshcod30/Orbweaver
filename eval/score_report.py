"""Train the scorer and report detection numbers, with the caveats attached.

Four blocks come out of this, and reporting fewer would mislead:

- held-out accounts, labelled only    <- the honest headline
- training-pool accounts, labelled    <- the gap between these two is memorisation
- held-out, unlabelled counted normal <- the only block comparable to the paper
- all accounts, unlabelled as normal  <- the same convention over everyone

Every block carries its false-positive count and the rupee cost of those
false positives.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from eval.metrics import evaluate, ltv_proxy
from eval.split import make_split
from orbweaver.config import load_config
from orbweaver.data.windows import EARLY, LATE, week2_windows
from orbweaver.scoring.xgb_graph import fit_scorer


def main() -> None:
    from datetime import date

    cfg = load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n_users = split.labels.size

    result = fit_scorer(cfg, split)
    scores = result.scores

    # Customer-value proxy from the early window's order counts. An
    # assumption, and labelled as one wherever it appears.
    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet",
                      columns=["user_id", "n_orders"])
    orders = np.zeros(n_users, dtype=np.float64)
    uid = f["user_id"].to_numpy()
    keep = uid < n_users
    orders[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders, cfg.cost.assumed_avg_order_value_inr)

    windows = {k: [date.fromordinal(v[0]).isoformat(),
                   date.fromordinal(v[1]).isoformat()]
               for k, v in week2_windows(cfg).items()}

    report: dict = {
        "seed": cfg.seed,
        "protocol": {
            "design": ("Account-disjoint within week 2, and forward in time: "
                       "training features and graph from the early window, "
                       "held-out accounts scored on the late window."),
            "windows": windows,
            "why_not_week1_to_week2": (
                "The two order files are separately re-indexed from zero, so no "
                "key joins an account across them. See docs/data.md finding E."),
        },
        "split": split.summary(),
        "assumptions": {
            "assumed_avg_order_value_inr": cfg.cost.assumed_avg_order_value_inr,
            "note": ("PPA ships no monetary amounts. The rupee costs below are an "
                     "order count times an assumed average order value, and are "
                     "meaningful only as a relative ranking of customers."),
        },
        "best_iteration": result.best_iteration,
        "feature_importance_top10": dict(list(result.feature_importance.items())[:10]),
        "results": {},
    }

    for name, idx in (("test_heldout", split.test),
                      ("train_pool", split.train_pool)):
        report["results"][f"{name}__labelled_only"] = evaluate(
            split.y(idx), scores[idx], ltv[idx])

    # The convention the authors' test.py uses: unlabelled accounts count as
    # negatives. Only in this convention is a comparison to their published
    # 0.9107 / 0.6992 / 0.7911 like-for-like.
    y_all = (split.labels == 1).astype(np.int8)
    mask = np.zeros(n_users, dtype=bool)
    mask[split.test] = True
    unlabelled = split.labels == -1
    held_or_unlabelled = mask | unlabelled
    report["results"]["test_heldout_plus_unlabelled__as_normal"] = evaluate(
        y_all[held_or_unlabelled], scores[held_or_unlabelled], ltv[held_or_unlabelled])
    report["results"]["all_accounts__unlabelled_as_normal"] = evaluate(
        y_all, scores, ltv)

    report["reference"] = {
        "promoguardian_reported": {"precision": 0.9107, "recall": 0.6992, "f1": 0.7911},
        "comparability": (
            "Their test.py counts unlabelled accounts as negatives across all "
            "3,267,961 nodes and applies no account holdout, so their setting is "
            "closest to all_accounts__unlabelled_as_normal evaluated on accounts "
            "the model has seen. The test_heldout blocks are strictly harder."),
    }

    dest = proc / "score_report.json"
    dest.write_text(json.dumps(report, indent=2))

    def line(tag: str, b: dict) -> str:
        a = b["at_best_f1"]
        return (f"  {tag:46s} n={b['n']:>9,} base={b['base_rate']:.4f} "
                f"AUPRC={b['auprc']:.4f} (x{b['auprc_lift_over_random']:.2f})  "
                f"P={a['precision']:.4f} R={a['recall']:.4f} F1={a['f1']:.4f}  "
                f"FP={a['fp']:,} FP/TP={a['false_positives_per_true_positive']}")

    print(f"windows: early {windows[EARLY]}  ->  late {windows[LATE]}")
    print("detection results (at the best-F1 threshold)")
    for k, v in report["results"].items():
        print(line(k, v))
    print("\ntop features:", ", ".join(list(result.feature_importance)[:6]))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
