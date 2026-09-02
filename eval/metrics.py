"""Detection metrics, always with the cost of being wrong attached.

Two things here are deliberate and matter more than the formulas.

**Every metric carries its false-positive side.** A precision figure on its
own is not a result. `binary_metrics` always returns the false-positive count
and, when an LTV proxy is supplied, the rupee cost of those false positives,
because wrongly flagging a real customer is a harm and it should appear in the
same row as the win.

**Two labelling conventions, reported separately.** 90.6 % of PPA accounts are
labelled `-1`, meaning unknown rather than normal, and how you treat them
changes precision by an order of magnitude:

- `labelled_only` — score only accounts with a real 0/1 label. The base rate
  is 22.4 %. This is the defensible convention: an unknown account is not
  evidence of anything, and counting it as normal invents negatives.
- `unlabelled_as_normal` — count every `-1` as a negative, over all 3,267,961
  accounts, base rate 2.1 %. This is what the PromoGuardian authors' `test.py`
  does, so it is the only convention in which a comparison to their published
  0.9107 / 0.6992 / 0.7911 is like-for-like.

Reporting only the first would quietly compare a 22.4 % prior against their
2.1 % prior and flatter this project badly. Both are computed and both are
reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve. Prevalence-sensitive, so it is
    only comparable between runs that share a labelling convention."""
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
                   ltv: np.ndarray | None = None) -> dict:
    """Precision, recall, F1 and the false-positive side at one threshold."""
    pred = y_score >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    out = {
        "threshold": round(float(threshold), 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged": tp + fp,
        "flagged_share": round((tp + fp) / max(y_true.size, 1), 6),
        # How many real customers are wrongly flagged for each fraudster
        # caught. This is the number an operations team actually feels.
        "false_positives_per_true_positive": round(fp / tp, 3) if tp else None,
    }
    if ltv is not None:
        wrong = pred & (y_true == 0)
        out["fp_cost_inr"] = round(float(ltv[wrong].sum()), 2)
        out["fp_cost_inr_mean"] = round(float(ltv[wrong].mean()), 2) if fp else 0.0
    return out


def threshold_for_precision(y_true: np.ndarray, y_score: np.ndarray,
                            target: float) -> float:
    """Lowest threshold reaching `target` precision, so recall is maximal at
    that precision. Returns +inf when the target is unreachable."""
    precision, _, thresholds = precision_recall_curve(y_true, y_score)
    ok = np.flatnonzero(precision[:-1] >= target)
    return float(thresholds[ok[0]]) if ok.size else float("inf")


def threshold_for_flag_budget(y_score: np.ndarray, budget: int) -> float:
    """Threshold that flags at most `budget` accounts. Review capacity, not
    statistics, is usually what sets the operating point."""
    if budget <= 0:
        return float("inf")
    if budget >= y_score.size:
        return float(y_score.min())
    return float(np.partition(y_score, -budget)[-budget])


def operating_points(y_true: np.ndarray, y_score: np.ndarray,
                     ltv: np.ndarray | None = None,
                     precisions: tuple[float, ...] = (0.80, 0.90, 0.95)) -> list[dict]:
    """A small table of operating points rather than one cherry-picked row."""
    rows = []
    for p in precisions:
        thr = threshold_for_precision(y_true, y_score, p)
        if np.isinf(thr):
            rows.append({"target_precision": p, "reachable": False})
            continue
        row = binary_metrics(y_true, y_score, thr, ltv)
        row["target_precision"] = p
        row["reachable"] = True
        rows.append(row)
    return rows


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising F1. Reported because 0.5 is arbitrary for a
    calibrated score on an imbalanced problem and understates the model."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))]) if f1.size else 0.5


def evaluate(y_true: np.ndarray, y_score: np.ndarray,
             ltv: np.ndarray | None = None, threshold: float = 0.5) -> dict:
    base = float(y_true.mean()) if y_true.size else 0.0
    ap = auprc(y_true, y_score)
    return {
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "base_rate": round(base, 6),
        "auprc": round(ap, 4),
        # AUPRC of a random ranker equals the base rate, so this is the honest
        # statement of how much the model actually adds.
        "auprc_lift_over_random": round(ap / base, 3) if base > 0 else None,
        "at_threshold_0.5": binary_metrics(y_true, y_score, threshold, ltv),
        "at_best_f1": binary_metrics(
            y_true, y_score, best_f1_threshold(y_true, y_score), ltv),
        "operating_points": operating_points(y_true, y_score, ltv),
    }


def ltv_proxy(week1_order_counts: np.ndarray, avg_order_value_inr: float) -> np.ndarray:
    """A stand-in for customer value: week-1 orders x an assumed order value.

    PPA ships no monetary amounts at all, so this is an assumption and is
    labelled as one everywhere it is used. It is only meaningful as a relative
    ranking of customers, not as a rupee figure with any external validity.
    """
    return week1_order_counts.astype(np.float64) * float(avg_order_value_inr)
