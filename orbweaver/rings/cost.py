"""Ring-level metrics, and the cost of getting a ring wrong.

Precision on its own is not a result. A ring is a recommendation to act on a
group of people at once, so being wrong is expensive in a way a single
misclassified account is not: block a 40-account "ring" that is really a
hostel and you have lost 40 real customers in one stroke.

So every number here comes with its counterpart:

- **ring precision** - of the labelled members in extracted rings, what share
  are fraud. Computed over labelled members only; unlabelled accounts are
  unknown, not innocent, and counting them either way would be a choice
  disguised as a measurement.
- **ring recall** - what share of all labelled fraudsters are inside some
  extracted ring.
- **false-positive cost** - the customer value sitting on accounts labelled
  normal that were swept into a ring, using a stated assumption about order
  value.
"""
from __future__ import annotations

import numpy as np

from orbweaver.rings.peel import Ring

FRAUD, NORMAL, UNLABELLED = 1, 0, -1


def ring_label_counts(members: np.ndarray, labels: np.ndarray) -> dict:
    lab = labels[members]
    return {
        "size": int(members.size),
        "fraud": int((lab == FRAUD).sum()),
        "normal": int((lab == NORMAL).sum()),
        "unlabelled": int((lab == UNLABELLED).sum()),
    }


def ring_precision(members: np.ndarray, labels: np.ndarray) -> float | None:
    """Share of a ring's *labelled* members that are fraud. None when the
    ring contains no labelled member at all, which is itself worth knowing."""
    lab = labels[members]
    known = int((lab == FRAUD).sum() + (lab == NORMAL).sum())
    return round(float((lab == FRAUD).sum() / known), 4) if known else None


def evaluate_rings(rings: list[Ring], labels: np.ndarray,
                   ltv: np.ndarray, *, restrict_to: np.ndarray | None = None) -> dict:
    """Aggregate ring metrics with the false-positive side attached.

    `restrict_to` limits scoring to a set of accounts (the held-out set), so
    ring quality is measured on accounts the scorer never trained on.
    """
    if not rings:
        return {"n_rings": 0, "note": "no rings extracted above the density floor"}

    allowed = None
    if restrict_to is not None:
        allowed = np.zeros(labels.size, dtype=bool)
        allowed[restrict_to] = True

    per_ring, all_members = [], []
    for r in rings:
        m = r.members
        if allowed is not None:
            m = m[allowed[m]]
        all_members.append(r.members)
        counts = ring_label_counts(r.members, labels)
        fp_cost = float(ltv[r.members[labels[r.members] == NORMAL]].sum())
        per_ring.append({
            "rank": r.rank,
            "size": counts["size"],
            "density": round(float(r.density), 4),
            "internal_weight": round(float(r.internal_weight), 2),
            "fraud": counts["fraud"], "normal": counts["normal"],
            "unlabelled": counts["unlabelled"],
            "precision": ring_precision(r.members, labels),
            "fp_cost_inr": round(fp_cost, 2),
            "labelled_share": round(
                (counts["fraud"] + counts["normal"]) / max(counts["size"], 1), 4),
        })

    members = np.unique(np.concatenate(all_members))
    lab = labels[members]
    n_fraud_caught = int((lab == FRAUD).sum())
    n_normal_caught = int((lab == NORMAL).sum())
    known = n_fraud_caught + n_normal_caught
    total_fraud = int((labels == FRAUD).sum())

    # Held-out-only figures are the strict ones, but rings surface only a few
    # hundred accounts, of which a quarter of the labelled ones are held out -
    # often too few to estimate a rate from. Both are reported: the all-labelled
    # figure for a usable sample size, the held-out figure for strictness.
    heldout_block = None
    if allowed is not None:
        in_scope = members[allowed[members]]
        lab_s = labels[in_scope]
        hf = int((lab_s == FRAUD).sum())
        hn = int((lab_s == NORMAL).sum())
        hk = hf + hn
        ho_total = int((labels[restrict_to] == FRAUD).sum())
        heldout_block = {
            "labelled_members": hk,
            "fraud_members": hf,
            "normal_members": hn,
            "ring_precision": round(hf / hk, 4) if hk else None,
            "ring_recall": round(hf / ho_total, 6) if ho_total else None,
            "total_fraud_in_scope": ho_total,
            "small_sample": hk < 30,
        }

    fp_cost_total = float(ltv[members[labels[members] == NORMAL]].sum())

    return {
        "heldout_only": heldout_block,
        "n_rings": len(rings),
        "accounts_in_rings": int(members.size),
        "ring_sizes": {
            "min": int(min(r.size for r in rings)),
            "median": int(np.median([r.size for r in rings])),
            "max": int(max(r.size for r in rings)),
        },
        "labelled_members": known,
        "fraud_members": n_fraud_caught,
        "normal_members": n_normal_caught,
        "unlabelled_members": int((lab == UNLABELLED).sum()),
        # Of the members we can check, how many are actually fraud.
        "ring_precision": round(n_fraud_caught / known, 4) if known else None,
        # Of all the fraudsters there are, how many did we surround.
        "ring_recall": round(n_fraud_caught / total_fraud, 4) if total_fraud else None,
        "total_fraud_in_scope": total_fraud,
        # The number an operations lead asks for: how many real customers do
        # we disturb for each fraudster we catch.
        "normal_flagged_per_fraud_caught": round(
            n_normal_caught / n_fraud_caught, 3) if n_fraud_caught else None,
        "fp_cost_inr": round(fp_cost_total, 2),
        "per_ring": per_ring,
    }
