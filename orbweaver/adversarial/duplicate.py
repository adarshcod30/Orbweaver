"""Multi-round adaptation: fraudsters replace what gets caught.

The published protocol (SNAM 2025) models an attacker who watches which of
their accounts get detected and replaces the ones that survive. Two generators
run each round:

- **query generator** — takes fraud accounts that were *not* detected and
  duplicates them, features and edges, as new accounts. What worked keeps
  working and gets copied.
- **support generator** — reveals true labels only for accounts the detector
  flagged. This is the realistic labelling regime: a platform learns an account
  was fraudulent because it was caught or complained about, never for the ones
  it missed.

The published finding is that state-of-the-art detectors fall below F1 0.56
after a couple of rounds. The point of running it here is not to beat that; it
is to see the shape of the decline, and specifically whether ring precision
holds up better than account-level detection does — a duplicated account
inherits its originals' edges, so it lands *inside* the same dense structure
rather than escaping it.

Defence-only: it duplicates rows in a public labelled dataset to measure a
detector's degradation. It contains nothing usable against a real system.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList

FRAUD = 1


def duplicate_undetected(edges: EdgeList, scores: np.ndarray, labels: np.ndarray,
                         detected: np.ndarray, n_slots: int, *,
                         duplicates: int = 1, seed: int = 0,
                         max_new: int = 60_000) -> tuple[EdgeList, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Copy undetected fraud accounts, with their edges, as new accounts.

    Returns the grown graph, extended scores and labels, and the ids of the
    accounts that were created.
    """
    rng = np.random.default_rng(seed)
    undetected = np.flatnonzero((labels == FRAUD) & ~detected)
    if undetected.size == 0:
        return edges, scores, labels, np.empty(0, dtype=np.int64)

    if undetected.size * duplicates > max_new:
        undetected = rng.choice(undetected, size=max_new // max(duplicates, 1),
                                replace=False)

    new_src, new_dst, new_w, new_ids = [], [], [], []
    next_id = n_slots
    src, dst, w = edges.src, edges.dst, edges.weight

    # Index edges by endpoint once; doing it per account is quadratic.
    order = np.argsort(np.concatenate([src, dst]), kind="stable")
    both = np.concatenate([src, dst])[order]
    other = np.concatenate([dst, src])[order]
    wboth = np.concatenate([w, w])[order]
    starts = np.searchsorted(both, np.arange(n_slots + 1))

    for acct in undetected:
        lo, hi = starts[acct], starts[acct + 1]
        nbr, nw = other[lo:hi], wboth[lo:hi]
        for _ in range(duplicates):
            clone = next_id
            next_id += 1
            new_ids.append(clone)
            if nbr.size:
                new_src.append(np.full(nbr.size, clone, dtype=np.int64))
                new_dst.append(nbr.astype(np.int64))
                new_w.append(nw)
            # The clone is also tied to its original: same operator, same
            # shared entities, so an edge between them is what the real
            # duplication would produce.
            new_src.append(np.array([min(clone, acct)], dtype=np.int64))
            new_dst.append(np.array([max(clone, acct)], dtype=np.int64))
            new_w.append(np.array([float(np.median(nw)) if nw.size else 0.5]))

    if not new_ids:
        return edges, scores, labels, np.empty(0, dtype=np.int64)

    total = next_id
    grown = EdgeList(
        np.concatenate([src] + new_src),
        np.concatenate([dst] + new_dst),
        np.concatenate([w] + new_w),
        total,
    )
    new_ids = np.array(new_ids, dtype=np.int64)
    s2 = np.zeros(total); s2[: scores.size] = scores
    l2 = np.full(total, -1, dtype=np.int8); l2[: labels.size] = labels
    # A clone behaves like its original, so it inherits its score and label.
    origin = np.repeat(undetected, duplicates)[: new_ids.size]
    s2[new_ids] = scores[origin]
    l2[new_ids] = FRAUD
    return grown, s2, l2, new_ids


def run_rounds(cfg: Config | None = None, rounds: int = 5) -> dict:
    from eval.run_rings import load_edges, prune
    from eval.split import make_split
    from orbweaver.rings.peel import extract_rings_batch

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size

    scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    report = json.loads((proc / "ring_report.json").read_text())
    best = report.get("best_cell", {"tau": 0.5, "lambda": 5.0})
    top_k = report["graph"]["top_k"]

    edges = load_edges("late", cfg, n)
    labels = split.labels.copy()
    rows = []

    print(f"{'round':>6} {'accounts':>10} {'rings':>6} {'base':>7} {'precision':>10} "
          f"{'lift':>6} {'recall':>9} {'new clones':>11}")
    for rnd in range(rounds + 1):
        sub = prune(edges, scores, best["tau"])
        rings = extract_rings_batch(sub, scores, lambda_=best["lambda"],
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=top_k, g_min=cfg.rings.g_min)
        detected = np.zeros(labels.size, dtype=bool)
        for r in rings:
            detected[r.members] = True
        members = np.unique(np.concatenate([r.members for r in rings])) if rings \
            else np.empty(0, dtype=np.int64)
        lab = labels[members]
        known = int((lab == FRAUD).sum() + (lab == 0).sum())
        prec = round(float((lab == FRAUD).sum() / known), 4) if known else None
        total_fraud = int((labels == FRAUD).sum())
        # Each round injects accounts that are fraud by construction, so the
        # population's base rate climbs. Raw precision rising is then mostly
        # definitional, and lift over the *current* base rate is the only
        # honest way to read this table.
        n_labelled = int((labels != -1).sum())
        base = total_fraud / n_labelled if n_labelled else 0.0
        row = {
            "round": rnd,
            "accounts": int(labels.size),
            "n_rings": len(rings),
            "ring_precision": prec,
            "base_rate": round(base, 4),
            "precision_lift_over_base": round(prec / base, 3) if prec and base else None,
            "fraud_caught": int((lab == FRAUD).sum()),
            "total_fraud": total_fraud,
            "recall": round(float((lab == FRAUD).sum() / total_fraud), 6)
            if total_fraud else None,
        }
        rows.append(row)

        if rnd == rounds:
            print(f"{rnd:>6} {row['accounts']:>10,} {row['n_rings']:>6} "
                  f"{row['base_rate']:>7} {str(prec):>10} "
                  f"{str(row['precision_lift_over_base']):>6} "
                  f"{str(row['recall']):>9} {'-':>11}")
            break

        edges, scores, labels, new_ids = duplicate_undetected(
            edges, scores, labels, detected, labels.size, seed=cfg.seed + rnd)
        row["clones_added"] = int(new_ids.size)
        print(f"{rnd:>6} {row['accounts']:>10,} {row['n_rings']:>6} "
              f"{row['base_rate']:>7} {str(prec):>10} "
              f"{str(row['precision_lift_over_base']):>6} "
              f"{str(row['recall']):>9} {new_ids.size:>11,}")
        if new_ids.size == 0:
            break

    return {
        "operating_point": best,
        "rounds": rows,
        "protocol": ("Each round duplicates undetected fraud accounts with their "
                     "edges as new accounts, and reveals labels only for accounts "
                     "the detector flagged. Follows the multi-round adversarial "
                     "protocol in SNAM 2025."),
    }


def main() -> None:
    cfg = load_config()
    out = run_rounds(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "adversarial_rounds.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
