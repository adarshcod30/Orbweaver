"""How many confirmed cases before this works.

The first thing a risk lead asks about a system like this is not "how good
is it" but "we have no confirmed ring labels in India yet - does it work
before we have them?" This project has answered the two extreme points
without meaning to: the zero-label answer (unpruned peeling lands at 0.0696,
below the 0.2242 base rate - dense is not fraud) and the all-labels answer
(0.7292). Nothing between them, and the space between is the entire
deployment plan for a team starting from nothing.

**Design.** Fractions of the training pool's labels - 0.5%, 1%, 2%, 5%, 10%,
20%, 50%, 100% - stratified by label so the fraud rate in every subset
matches the full pool's, three seeds each. At every point the scorer and its
isotonic calibration are refitted from scratch on that subset only, using
`fit_scorer` exactly as it runs everywhere else in this project - this module
adds no new training logic, only a different `Split` handed to the existing
one. Held-out accounts (`split.test`) never appear in any subset, at any
fraction, under any seed: they are the fixed ruler every point is measured
against.

**The 100% point is not resampled at all.** Routing it through the same
stratified-sampling code that produces the smaller fractions could, at best,
reproduce the full training pool in a different row order - and XGBoost's
own row and column subsampling is seeded, so a different order can fit a
subtly different model even over the identical set of accounts. The 100%
point reuses today's `split.train` and `split.val` arrays unchanged, which is
what makes it a real guard rather than a coincidence: if this file has
disturbed anything, that point stops matching `docs/results.md` exactly.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from eval.split import Split, make_split
from orbweaver.config import Config, load_config

FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
N_SEEDS = 3
BASE_RATE = 0.2242
ZERO_LABEL_RING_PRECISION = 0.0696
# What "more labels stop buying much" means, stated rather than eyeballed:
# less than one percentage point of held-out AUPRC gained by doubling-ish to
# the next fraction on the sweep.
DIMINISHING_RETURNS_AUPRC_INCREMENT = 0.01


def label_permutation(pool: np.ndarray, labels: np.ndarray,
                      seed: int) -> dict[int, np.ndarray]:
    """One random ordering of each label's members in `pool`, for one seed.

    A prefix of this ordering at any length is what every fraction's subset
    at this seed is built from, which is what makes the fractions nested:
    the 1% subset is the first slice of the same shuffled list the 2% subset
    is a longer slice of, not an independent draw that could disagree with
    it about which accounts to include.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for cls in (0, 1):
        members = pool[labels[pool] == cls]
        out[cls] = members[rng.permutation(members.size)]
    return out


def stratified_subset(perm: dict[int, np.ndarray], fraction: float) -> np.ndarray:
    """The prefix of each label's permutation at `fraction`, so the subset's
    fraud rate matches the full pool's and every smaller fraction's subset
    at the same seed is contained in this one."""
    parts = []
    for cls, members in perm.items():
        n = max(1, int(round(fraction * members.size))) if fraction < 1.0 else members.size
        parts.append(members[:min(n, members.size)])
    return np.sort(np.concatenate(parts))


def make_subset_split(cfg: Config, split: Split, perm: dict[int, np.ndarray],
                      fraction: float, seed: int) -> Split:
    """A Split whose train/val/train_pool are a stratified, nested fraction
    of today's training pool. `test`, `labels` and `n_users_week2` are
    always the real ones - the held-out ruler never moves."""
    if fraction >= 1.0:
        return split
    pool = stratified_subset(perm, fraction)
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool.size)
    n_val = max(1, int(round(cfg.scoring.val_fraction * pool.size)))
    val = np.sort(pool[order[:n_val]])
    train = np.sort(pool[order[n_val:]])
    if train.size == 0:
        train = val  # a fraction small enough that the split leaves nothing;
        # train on it anyway rather than crashing, and let the result speak.
    return Split(train=train, val=val, test=split.test, train_pool=pool,
                labels=split.labels, n_users_week2=split.n_users_week2)


def evaluate_point(cfg: Config, split: Split, sub: Split) -> dict:
    """Fit the scorer on `sub`, then measure it the same two ways every
    other headline number in this project is measured: held-out AUPRC, and
    ring precision after pruning at the standard operating point."""
    from eval.metrics import evaluate, ltv_proxy
    from eval.run_rings import load_edges, prune
    from orbweaver.rings.cost import evaluate_rings
    from orbweaver.rings.peel import extract_rings_batch
    from orbweaver.scoring.xgb_graph import fit_scorer

    result = fit_scorer(cfg, sub)
    scores = result.scores
    n = scores.size

    y_test = split.y(split.test)
    node = evaluate(y_test, scores[split.test])

    proc = cfg.abs_path(cfg.paths.processed)
    f = pq.read_table(proc / "features_week2_early.parquet", columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    edges = prune(load_edges("late", cfg, n), scores, cfg.rings.prune_tau_headline)
    rings = extract_rings_batch(edges, scores, lambda_=cfg.rings.lambda_headline,
                                k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
    block = evaluate_rings(rings, split.labels, ltv, restrict_to=split.test) if rings else {}

    return {
        "labelled_accounts_used": int(sub.train.size + sub.val.size),
        "train_accounts": int(sub.train.size), "val_accounts": int(sub.val.size),
        "auprc": node["auprc"], "auprc_lift_over_random": node["auprc_lift_over_random"],
        "n_rings": len(rings),
        "ring_precision": block.get("ring_precision"),
        "normal_flagged_per_fraud_caught": block.get("normal_flagged_per_fraud_caught"),
        "fraud_members": block.get("fraud_members"),
        "accounts_in_rings": block.get("accounts_in_rings"),
    }


def aggregate(fraction: float, runs: list[dict]) -> dict:
    def stat(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if not vals:
            return {"mean": None, "min": None, "max": None, "n": 0}
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(min(vals)), 4),
               "max": round(float(max(vals)), 4), "n": len(vals)}

    return {
        "fraction": fraction,
        "labelled_accounts_used": stat("labelled_accounts_used")["mean"],
        "auprc": stat("auprc"),
        "auprc_lift_over_random": stat("auprc_lift_over_random"),
        "ring_precision": stat("ring_precision"),
        "normal_flagged_per_fraud_caught": stat("normal_flagged_per_fraud_caught"),
        "fraud_members": stat("fraud_members"),
        "runs": runs,
    }


def find_knee(points: list[dict]) -> dict:
    """Smallest label count beating the base rate, and where AUPRC gains
    fall below the stated increment - both in absolute counts, because a
    count is what a team can plan a labelling effort around, a percentage of
    an unknown-sized future pool is not.

    "Diminishing returns" means a genuine plateau: every remaining step on
    the sweep stays under the increment, not just the next one. The first
    version of this function broke on the first small step it saw, which on
    the real sweep was the 0.5%-to-1% step - the two smallest, noisiest
    points, three seeds of ~1,150 accounts each - even though the very next
    step gained six times the threshold. A single noisy dip is not a plateau;
    requiring every later step to also stay small is.
    """
    beats_base = [p for p in points if (p["ring_precision"]["mean"] or 0) > BASE_RATE]
    first_beat = beats_base[0] if beats_base else None

    gains = []
    for prev, cur in zip(points, points[1:]):
        pa, ca = prev["auprc"]["mean"], cur["auprc"]["mean"]
        gains.append(None if pa is None or ca is None else round(ca - pa, 4))

    diminishing, gain_at_knee = None, None
    for i, g in enumerate(gains):
        rest = gains[i:]
        if g is not None and all(r is not None and r < DIMINISHING_RETURNS_AUPRC_INCREMENT for r in rest):
            diminishing, gain_at_knee = points[i + 1], g
            break

    return {
        "beats_base_rate_at": {
            "fraction": first_beat["fraction"], "labelled_accounts": first_beat["labelled_accounts_used"],
            "ring_precision_mean": first_beat["ring_precision"]["mean"],
        } if first_beat else None,
        "diminishing_returns_after": {
            "fraction": diminishing["fraction"], "labelled_accounts": diminishing["labelled_accounts_used"],
            "auprc_gain_from_previous_point": gain_at_knee,
            "increment_threshold": DIMINISHING_RETURNS_AUPRC_INCREMENT,
        } if diminishing else None,
    }


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    split = make_split(cfg)

    # One permutation per seed, computed once and sliced at every fraction,
    # so the fractions are nested within a seed rather than eight independent
    # draws that could each disagree about which accounts to include.
    seed_perms = {i: label_permutation(split.train_pool, split.labels,
                                       cfg.seed * 10_000 + i)
                 for i in range(N_SEEDS)}

    by_fraction: dict[float, list[dict]] = {f: [] for f in FRACTIONS}
    for frac in FRACTIONS:
        print(f"  {frac:>6.1%} of the training pool ...", flush=True)
        if frac >= 1.0:
            # No seed varies anything at 100% - there is nothing left to
            # subsample - so fitting it three times would spend real minutes
            # confirming floating point is deterministic. One fit, reported
            # under all three seed slots so the point keeps the same shape
            # as every other row in the table.
            r = evaluate_point(cfg, split, split)
            print(f"      (single fit; 100% leaves nothing to vary across seeds) "
                  f"{r['labelled_accounts_used']:>7,} accounts  "
                  f"AUPRC {r['auprc']}  ring precision {r['ring_precision']}")
            by_fraction[frac] = [r, r, r]
            continue
        for i in range(N_SEEDS):
            seed = cfg.seed * 10_000 + int(round(frac * 10_000)) + i
            sub = make_subset_split(cfg, split, seed_perms[i], frac, seed)
            r = evaluate_point(cfg, split, sub)
            by_fraction[frac].append(r)
            print(f"      seed {i}: {r['labelled_accounts_used']:>7,} accounts  "
                  f"AUPRC {r['auprc']}  ring precision {r['ring_precision']}")

    points = [aggregate(frac, by_fraction[frac]) for frac in FRACTIONS]

    knee = find_knee(points)
    return {
        "method": ("The scorer and its isotonic calibration are refitted from scratch on a "
                  "stratified fraction of today's training pool, using fit_scorer unchanged - "
                  "this module supplies a different Split, not a different training routine. "
                  "Held-out accounts never appear in any subset. The 100% point reuses "
                  "split.train/split.val unmodified rather than resampling 'everything', so it "
                  "is a guard against this file having disturbed anything, not a coincidence."),
        "fractions": list(FRACTIONS), "seeds_per_fraction": N_SEEDS,
        "reference_lines": {"base_rate": BASE_RATE,
                            "zero_label_unpruned_ring_precision": ZERO_LABEL_RING_PRECISION},
        "points": points,
        "knee": knee,
        "diminishing_returns_auprc_increment": DIMINISHING_RETURNS_AUPRC_INCREMENT,
        "ieee_cis": None,
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "label_budget.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    k = out["knee"]
    print()
    if k["beats_base_rate_at"]:
        b = k["beats_base_rate_at"]
        print(f"beats the base rate at {b['labelled_accounts']:,} labelled accounts "
              f"({b['fraction']:.1%}), ring precision {b['ring_precision_mean']}")
    else:
        print("never beats the base rate across the sweep")
    if k["diminishing_returns_after"]:
        d = k["diminishing_returns_after"]
        print(f"diminishing returns after {d['labelled_accounts']:,} accounts "
              f"({d['fraction']:.1%}): +{d['auprc_gain_from_previous_point']} AUPRC, "
              f"below the {d['increment_threshold']} threshold")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
