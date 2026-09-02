"""What the graph knows about an account, handed back to the per-account score.

The argument this project makes is that a ring is invisible to a system that
scores one transaction at a time. The fair follow-up is: fine, then can the
ring view give something back to the per-account view? If it can, a
per-transaction system does not have to be replaced to benefit from any of
this — it can consume a few extra columns.

Four context features per account, all derived from rings found in an
**earlier** window:

- whether the account was itself in a previous-window ring;
- that ring's confidence, where a ring model exists;
- how many of its current neighbours were in previous-window rings;
- what share of its rare shared entities are also held by previous-window ring
  members.

**The horizon rule.** Context must come from days strictly before the window
whose features it joins, for every account, or it is not used at all. Early
rings joined to late features satisfy this by construction — every day in the
early window precedes every day in the late one — and `assert_horizon` checks
it against the window manifests rather than trusting the naming convention.

The primary result is deliberately parameter-free. Re-ranking held-out accounts
by score and context in two fixed ways needs nothing fitted, so there is no
opportunity to tune on the thing being measured. A one-parameter blend is
reported second, fitted only on the validation split.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

CONTEXT_FEATURES = [
    "in_previous_ring",
    "previous_ring_confidence",
    "neighbours_in_previous_rings",
    "rare_entity_overlap_with_previous_rings",
]
# An entity is "rare" at or below this many accounts platform-wide, the same
# bar the evidence extractor uses when deciding what counts as a lead.
RARE_ENTITY_MAX = 10


def assert_horizon(cfg: Config, context_tag: str, feature_tag: str) -> dict:
    """Every day of the context window must precede every day of the feature
    window. Checked against the manifests, not against the tag names."""
    proc = cfg.abs_path(cfg.paths.processed)
    ctx = json.loads((proc / f"edges_week2_{context_tag}_manifest.json").read_text())
    fea = json.loads((proc / f"edges_week2_{feature_tag}_manifest.json").read_text())
    c_days, f_days = ctx["days"], fea["days"]
    if not (c_days and f_days):
        raise ValueError("a window manifest is missing its day range")
    if c_days[1] >= f_days[0]:
        raise ValueError(
            f"context window {c_days} is not strictly before feature window "
            f"{f_days}; that would let the future inform the past")
    return {"context_days": c_days, "feature_days": f_days,
            "gap_days": f_days[0] - c_days[1]}


def previous_ring_membership(cfg: Config, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Which accounts were in an earlier-window ring, and how confident it was.

    Prefers the ring model's confidences when they exist; falls back to the
    density-ranked rings otherwise, so this works whether or not the ring
    scorer landed.
    """
    proc = cfg.abs_path(cfg.paths.processed)
    member = np.zeros(n, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)

    rs = proc / "ring_scorer_early_rings.json"
    if rs.exists():
        payload = json.loads(rs.read_text())
        for r in payload["rings"]:
            m = np.asarray(r["members"], dtype=np.int64)
            member[m] = 1.0
            conf[m] = np.maximum(conf[m], float(r.get("confidence", 0.0)))
        return member, conf

    # Fall back to peeling the early window at the headline point. No ring
    # model, so confidence is the density rank scaled to [0, 1] - a ranking,
    # honestly labelled as one rather than dressed up as a probability.
    from orbweaver.rings.peel import EdgeList, extract_rings_batch

    e = pq.read_table(proc / "edges_week2_early.parquet",
                      columns=["src", "dst", "weight"])
    scores = np.zeros(n)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()
    edges = EdgeList(e["src"].to_numpy().astype(np.int64),
                     e["dst"].to_numpy().astype(np.int64),
                     e["weight"].to_numpy().astype(np.float64), n)
    keep = scores > cfg.rings.prune_tau_headline
    m = keep[edges.src] & keep[edges.dst]
    sub = EdgeList(edges.src[m], edges.dst[m], edges.weight[m], n)
    rings = extract_rings_batch(sub, scores, lambda_=cfg.rings.lambda_headline,
                                k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
    for i, r in enumerate(rings):
        member[r.members] = 1.0
        conf[r.members] = np.maximum(conf[r.members],
                                     1.0 - i / max(len(rings), 1))
    return member, conf


def build_context(cfg: Config | None = None, n: int | None = None) -> dict:
    """The four context columns, plus the horizon check that licenses them."""
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    n = n or int(pq.read_table(proc / "nodes.parquet").num_rows)

    horizon = assert_horizon(cfg, "early", "late")
    member, conf = previous_ring_membership(cfg, n)

    # Neighbours in the CURRENT window who were in a previous-window ring.
    e = pq.read_table(proc / "edges_week2_late.parquet",
                      columns=["src", "dst", "min_entity_size"])
    src = e["src"].to_numpy().astype(np.int64)
    dst = e["dst"].to_numpy().astype(np.int64)
    esz = e["min_entity_size"].to_numpy().astype(np.float64)

    both = np.concatenate([src, dst])
    other = np.concatenate([dst, src])
    nbr_ring = np.bincount(both, weights=member[other], minlength=n)

    # Of an account's rare edges, how many reach a previous-window ring member.
    rare = np.concatenate([esz, esz]) <= RARE_ENTITY_MAX
    rare_deg = np.bincount(both[rare], minlength=n).astype(np.float64)
    rare_to_ring = np.bincount(both[rare], weights=member[other[rare]], minlength=n)
    rare_overlap = np.where(rare_deg > 0, rare_to_ring / np.maximum(rare_deg, 1), 0.0)

    return {
        "horizon": horizon,
        "features": {
            "in_previous_ring": member,
            "previous_ring_confidence": conf,
            "neighbours_in_previous_rings": nbr_ring.astype(np.float32),
            "rare_entity_overlap_with_previous_rings": rare_overlap.astype(np.float32),
        },
        "accounts_in_previous_rings": int(member.sum()),
    }


def combine(score: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
    """Two fixed ways to fold context into a score, neither of them fitted.

    Nothing here has a parameter, so there is no way to tune on the thing being
    measured. A fitted blend is reported separately and only ever fitted on the
    validation split.
    """
    ctx = context.astype(np.float64)
    if ctx.max() > 0:
        ctx = ctx / ctx.max()
    return {
        # Multiplicative: context can at most double a score, and can never
        # rescue an account the model thinks is clean.
        "score_times_context": score * (1.0 + ctx),
        # Lexicographic: score decides, context breaks ties. The epsilon is
        # smaller than any gap the calibrated score produces.
        "score_then_context": score + 1e-6 * ctx,
    }
