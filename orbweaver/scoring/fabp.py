"""Spread the few confirmed labels there are, by the graph's own structure.

Gradient boosting and GraphSAGE both need enough labelled rows to fit a
model before they say anything. Fast Belief Propagation (Koutra, Ke, Kang,
Chau, Pao, Faloutsos - "Unifying Guilt-by-Association Approaches: Theorems
and Fast Algorithms", ECML-PKDD 2011) needs no fitting at all: it linearises
belief propagation into one sparse linear system,

    [I + a.D - c'.A] b_h = phi_h

solves it by a power iteration that is exactly the series
(I-W)^-1 = I + W + W^2 + ... applied to phi_h a term at a time, and reads
the belief off directly. `D` is the account graph's weighted degree, `A` its
weighted adjacency (the same edges_week2_late.parquet everything else in
this project peels and prunes), and `phi_h` the prior beliefs - a small
positive value for a training-pool fraud account, a small negative value for
a training-pool normal one, zero for everyone else, held-out accounts always
included. `a` and `c'` come from one homophily factor `h_h` (Theorem 1); the
paper proves two independent sufficient conditions under which the power
series is guaranteed to converge (Lemmas 5 and 6), and Section 5's own
algorithm picks `h_h` as the larger of the two bounds. Here `h_h` is instead
picked from the graph's own measured fraud assortativity - "set from the
graph's measured assortativity" - and only *capped* at whichever bound is
looser, with the bound re-checked and enforced (not assumed) immediately
before every solve.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from orbweaver.config import Config, load_config

FRAUD, NORMAL, UNLABELLED = 1, 0, -1

# "About-half" prior magnitude. Small by design, matching the paper's own
# choice of {h_h, priors} on the order of 1e-3 to 1e-2 (their DBLP
# experiment used {0.002, +-0.001}) so the linearisation - a first-order
# expansion around the neutral point h=phi=0.5 - stays in the regime it was
# derived for. The exact value never changes a ranking: the system is linear
# in phi_h, so every metric read off the beliefs here (AUPRC, precision@k)
# is invariant to rescaling it.
PRIOR_MAGNITUDE = 0.01
TOLERANCE = 1e-6
MAX_ITERATIONS = 500
# Both convergence lemmas are strict inequalities (h_h < bound, not <=), so
# capping h_h at exactly the bound would leave it right on the boundary the
# theorem excludes. A margin of one part in a million costs nothing in
# practice - h_h is already tiny - and keeps the cap strictly inside the
# region the proof actually covers.
CONVERGENCE_SAFETY_MARGIN = 1.0 - 1e-6


# --------------------------------------------------------------- Theorem 1 --

def fabp_constants(h_h: float) -> tuple[float, float]:
    """a = 4h_h^2 / (1 - 4h_h^2), c' = 2h_h / (1 - 4h_h^2)."""
    denom = 1.0 - 4.0 * h_h * h_h
    if denom <= 0:
        raise ValueError(f"h_h={h_h} is outside the linearisation's domain (needs 4h_h^2 < 1)")
    return 4.0 * h_h * h_h / denom, 2.0 * h_h / denom


# ------------------------------------------------------------- convergence --

def convergence_bounds(d_diag: np.ndarray) -> dict:
    """The paper's two independent sufficient conditions on h_h (Lemmas 5
    and 6), each a different matrix-norm bound on the spectral radius of
    c'A - aD. Either alone guarantees the power series converges.

    Lemma 5 (1-norm): h_h < 1 / (2 + 2 * max_j d_jj).
    Lemma 6 (Frobenius norm): h_h < sqrt((-c1 + sqrt(c1^2 + 4c2)) / (8c2)),
    c1 = 2 + sum(d_ii), c2 = sum(d_ii^2) - 1.

    The paper notes the Frobenius bound is preferable when degrees have
    considerable spread (the 1-norm bound is dragged down by a single very
    high degree node), which is exactly this graph's shape - a handful of
    accounts touch many entities, most touch few - so both are computed and
    the algorithm is free to use whichever is looser.
    """
    d = d_diag.astype(np.float64)
    max_d = float(d.max()) if d.size else 0.0
    one_norm = 1.0 / (2.0 + 2.0 * max_d)

    c1 = 2.0 + float(d.sum())
    c2 = float((d * d).sum()) - 1.0
    if c2 > 0:
        frobenius = float(np.sqrt((-c1 + np.sqrt(c1 * c1 + 4 * c2)) / (8 * c2)))
    else:
        # Only possible on a graph small or sparse enough that sum(d_ii^2) <=
        # 1 - the Frobenius bound's own algebra divides by c2 and is not
        # meaningful there. The 1-norm bound stands on its own in that case.
        frobenius = one_norm

    return {"one_norm": one_norm, "frobenius": frobenius, "max": max(one_norm, frobenius)}


def choose_h_h(d_diag: np.ndarray, lift: float) -> dict:
    """h_h set from the graph's measured assortativity, capped at whichever
    convergence bound is looser.

    A fraud-fraud lift of 1 means no assortativity at all and maps to
    h_h=0, the neutral point; an unboundedly large lift maps toward 0.5, the
    paper's own upper limit on homophily. This is a bounded, monotone
    translation between two scales, not a fitted parameter - there is
    nothing here for training data to overfit, only unit conversion from a
    lift ratio (no natural upper bound) onto a homophily factor (hard-bounded
    at 0.5 by the "about-half" definition itself).
    """
    desired = 0.5 * (1.0 - 1.0 / lift) if lift > 1.0 else 0.0
    bounds = convergence_bounds(d_diag)
    capped = desired > bounds["max"]
    h_h = bounds["max"] * CONVERGENCE_SAFETY_MARGIN if capped else desired
    return {"desired_from_assortativity": round(desired, 6), "convergence_bounds": bounds,
            "h_h": h_h, "capped_by_convergence": capped}


def assert_convergent(h_h: float, d_diag: np.ndarray) -> None:
    """The paper's stated convergence condition, checked directly rather
    than assumed: h_h must clear at least one of Lemma 5 or Lemma 6. Every
    call to `solve_fabp` passes through this gate first."""
    bounds = convergence_bounds(d_diag)
    if not (h_h < bounds["one_norm"] or h_h < bounds["frobenius"]):
        raise ValueError(
            f"h_h={h_h} satisfies neither convergence bound "
            f"(1-norm {bounds['one_norm']:.6g}, Frobenius {bounds['frobenius']:.6g}) - "
            "the power iteration is not guaranteed to converge")


# -------------------------------------------------------------- the solver --

def solve_fabp(A: sp.spmatrix, d_diag: np.ndarray, phi_h: np.ndarray, h_h: float, *,
               tol: float = TOLERANCE, max_iter: int = MAX_ITERATIONS) -> dict:
    """Solve [I + aD - c'A] b_h = phi_h by the paper's power method (Eq. 6):
    the Neumann series (I-W)^-1 = I + W + W^2 + ..., W = c'A - aD, applied to
    phi_h one term at a time so every step is a sparse matrix-vector product
    rather than ever forming W or inverting anything.
    """
    assert_convergent(h_h, d_diag)
    a, c = fabp_constants(h_h)

    b = phi_h.astype(np.float64).copy()
    term = phi_h.astype(np.float64).copy()
    for it in range(1, max_iter + 1):
        term = c * (A @ term) - a * (d_diag * term)
        b_next = b + term
        delta = float(np.max(np.abs(b_next - b))) if b_next.size else 0.0
        b = b_next
        if delta < tol:
            return {"beliefs": b, "iterations": it, "converged": True,
                    "final_delta": delta, "a": a, "c_prime": c}
    raise RuntimeError(f"FaBP power iteration did not reach tol={tol} within "
                       f"{max_iter} iterations (h_h={h_h} should have guaranteed this)")


def fabp_beliefs_for_prior(A: sp.spmatrix, d_diag: np.ndarray, h_h: float,
                           phi: np.ndarray) -> dict:
    """The solve plus timing, with no I/O - reusable across many priors over
    the same fixed graph (e.g. the label-budget sweep, which reuses one
    account graph for all twenty-four subsets)."""
    t0 = time.time()
    solved = solve_fabp(A, d_diag, phi, h_h)
    return {"scores": solved["beliefs"] + 0.5, "iterations": solved["iterations"],
            "solve_seconds": round(time.time() - t0, 3),
            "a": solved["a"], "c_prime": solved["c_prime"]}


# ----------------------------------------------------------- assortativity --

def measured_assortativity(labels: np.ndarray, src: np.ndarray, dst: np.ndarray,
                           visible: np.ndarray) -> dict:
    """Fraud-fraud edges against what independence would predict - the same
    statistic FAILURES.md's first entry used to show fraud is only mildly
    assortative (2.4x, measured early in the project on a different graph
    state before the weighted edges existed), recomputed here on the graph
    FaBP actually propagates over. Restricted to accounts visible on both
    ends, exactly the discipline `relation_weights.py` already enforces for
    its own lift estimate, so this cannot see a held-out label either."""
    both = visible[src] & visible[dst]
    n = int(both.sum())
    if n == 0:
        return {"edges_visible": 0, "fraud_fraud_rate": None,
                "expected_if_random": None, "lift": 1.0}
    a, b = labels[src[both]], labels[dst[both]]
    ff = float(((a == FRAUD) & (b == FRAUD)).sum()) / n
    p = (float((a == FRAUD).sum()) + float((b == FRAUD).sum())) / (2 * n)
    expected = p * p
    lift = ff / expected if expected > 0 else 1.0
    return {"edges_visible": n, "fraud_fraud_rate": round(ff, 6),
            "expected_if_random": round(expected, 6), "lift": round(lift, 4)}


def build_prior(labels: np.ndarray, visible: np.ndarray,
                magnitude: float = PRIOR_MAGNITUDE) -> np.ndarray:
    """phi_h: +magnitude for a visible fraud account, -magnitude for a
    visible normal one, 0 for every unlabelled or held-out account."""
    phi = np.zeros(labels.size, dtype=np.float64)
    phi[visible & (labels == FRAUD)] = magnitude
    phi[visible & (labels == NORMAL)] = -magnitude
    return phi


def visible_mask(split) -> np.ndarray:
    n = split.labels.size
    visible = np.zeros(n, dtype=bool)
    visible[split.train] = True
    visible[split.val] = True
    if visible[split.test].any():
        raise RuntimeError("held-out accounts are visible to the FaBP prior")
    return visible


# ------------------------------------------------------------- the account graph --

def load_account_graph(cfg: Config, n: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """The symmetric weighted account adjacency FaBP propagates over - the
    same edges_week2_late.parquet everything else in this project peels and
    prunes, stored once per pair (src < dst) and symmetrised here."""
    proc = cfg.abs_path(cfg.paths.processed)
    e = pq.read_table(proc / "edges_week2_late.parquet", columns=["src", "dst", "weight"])
    src = e["src"].to_numpy().astype(np.int64)
    dst = e["dst"].to_numpy().astype(np.int64)
    w = e["weight"].to_numpy().astype(np.float64)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([w, w])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    d_diag = np.asarray(A.sum(axis=1)).ravel()
    return A, d_diag


def node_beliefs(cfg: Config, split, *, A: sp.csr_matrix | None = None,
                 d_diag: np.ndarray | None = None, h_h: float | None = None) -> dict:
    """FaBP over the full account graph, priors from `split`'s visible
    (train + val) labels only. `A`/`d_diag` can be supplied by a caller
    solving many priors over the same fixed graph; `h_h` can likewise be
    supplied to reuse one already-chosen value rather than re-deriving it
    from a possibly small, noisy subset's own visible edges - the
    label-budget sweep does both, so the graph and the homophily factor stay
    fixed and only the prior varies with the label fraction, isolating the
    experiment to what having fewer labels does to the belief, not to what a
    different h_h would have done anyway.
    """
    n = split.labels.size
    if A is None or d_diag is None:
        A, d_diag = load_account_graph(cfg, n)
    visible = visible_mask(split)

    choice = None
    assortativity = None
    if h_h is None:
        proc = cfg.abs_path(cfg.paths.processed)
        e = pq.read_table(proc / "edges_week2_late.parquet", columns=["src", "dst"])
        assortativity = measured_assortativity(split.labels, e["src"].to_numpy(),
                                               e["dst"].to_numpy(), visible)
        choice = choose_h_h(d_diag, assortativity["lift"])
        h_h = choice["h_h"]

    phi = build_prior(split.labels, visible)
    solved = fabp_beliefs_for_prior(A, d_diag, h_h, phi)
    return {**solved, "h_h": h_h, "h_h_choice": choice, "assortativity": assortativity}


# --------------------------------------------------- label-budget comparison --

def label_curve(cfg: Config) -> dict:
    """FaBP and GraphSAGE, retrained/resolved at every point of item 3's
    label-budget sweep; XGBoost's numbers are read from that sweep's own
    artefact rather than recomputed, since they are already the exact,
    already-verified numbers this comparison needs to sit beside.

    The account graph is loaded once and reused across all twenty-four
    points (eight fractions, three seeds - a hundredth of a second's worth
    of sparse matvecs each), and `h_h` is fixed once, from the real training
    pool's own measured assortativity, for the same reason `node_beliefs`
    documents: only the prior should vary with the label fraction.
    """
    from eval.label_budget import FRACTIONS, N_SEEDS, label_permutation, make_subset_split
    from eval.metrics import evaluate
    from eval.split import make_split
    from orbweaver.scoring.sage import train_sage

    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size

    xgb_sweep = json.loads((proc / "label_budget.json").read_text())
    xgb_by_fraction = {p["fraction"]: p for p in xgb_sweep["points"]}
    sage_full = json.loads((proc / "sage_report.json").read_text())
    sage_full_auprc = sage_full["results"]["test_heldout__labelled_only"]["auprc"]

    A, d_diag = load_account_graph(cfg, n)
    e = pq.read_table(proc / "edges_week2_late.parquet", columns=["src", "dst"])
    src_all, dst_all = e["src"].to_numpy(), e["dst"].to_numpy()

    visible_full = visible_mask(split)
    assortativity = measured_assortativity(split.labels, src_all, dst_all, visible_full)
    choice = choose_h_h(d_diag, assortativity["lift"])
    h_h = choice["h_h"]
    print(f"  h_h = {h_h:.6g} (desired {choice['desired_from_assortativity']} from "
          f"{assortativity['lift']}x lift, capped by convergence: "
          f"{choice['capped_by_convergence']})", flush=True)

    seed_perms = {i: label_permutation(split.train_pool, split.labels, cfg.seed * 10_000 + i)
                 for i in range(N_SEEDS)}

    points = []
    for frac in FRACTIONS:
        print(f"  {frac:>6.1%} of the training pool ...", flush=True)
        if frac >= 1.0:
            phi = build_prior(split.labels, visible_full)
            fb = fabp_beliefs_for_prior(A, d_diag, h_h, phi)
            fabp_auprc = evaluate(split.y(split.test), fb["scores"][split.test])["auprc"]
            fabp_runs = [fabp_auprc] * N_SEEDS
            sage_runs = [sage_full_auprc] * N_SEEDS
            print(f"      (single fit; 100% leaves nothing to vary across seeds) "
                 f"FaBP AUPRC {fabp_auprc}  GraphSAGE AUPRC {sage_full_auprc} (reused)")
        else:
            fabp_runs, sage_runs = [], []
            for i in range(N_SEEDS):
                seed = cfg.seed * 10_000 + int(round(frac * 10_000)) + i
                sub = make_subset_split(cfg, split, seed_perms[i], frac, seed)
                visible = np.zeros(n, dtype=bool)
                visible[sub.train] = True
                visible[sub.val] = True
                phi = build_prior(sub.labels, visible)
                fb = fabp_beliefs_for_prior(A, d_diag, h_h, phi)
                fabp_auprc = evaluate(split.y(split.test), fb["scores"][split.test])["auprc"]
                fabp_runs.append(fabp_auprc)

                sres = train_sage(cfg, split=sub)
                sage_auprc = sres["results"]["test_heldout__labelled_only"]["auprc"]
                sage_runs.append(sage_auprc)
                print(f"      seed {i}: FaBP AUPRC {fabp_auprc:.4f}  "
                     f"GraphSAGE AUPRC {sage_auprc:.4f}", flush=True)

        xgb_p = xgb_by_fraction[frac]

        def stat(vals):
            return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
                   "max": round(float(np.max(vals)), 4)}

        points.append({
            "fraction": frac, "labelled_accounts_used": xgb_p["labelled_accounts_used"],
            "xgboost_auprc": xgb_p["auprc"],
            "fabp_auprc": stat(fabp_runs), "graphsage_auprc": stat(sage_runs),
        })

    return {"h_h": h_h, "h_h_choice": choice, "assortativity": assortativity, "points": points}


# ------------------------------------------------------------------- ring test --

def ring_test(cfg: Config, split) -> dict:
    """Prune on FaBP beliefs instead of the calibrated XGBoost score, peel
    with the same objective, and compare against the XGBoost-pruned result
    at "the same operating point" - read here as the same *share of accounts
    kept* after pruning, since FaBP's belief is not a calibrated probability
    and reusing XGBoost's own tau value verbatim would compare two different
    thresholds' worth of accounts rather than two scorers at one threshold.
    """
    from eval.metrics import ltv_proxy
    from eval.run_rings import load_edges, prune
    from orbweaver.rings.cost import evaluate_rings
    from orbweaver.rings.peel import extract_rings_batch

    n = split.labels.size
    proc = cfg.abs_path(cfg.paths.processed)

    xgb_scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    xgb_scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    beliefs = node_beliefs(cfg, split)
    fabp_scores = beliefs["scores"]

    share_kept = float((xgb_scores > cfg.rings.prune_tau_headline).mean())
    tau_fabp = float(np.quantile(fabp_scores, 1.0 - share_kept))

    f = pq.read_table(proc / "features_week2_early.parquet", columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    def peel_and_eval(scores, tau):
        edges = prune(load_edges("late", cfg, n), scores, tau)
        rings = extract_rings_batch(edges, scores, lambda_=cfg.rings.lambda_headline,
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=cfg.rings.top_k, g_min=cfg.rings.g_min)
        block = evaluate_rings(rings, split.labels, ltv, restrict_to=split.test) if rings else {}
        return {"n_rings": len(rings), "tau": round(float(tau), 6),
                "share_of_accounts_kept": round(float((scores > tau).mean()), 6), **block}

    return {
        "operating_point": {"share_of_accounts_kept": round(share_kept, 6),
                            "xgboost_tau": cfg.rings.prune_tau_headline,
                            "fabp_tau_matched_to_same_share": round(tau_fabp, 6)},
        "xgboost_pruned": peel_and_eval(xgb_scores, cfg.rings.prune_tau_headline),
        "fabp_pruned": peel_and_eval(fabp_scores, tau_fabp),
        "fabp_h_h": beliefs["h_h"],
    }


# -------------------------------------------------------- bipartite variant --

def load_bipartite_graph(cfg: Config, split) -> dict:
    """Account-entity incidence over r6/r7/r8 in the late scoring window,
    weighted by the same entity-rarity formula as the account graph
    (`rarity_weight`) and capped at the same `n_max` - reusing both rather
    than inventing a new weighting scheme, and for the same reason the
    account graph caps entity size at all: an uncapped entity (a coupon-type
    default used by 96% of active accounts) would dominate its own degree
    and, through the convergence bound, cap `h_h` for the *entire* graph
    down near zero, killing the propagation signal everywhere rather than
    just at that one entity.
    """
    from eval.offers import PROMO_RELATIONS, entity_groups
    from orbweaver.data.build_graph import rarity_weight
    from orbweaver.data.windows import LATE, week2_windows

    proc = cfg.abs_path(cfg.paths.processed)
    orders = pq.read_table(proc / "orders_week2.parquet")
    day = orders["day_ordinal"].to_numpy()
    lo, hi = week2_windows(cfg)[LATE]
    inwin = (day >= lo) & (day <= hi)
    uid_all = orders["user_id"].to_numpy()[inwin]

    n_accounts = split.labels.size
    entity_index: dict[tuple[str, int], int] = {}
    next_id = n_accounts
    src_parts, dst_parts, w_parts = [], [], []

    for rel in PROMO_RELATIONS:
        ent_full = orders[rel].to_numpy(zero_copy_only=False).astype(np.float64)[inwin]
        ent_ids, starts, sizes, u_sorted = entity_groups(uid_all, ent_full)
        keep = (sizes >= 2) & (sizes <= cfg.graph.n_max)
        weights = rarity_weight(sizes[keep], cfg.graph.rarity_base)
        for entity, s, sz, wt in zip(ent_ids[keep], starts[keep], sizes[keep], weights):
            node_id = next_id
            next_id += 1
            entity_index[(rel, int(entity))] = node_id
            members = u_sorted[s:s + sz]
            src_parts.append(members)
            dst_parts.append(np.full(members.size, node_id, dtype=np.int64))
            w_parts.append(np.full(members.size, wt, dtype=np.float64))

    src = np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64)
    w = np.concatenate(w_parts) if w_parts else np.empty(0, dtype=np.float64)
    n_total = next_id
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([w, w])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_total, n_total))
    d_diag = np.asarray(A.sum(axis=1)).ravel()
    return {"A": A, "d_diag": d_diag, "n_accounts": n_accounts,
            "n_entities": n_total - n_accounts, "entity_index": entity_index,
            "n_edges": int(src.size)}


def bipartite_offer_beliefs(cfg: Config, split, account_graph_lift: float) -> dict:
    """Entities receive FaBP beliefs directly from the accounts that redeemed
    them, compared against the label-free leakage ranking from "Which offers
    are being farmed" on the same precision@k evaluation, over the same
    offers - which is possible without persisting any account's membership
    because `build_offer_table` is called fresh here (exactly as
    `eval.offers.run` does) and `members` is dropped again before this
    function returns, the same discipline that file's own artefact already
    follows.
    """
    from eval.offers import build_offer_table, leakage_score, precision_at_k, rank_offers

    n = split.labels.size
    proc = cfg.abs_path(cfg.paths.processed)
    xgb_scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    xgb_scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    ring_report = json.loads((proc / "ring_report.json").read_text())
    ring_of = np.full(n, -1, dtype=np.int32)
    ring_ids_by_account: dict[int, set] = {}
    for c in ring_report.get("case_files", []):
        for m in c["members"]:
            ring_of[m] = c["rank"]
            ring_ids_by_account.setdefault(m, set()).add(c["rank"])

    offers, _ = build_offer_table(cfg, split, xgb_scores, ring_of, ring_ids_by_account)
    for o in offers:
        o["_leak"] = leakage_score(o["redeemers"], o["redeemers_in_a_ring"], o["mean_score"])

    bp = load_bipartite_graph(cfg, split)
    n_total = bp["A"].shape[0]
    visible = np.zeros(n_total, dtype=bool)
    visible[:n][split.train] = True
    visible[:n][split.val] = True
    if visible[:n][split.test].any():
        raise RuntimeError("held-out accounts are visible to the bipartite FaBP prior")

    choice = choose_h_h(bp["d_diag"], account_graph_lift)
    h_h = choice["h_h"]

    labels_padded = np.concatenate([split.labels, np.full(bp["n_entities"], UNLABELLED, dtype=split.labels.dtype)])
    phi = build_prior(labels_padded, visible)
    solved = fabp_beliefs_for_prior(bp["A"], bp["d_diag"], h_h, phi)
    beliefs = solved["scores"]

    for o in offers:
        node_id = bp["entity_index"].get((o["relation"], o["entity"]))
        o["fabp_belief"] = round(float(beliefs[node_id]), 6) if node_id is not None else None

    comparable = [o for o in offers if o["fabp_belief"] is not None]
    ranked_fabp = sorted(comparable, key=lambda o: (-o["fabp_belief"], -o["redeemers"], o["entity"]))
    ranked_leakage = rank_offers(comparable, "ring_share")

    base_rate = float((split.labels[split.labels != -1] == 1).mean())
    prec_fabp = precision_at_k(ranked_fabp, split.labels, base_rate, seed=cfg.seed + 7)
    prec_leakage = precision_at_k(ranked_leakage, split.labels, base_rate, seed=cfg.seed + 7)

    for o in offers:
        o.pop("members", None)

    return {
        "n_offers_from_offers_json_method": len(offers),
        "n_comparable_within_bipartite_cap": len(comparable),
        "excluded_outside_bipartite_cap": len(offers) - len(comparable),
        "bipartite_graph": {"accounts": bp["n_accounts"], "entities": bp["n_entities"],
                            "edges": bp["n_edges"]},
        "h_h": h_h, "h_h_choice": choice, "solve_seconds": solved["solve_seconds"],
        "precision_at_k": {"fabp_belief_ranked": prec_fabp, "leakage_ranked": prec_leakage},
    }


# ------------------------------------------------------------------------ run --

def _peak_rss_bytes() -> int:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB; macOS (this project's platform) reports
    # it directly in bytes. Getting this wrong under-reports by 1024x.
    return rss if sys.platform == "darwin" else rss * 1024


def run(cfg: Config | None = None) -> dict:
    from eval.split import make_split

    cfg = cfg or load_config()
    split = make_split(cfg)
    n = split.labels.size

    print("loading the account graph ...", flush=True)
    t0 = time.time()
    A, d_diag = load_account_graph(cfg, n)
    load_seconds = time.time() - t0
    print(f"  {A.nnz:,} nonzeros in A, loaded in {load_seconds:.1f}s", flush=True)

    print("solving FaBP on the full account graph ...", flush=True)
    headline = node_beliefs(cfg, split, A=A, d_diag=d_diag)
    print(f"  h_h={headline['h_h']:.6g}  {headline['iterations']} iterations  "
          f"{headline['solve_seconds']}s", flush=True)

    print("label-budget curve: FaBP and GraphSAGE at every point of item 3's sweep ...", flush=True)
    curve = label_curve(cfg)

    print("ring test: pruning on FaBP beliefs at the same operating point as the headline ...",
          flush=True)
    rings = ring_test(cfg, split)

    print("bipartite account-entity variant ...", flush=True)
    bipartite = bipartite_offer_beliefs(cfg, split, headline["assortativity"]["lift"])

    account_matrix_bytes = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    peak_rss = _peak_rss_bytes()

    out = {
        "hypothesis": ("Propagation wins when confirmed labels are scarce and loses when they "
                      "are plentiful: it needs no feature model, only the graph and whatever "
                      "priors already exist, while a feature model needs enough labelled rows "
                      "to fit one at all."),
        "method": ("[I + aD - c'A] b_h = phi_h, solved by the paper's own power iteration to a "
                  f"fixed tolerance ({TOLERANCE}). h_h is set from the account graph's own "
                  "measured fraud-fraud lift and capped at whichever of the paper's two "
                  "convergence bounds (Lemma 5, Lemma 6) is looser; the bound is re-checked "
                  "and enforced immediately before every solve, not assumed from the choice "
                  "that produced it. Priors come from training-pool labels only, at every "
                  "point in this file."),
        "graph": {"accounts": n, "edges": int(A.nnz // 2), "nonzeros_in_a": int(A.nnz),
                 "estimated_matrix_memory_mb": round(account_matrix_bytes / 1e6, 1),
                 "load_seconds": round(load_seconds, 1)},
        "headline": {k: v for k, v in headline.items() if k != "scores"},
        "label_budget_curve": curve,
        "ring_test": rings,
        "bipartite_offers": bipartite,
        "runtime_memory": {
            "full_graph_accounts": n, "full_graph_edges": int(A.nnz // 2),
            "peak_rss_mb": round(peak_rss / 1e6, 1),
            "fits_in_16gb": bool(peak_rss < 16 * 1_000_000_000),
        },
    }
    dest = cfg.abs_path(cfg.paths.processed) / "propagate.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    run()
