"""A ring you can find again tomorrow.

The nightly replay found that ring identity does not survive a night. Peeling
is a global optimisation: one more day of edges shifts densities everywhere,
and the top rings are recomposed rather than extended. The best overlap between
a final ring and anything from an earlier night was a median Jaccard of 0.124,
and not one of the 25 final rings had a recognisable predecessor. A team cannot
open a case on Monday and find it on Tuesday, and days-to-detection could not
be measured at all.

This extracts rings **around anchors** instead. For an anchor account `a`, the
candidate set is the ball `{a} ∪ N(a) ∪ N²(a)` inside the reference set
`R = {v : s(v) > τ}`, and greedy peeling runs on that ball with `a` pinned so it
is never removed. Because the anchor is fixed and the ball is local, the ring
found tomorrow around the same anchor is mostly the same accounts, and the
question "is this still the same case?" has an answer.

This is the *anchored densest subgraph* of Dai, Qiao, Chang and Qin (SIGMOD
2022) in its strict form. Their objective is the R-subgraph density

    ρ_R(S) = (2|E(S)| − Σ_{v ∈ S∖R} deg(v)) / |S|,    S ⊇ A,

which rewards density around the anchors and penalises every outsider by its
degree. Here the ball is intersected with R before peeling, so `S ∖ R` is empty
by construction, the penalty term vanishes, and ρ_R reduces to the weighted
density this project already uses. That is the special case in which outsiders
are forbidden rather than penalised; the general case, the exact flow method,
and the later convex-programming and approximate dynamic versions are not
implemented.

Identity from night to night follows Greene, Doyle and Cunningham (*Tracking
the Evolution of Communities in Dynamic Social Networks*, ASONAM 2010): tonight's
rings are matched to last night's by Jaccard above a threshold θ - they used
0.3, and both 0.3 and 0.5 are reported - and each ring is assigned one of the
life-cycle events: born, continued, merged, split, or died. A case id is the id
of the first ring in a timeline and is carried along it. Persistence as an
objective in its own right is Semertzidis, Pitoura, Terzi and Tsaparas (DMKD
2019); that is cited, not implemented.
"""
from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field

import numpy as np

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import CSR, EdgeList, Ring, build_csr

# Stated in the results section, all of them.
N_ANCHORS = 500                  # top-N accounts by score inside R, per night
ANCHOR_SWEEP = (200, 500, 1000)  # N's effect, reported on the final night
BALL_CAP = 3000                  # nodes per candidate ball, by descending weight
DEDUPE_JACCARD = 0.5             # two anchors that found the same ring
THETA_GREENE = 0.3               # Greene et al. 2010 used 0.3
THETA_STRICT = 0.5               # the replay's original, stricter bar
LATENCY_SAMPLE = 1000            # anchors timed for the on-demand figure
POLICY_QUEUE_DEPTH = 200         # nightly queue kept for the review policy


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


# ---------------------------------------------------------------- the ball --

def ball(csr: CSR, anchor: int, in_ref: np.ndarray, cap: int = BALL_CAP) -> np.ndarray:
    """`{a} ∪ N(a) ∪ N²(a)`, inside the reference set, capped.

    Order is the anchor, then its neighbours by descending edge weight, then
    the two-hop nodes by the heaviest edge that reaches them, so the cap drops
    the weakest two-hop nodes first and never the anchor's own neighbours
    until those alone exceed it. Ties break on account id so the ball is the
    same every time.
    """
    lo, hi = csr.indptr[anchor], csr.indptr[anchor + 1]
    n1, w1 = csr.indices[lo:hi], csr.weights[lo:hi]
    keep = in_ref[n1] & (n1 != anchor)
    n1, w1 = n1[keep], w1[keep]
    order = np.lexsort((n1, -w1))
    n1 = n1[order]

    taken = np.zeros(csr.n_nodes, dtype=bool)
    taken[anchor] = True
    taken[n1] = True

    if n1.size:
        starts, ends = csr.indptr[n1], csr.indptr[n1 + 1]
        idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) \
            if n1.size else np.empty(0, dtype=np.int64)
        cand, cw = csr.indices[idx], csr.weights[idx]
        keep = in_ref[cand] & ~taken[cand]
        cand, cw = cand[keep], cw[keep]
        if cand.size:
            uniq, inv = np.unique(cand, return_inverse=True)
            best = np.zeros(uniq.size, dtype=np.float64)
            np.maximum.at(best, inv, cw)
            n2 = uniq[np.lexsort((uniq, -best))]
        else:
            n2 = np.empty(0, dtype=np.int64)
    else:
        n2 = np.empty(0, dtype=np.int64)

    nodes = np.concatenate([np.array([anchor], dtype=np.int64),
                            n1.astype(np.int64), n2.astype(np.int64)])
    return nodes[:cap]


def induced(csr: CSR, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The subgraph on `nodes`, relabelled 0..b-1 in the order given."""
    pos = np.full(csr.n_nodes, -1, dtype=np.int64)
    pos[nodes] = np.arange(nodes.size)
    ptr = [0]
    idx_parts, w_parts = [], []
    for v in nodes:
        lo, hi = csr.indptr[v], csr.indptr[v + 1]
        nbr, w = csr.indices[lo:hi], csr.weights[lo:hi]
        local = pos[nbr]
        m = local >= 0
        idx_parts.append(local[m])
        w_parts.append(w[m])
        ptr.append(ptr[-1] + int(m.sum()))
    indptr = np.asarray(ptr, dtype=np.int64)
    indices = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64)
    weights = np.concatenate(w_parts) if w_parts else np.empty(0, dtype=np.float64)
    return indptr, indices, weights


# ------------------------------------------------------------- pinned peel --

def peel_pinned(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                scores: np.ndarray, pinned: np.ndarray, *, lambda_: float,
                k_min: int, k_max: int) -> tuple[np.ndarray, float, float, float]:
    """Greedy peeling on a small local graph with some nodes never removed.

    The same objective as `peel.peel_once` - (Σ edge weight + λ Σ score) / |S| -
    and the same lazy-deletion heap, with two differences: pinned nodes are
    never pushed onto the heap, so the anchor is in every prefix and therefore
    in the answer; and the best prefix is taken only among sizes in
    [k_min, k_max], as the headline extraction does.

    Pinning does not break the ½-approximation argument for the unconstrained
    problem *restricted to sets containing the anchor*: the set returned is the
    best of a peeling sequence that starts from the whole ball and never drops
    the anchor, and Charikar's bound compares against the densest set of that
    family. It is a bound on the search, not on whether density means fraud.
    """
    b = int(scores.size)
    alive = np.ones(b, dtype=bool)
    deg = np.zeros(b, dtype=np.float64)
    np.add.reduceat(weights, indptr[:-1], out=deg) if weights.size else None
    deg[np.diff(indptr) == 0] = 0.0
    contrib = deg + lambda_ * scores
    edge_w = float(deg.sum()) / 2.0
    score_mass = float(scores.sum())
    n_alive = b

    def g(n, ew, sm):
        return (ew + lambda_ * sm) / n if n else -np.inf

    best_g, best_n = -np.inf, None
    if k_min <= n_alive <= k_max:
        best_g, best_n = g(n_alive, edge_w, score_mass), n_alive

    heap = [(contrib[v], v) for v in range(b) if not pinned[v]]
    heapq.heapify(heap)
    removal: list[int] = []
    while n_alive > k_min and heap:
        c, v = heapq.heappop(heap)
        if not alive[v] or c != contrib[v]:
            continue
        alive[v] = False
        n_alive -= 1
        removal.append(v)
        lo, hi = indptr[v], indptr[v + 1]
        nbr, w = indices[lo:hi], weights[lo:hi]
        m = alive[nbr]
        edge_w -= float(w[m].sum())
        score_mass -= float(scores[v])
        for u, wu in zip(nbr[m], w[m]):
            contrib[u] -= wu
            if not pinned[u]:
                heapq.heappush(heap, (contrib[u], int(u)))
        cur = g(n_alive, edge_w, score_mass)
        if k_min <= n_alive <= k_max and cur > best_g:
            best_g, best_n = cur, n_alive

    if best_n is None:
        return np.empty(0, dtype=np.int64), -np.inf, 0.0, 0.0
    keep = np.ones(b, dtype=bool)
    for v in removal[: b - best_n]:
        keep[v] = False
    members = np.flatnonzero(keep)
    inside = keep
    ew = 0.0
    for v in members:
        lo, hi = indptr[v], indptr[v + 1]
        ew += float(weights[lo:hi][inside[indices[lo:hi]]].sum())
    ew /= 2.0
    sm = float(scores[members].sum())
    return members, (ew + lambda_ * sm) / max(members.size, 1), ew, sm


# ----------------------------------------------------- one night's rings --

@dataclass
class AnchoredRing:
    anchor: int
    members: np.ndarray          # sorted account ids, anchor included
    density: float
    internal_weight: float
    score_mass: float
    ball_size: int
    seconds: float
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.members.size)

    def as_ring(self, lambda_: float, rank: int) -> Ring:
        return Ring(members=self.members.astype(np.int32), density=self.density,
                    internal_weight=self.internal_weight, score_mass=self.score_mass,
                    lambda_=lambda_, rank=rank)


def ring_around(csr: CSR, scores: np.ndarray, in_ref: np.ndarray, anchor: int, *,
                lambda_: float, k_min: int, k_max: int,
                cap: int = BALL_CAP) -> AnchoredRing | None:
    """The anchored ring for one account, or None if its ball is too small."""
    t0 = time.perf_counter()
    nodes = ball(csr, anchor, in_ref, cap)
    if nodes.size < k_min:
        return None
    indptr, indices, weights = induced(csr, nodes)
    pinned = np.zeros(nodes.size, dtype=bool)
    pinned[0] = True
    local, dens, ew, sm = peel_pinned(indptr, indices, weights, scores[nodes],
                                      pinned, lambda_=lambda_, k_min=k_min, k_max=k_max)
    if local.size < k_min:
        return None
    members = np.sort(nodes[local])
    return AnchoredRing(anchor=int(anchor), members=members, density=float(dens),
                        internal_weight=float(ew), score_mass=float(sm),
                        ball_size=int(nodes.size), seconds=time.perf_counter() - t0)


def dedupe(rings: list[AnchoredRing], threshold: float = DEDUPE_JACCARD) -> list[AnchoredRing]:
    """Keep the first of any group of rings that overlap at `threshold`.

    Rings arrive in descending anchor-score order, so the survivor of a group
    is the one found from the highest-scoring anchor. Running this on its own
    output changes nothing; a test asserts that.
    """
    kept: list[AnchoredRing] = []
    for r in rings:
        if all(jaccard(r.members, k.members) < threshold for k in kept):
            kept.append(r)
    return kept


def choose_anchors(scores: np.ndarray, in_ref: np.ndarray, n_top: int,
                   carried: np.ndarray | None = None) -> np.ndarray:
    """Top-N by score inside R, plus every member of last night's rings.

    Sorted by descending score with ties broken on id, so the order in which
    rings are extracted - and therefore which anchor a de-duplicated ring keeps -
    is the same every run.
    """
    ref = np.flatnonzero(in_ref)
    order = np.lexsort((ref, -scores[ref]))
    top = ref[order[:n_top]]
    if carried is not None and carried.size:
        carried = carried[in_ref[carried]]
        top = np.union1d(top, carried)
    return top[np.lexsort((top, -scores[top]))]


def extract_night(csr: CSR, scores: np.ndarray, in_ref: np.ndarray,
                  anchors: np.ndarray, *, lambda_: float, k_min: int, k_max: int,
                  cap: int = BALL_CAP) -> tuple[list[AnchoredRing], list[AnchoredRing]]:
    """All anchored rings for one night: (before de-duplication, after)."""
    found: list[AnchoredRing] = []
    for a in anchors:
        r = ring_around(csr, scores, in_ref, int(a), lambda_=lambda_,
                        k_min=k_min, k_max=k_max, cap=cap)
        if r is not None:
            found.append(r)
    return found, dedupe(found)


# ------------------------------------------------------- identity, nightly --

@dataclass
class Front:
    """A ring as of last night, carrying its case id and when it was born."""
    case_id: int
    members: np.ndarray
    born: int                    # night index, 1-based
    anchor: int


def match_night(fronts: list[Front], rings: list[AnchoredRing], night: int,
                theta: float, next_id: int) -> tuple[list[dict], list[dict], int]:
    """Greene et al.'s matching for one night.

    Each of tonight's rings takes its case id from its best predecessor if it
    is also that predecessor's best successor (continued; merged if it had
    more than one predecessor above θ). A ring whose best predecessor
    continues elsewhere is a split and starts a new timeline. A ring with no
    predecessor is born. A front with no successor dies tonight; a front whose
    successors all inherited from someone else was merged into one of them.
    Ties on Jaccard break on the lower index, so this is deterministic.
    """
    F, Rn = len(fronts), len(rings)
    J = np.zeros((F, Rn), dtype=np.float64)
    for i, f in enumerate(fronts):
        for j, r in enumerate(rings):
            J[i, j] = jaccard(f.members, r.members)
    preds = [np.flatnonzero(J[:, j] >= theta) for j in range(Rn)]
    succs = [np.flatnonzero(J[i, :] >= theta) for i in range(F)]
    best_succ = {i: int(s[np.argmax(J[i, s])]) for i, s in enumerate(succs) if s.size}

    out: list[dict] = []
    for j, r in enumerate(rings):
        p = preds[j]
        if p.size == 0:
            out.append({"case_id": next_id, "event": "born", "born": night,
                        "predecessors": [], "jaccard": None})
            next_id += 1
            continue
        bi = int(p[np.argmax(J[p, j])])
        pred_ids = [fronts[i].case_id for i in p.tolist()]
        if best_succ.get(bi) == j:
            out.append({"case_id": fronts[bi].case_id,
                        "event": "merged" if p.size > 1 else "continued",
                        "born": fronts[bi].born, "predecessors": pred_ids,
                        "jaccard": round(float(J[bi, j]), 4)})
        else:
            out.append({"case_id": next_id, "event": "split", "born": night,
                        "split_from": fronts[bi].case_id, "predecessors": pred_ids,
                        "jaccard": round(float(J[bi, j]), 4)})
            next_id += 1

    inherited = {o["case_id"] for o in out if o["event"] in ("continued", "merged")}
    ended: list[dict] = []
    for i, f in enumerate(fronts):
        if f.case_id in inherited:
            continue
        if succs[i].size == 0:
            ended.append({"case_id": f.case_id, "event": "died", "born": f.born})
        else:
            ended.append({"case_id": f.case_id, "event": "merged_into",
                          "into": out[best_succ[i]]["case_id"], "born": f.born})
    return out, ended, next_id


class Tracker:
    """Timelines across nights at one θ."""

    def __init__(self, theta: float):
        self.theta = theta
        self.fronts: list[Front] = []
        self.next_id = 1
        self.nights: list[dict] = []

    def observe(self, night: int, rings: list[AnchoredRing]) -> list[dict]:
        assigned, ended, self.next_id = match_night(
            self.fronts, rings, night, self.theta, self.next_id)
        self.fronts = [Front(case_id=a["case_id"], members=r.members, born=a["born"],
                             anchor=r.anchor) for a, r in zip(assigned, rings)]
        matched = [a["jaccard"] for a in assigned if a["jaccard"] is not None
                   and a["event"] in ("continued", "merged")]
        self.nights.append({
            "night": night,
            "rings": len(rings),
            "born": sum(a["event"] == "born" for a in assigned),
            "continued": sum(a["event"] == "continued" for a in assigned),
            "merged": sum(a["event"] == "merged" for a in assigned),
            "split": sum(a["event"] == "split" for a in assigned),
            "died": sum(e["event"] == "died" for e in ended),
            "merged_into": sum(e["event"] == "merged_into" for e in ended),
            "share_with_a_predecessor": round(
                float(np.mean([a["event"] != "born" for a in assigned])), 4)
            if assigned else None,
            "median_jaccard_of_continued": round(float(np.median(matched)), 4)
            if matched else None,
        })
        return assigned


# ------------------------------------------------------------- the night --

def night_graph(cfg: Config, lo: int, d: int, scores_of, tau: float):
    """Build what was visible by night `d`, score it, prune to R, as CSR."""
    import pyarrow.parquet as pq
    from orbweaver.data.build_graph import build_graph
    from orbweaver.features.node_features import build_features
    from orbweaver.scoring.xgb_graph import load_features

    proc = cfg.abs_path(cfg.paths.processed)
    tag = f"late_upto_{d}"
    t0 = time.time()
    build_graph(2, cfg, days=(lo, d), tag=tag, force=True)
    build_features(2, cfg, days=(lo, d), tag=tag, force=True)
    n = int(pq.read_table(proc / "nodes.parquet").num_rows)
    scores = scores_of(load_features(2, cfg, n, tag)).astype(np.float64)
    e = pq.read_table(proc / f"edges_week2_{tag}.parquet", columns=["src", "dst", "weight"])
    src = e["src"].to_numpy().astype(np.int64)
    dst = e["dst"].to_numpy().astype(np.int64)
    w = e["weight"].to_numpy().astype(np.float64)
    in_ref = scores > tau
    m = in_ref[src] & in_ref[dst]
    csr = build_csr(src[m], dst[m], w[m], n)
    return csr, scores, in_ref, int(src.size), int(m.sum()), time.time() - t0


def run(cfg: Config | None = None) -> dict:
    import pyarrow.parquet as pq
    from eval.metrics import ltv_proxy
    from eval.replay import cleanup, promo_spend_by_day
    from eval.split import make_split
    from orbweaver.data.windows import EARLY, LATE, week2_windows
    from orbweaver.rings.cost import evaluate_rings
    from orbweaver.scoring.xgb_graph import load_scorer, score_features

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    tau, lam = cfg.rings.prune_tau_headline, cfg.rings.lambda_headline
    k_min, k_max, top_k = cfg.rings.k_min, cfg.rings.k_max, cfg.rings.top_k
    split = make_split(cfg)
    labels = split.labels
    n = labels.size
    model, calibrate = load_scorer(cfg)
    scores_of = lambda X: score_features(model, calibrate, X)
    lo, hi = week2_windows(cfg)[LATE]
    days = list(range(lo, hi + 1))

    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet", columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    def measure(rings: list[AnchoredRing]) -> dict:
        """Top-25 by density, the same ranking the global number rests on."""
        ranked = sorted(rings, key=lambda r: (-r.density, r.anchor))[:top_k]
        objs = [r.as_ring(lam, i + 1) for i, r in enumerate(ranked)]
        block = evaluate_rings(objs, labels, ltv, restrict_to=split.test) if objs else {}
        by_mean = sorted(rings, key=lambda r: (-float(np.mean(scores[r.members])), r.anchor))[:top_k]
        objs2 = [r.as_ring(lam, i + 1) for i, r in enumerate(by_mean)]
        block2 = evaluate_rings(objs2, labels, ltv, restrict_to=split.test) if objs2 else {}
        return {"rings_ranked": len(objs),
                "accounts_in_rings": block.get("accounts_in_rings"),
                "ring_precision": block.get("ring_precision"),
                "precision_lift_over_base": block.get("precision_lift_over_base"),
                "normal_flagged_per_fraud_caught": block.get("normal_flagged_per_fraud_caught"),
                "fraud_members": block.get("fraud_members"),
                "by_mean_member_score": {
                    "ring_precision": block2.get("ring_precision"),
                    "normal_flagged_per_fraud_caught": block2.get("normal_flagged_per_fraud_caught"),
                    "fraud_members": block2.get("fraud_members")}}

    trackers = {THETA_GREENE: Tracker(THETA_GREENE), THETA_STRICT: Tracker(THETA_STRICT)}
    nights, carried = [], None
    final = None
    print(f"{'night':>6s} {'|R|':>8s} {'edges(R)':>10s} {'anchors':>8s} {'found':>6s} "
          f"{'unique':>6s} {'prec@25':>8s} {'per catch':>9s} {'build':>6s} {'extract':>8s}")
    for i, d in enumerate(days, start=1):
        csr, scores, in_ref, n_edges, n_edges_ref, t_build = night_graph(cfg, lo, d, scores_of, tau)
        anchors = choose_anchors(scores, in_ref, N_ANCHORS, carried)
        t0 = time.time()
        found, rings = extract_night(csr, scores, in_ref, anchors,
                                     lambda_=lam, k_min=k_min, k_max=k_max)
        t_extract = time.time() - t0
        per_anchor = [r.seconds for r in found]
        assigned = {th: tr.observe(i, rings) for th, tr in trackers.items()}
        m = measure(rings)
        row = {
            "night": i, "day": d, "reference_set": int(in_ref.sum()),
            "edges": n_edges, "edges_inside_reference": n_edges_ref,
            "anchors": int(anchors.size), "anchors_carried": int(carried.size) if carried is not None else 0,
            "rings_found": len(found), "rings_after_dedupe": len(rings),
            "median_ring_size": float(np.median([r.size for r in rings])) if rings else None,
            "median_ball_size": float(np.median([r.ball_size for r in found])) if found else None,
            "final_night": m,
            "seconds": {"build_and_score": round(t_build, 1), "extract_all": round(t_extract, 1),
                        "per_anchor_median_ms": round(1000 * float(np.median(per_anchor)), 2) if per_anchor else None,
                        "per_anchor_p95_ms": round(1000 * float(np.percentile(per_anchor, 95)), 2) if per_anchor else None},
        }
        # The review policy runs night by night over these same queues. Keeping
        # them here costs a few hundred kilobytes and saves rebuilding every
        # night's graph a second time to get them back.
        queue = sorted(rings, key=lambda r: (-float(np.mean(scores[r.members])), r.anchor))[:POLICY_QUEUE_DEPTH]
        ev = {id(r): a for r, a in zip(rings, assigned[THETA_GREENE])}
        row["queue"] = [{
            "anchor": r.anchor, "size": r.size, "density": round(r.density, 4),
            "mean_member_score": round(float(np.mean(scores[r.members])), 6),
            "case_id": ev[id(r)]["case_id"], "event": ev[id(r)]["event"],
            "first_seen_night": ev[id(r)]["born"],
            "members": r.members.tolist(),
        } for r in queue]
        nights.append(row)
        print(f"{i:>6} {row['reference_set']:>8,} {n_edges_ref:>10,} {anchors.size:>8,} "
              f"{len(found):>6} {len(rings):>6} {str(m['ring_precision']):>8} "
              f"{str(m['normal_flagged_per_fraud_caught']):>9} {t_build:>6.0f} {t_extract:>8.1f}")
        carried = np.unique(np.concatenate([r.members for r in rings])) if rings else np.empty(0, dtype=np.int64)
        if i == len(days):
            final = {"csr": csr, "scores": scores, "in_ref": in_ref, "rings": rings,
                     "assigned": assigned, "anchors": anchors}
        else:
            cleanup(cfg, d)

    # ---- the final night, in detail -------------------------------------
    csr, scores, in_ref, rings = final["csr"], final["scores"], final["in_ref"], final["rings"]
    ranked = sorted(rings, key=lambda r: (-r.density, r.anchor))[:top_k]
    a03 = final["assigned"][THETA_GREENE]
    a05 = final["assigned"][THETA_STRICT]
    idx_of = {id(r): k for k, r in enumerate(rings)}
    detail = []
    for rank, r in enumerate(ranked, start=1):
        k = idx_of[id(r)]
        g_, s_ = a03[k], a05[k]
        spend = promo_spend_by_day(cfg, r.members, lo, hi)
        total = sum(spend.values())
        first_day = lo + g_["born"] - 1
        ahead = sum(v for day, v in spend.items() if day >= first_day)
        lab = labels[r.members]
        detail.append({
            "rank": rank, "anchor": r.anchor, "size": r.size, "density": round(r.density, 4),
            "mean_member_score": round(float(np.mean(scores[r.members])), 4),
            "fraud": int((lab == 1).sum()), "normal": int((lab == 0).sum()),
            "case_id": g_["case_id"], "event": g_["event"], "first_seen_night": g_["born"],
            "days_to_detection": g_["born"],
            "jaccard_to_predecessor": g_["jaccard"],
            "at_theta_0.5": {"case_id": s_["case_id"], "event": s_["event"], "first_seen_night": s_["born"]},
            "window_spend_inr": round(total, 2),
            "spend_on_or_after_first_seen_inr": round(ahead, 2),
            "share_still_ahead_when_first_seen": round(ahead / total, 4) if total else None,
            "members": r.members.tolist(),
        })

    # ---- what global peeling managed, from the replay, for the comparison --
    global_persist = None
    rp = proc / "replay.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        last = max(int(k) for k in rep["detection"][0]["best_ring_overlap_by_day"]) if rep["detection"] else None
        best_before = [max((v for k, v in r["best_ring_overlap_by_day"].items() if int(k) < last), default=0.0)
                       for r in rep["detection"]]
        # The anchored tracker only ever matches against *last night*, because
        # a front is one night old. Giving global peeling every earlier night
        # is therefore an easier test than the anchored one, so the previous
        # night alone is computed too and that is the like-for-like number.
        prev_night = max((int(k) for k in rep["detection"][0]["best_ring_overlap_by_day"] if int(k) < last),
                         default=None)
        prev_only = ([r["best_ring_overlap_by_day"][str(prev_night)] for r in rep["detection"]]
                     if prev_night is not None else [])
        global_persist = {
            "final_rings": len(best_before),
            "matched_against": "any earlier night",
            "median_best_overlap_with_an_earlier_night": round(float(np.median(best_before)), 4) if best_before else None,
            "share_with_a_predecessor_at_0.3": round(float(np.mean([b >= THETA_GREENE for b in best_before])), 4) if best_before else None,
            "share_with_a_predecessor_at_0.5": round(float(np.mean([b >= THETA_STRICT for b in best_before])), 4) if best_before else None,
            "previous_night_only": {
                "note": ("the like-for-like test: the anchored tracker only matches against "
                         "last night, so this restricts global peeling to the same one night"),
                "median_overlap": round(float(np.median(prev_only)), 4) if prev_only else None,
                "share_with_a_predecessor_at_0.3": round(float(np.mean([b >= THETA_GREENE for b in prev_only])), 4) if prev_only else None,
                "share_with_a_predecessor_at_0.5": round(float(np.mean([b >= THETA_STRICT for b in prev_only])), 4) if prev_only else None,
            },
            "final_night_precision": rep["snapshots"][-1]["ring_precision"],
            "final_night_normal_flagged_per_fraud_caught": rep["snapshots"][-1]["normal_flagged_per_fraud_caught"],
        }

    # ---- N's effect on the final night, top-N anchors only ---------------
    sweep = []
    for n_top in ANCHOR_SWEEP:
        anchors = choose_anchors(scores, in_ref, n_top, None)
        t0 = time.time()
        found, rr = extract_night(csr, scores, in_ref, anchors, lambda_=lam, k_min=k_min, k_max=k_max)
        m = measure(rr)
        sweep.append({"n_anchors": n_top, "rings_found": len(found), "rings_after_dedupe": len(rr),
                      "ring_precision": m["ring_precision"],
                      "normal_flagged_per_fraud_caught": m["normal_flagged_per_fraud_caught"],
                      "seconds": round(time.time() - t0, 1)})
        print(f"  N={n_top:<5} rings {len(rr):>4}  precision {m['ring_precision']}  "
              f"per catch {m['normal_flagged_per_fraud_caught']}  {time.time() - t0:.1f}s")

    # ---- on-demand latency: the ring around an account, computed live -----
    rng = np.random.default_rng(cfg.seed)
    ref = np.flatnonzero(in_ref)
    sample = rng.choice(ref, size=min(LATENCY_SAMPLE, ref.size), replace=False)
    lat = []
    for a in sample:
        t0 = time.perf_counter()
        ring_around(csr, scores, in_ref, int(a), lambda_=lam, k_min=k_min, k_max=k_max)
        lat.append(1000 * (time.perf_counter() - t0))
    lat = np.asarray(lat)

    # The final night's graph was kept on disk only because the sweep and the
    # latency sample needed it; both are done and the CSR is in memory, so it
    # goes the way every other night's did. Leaving it behind is a ~350 MB
    # leak that the replay's disk-hygiene test catches.
    freed = cleanup(cfg, days[-1])

    # ---- summary --------------------------------------------------------
    dtd = [r["days_to_detection"] for r in detail]
    total_spend = sum(r["window_spend_inr"] for r in detail)
    total_ahead = sum(r["spend_on_or_after_first_seen_inr"] for r in detail)
    last_night_03 = trackers[THETA_GREENE].nights[-1]
    last_night_05 = trackers[THETA_STRICT].nights[-1]
    out = {
        "method": ("Anchored densest subgraph (Dai et al., SIGMOD 2022) in the strict "
                   "form S ⊆ R: for each anchor, greedy peeling on the ball "
                   "{a} ∪ N(a) ∪ N²(a) inside R with the anchor pinned. Anchors are the "
                   "top-N accounts by score inside R plus every member of last night's "
                   "rings. Identity across nights by Jaccard matching with life-cycle "
                   "events (Greene et al., ASONAM 2010)."),
        "operating_point": {"tau": tau, "lambda": lam, "k_min": k_min, "k_max": k_max, "top_k": top_k},
        "design": {"n_anchors": N_ANCHORS, "ball_cap": BALL_CAP, "dedupe_jaccard": DEDUPE_JACCARD,
                   "theta_reported": [THETA_GREENE, THETA_STRICT], "death_after_unobserved_nights": 1},
        "window": {"first_day": lo, "last_day": hi, "nights": len(days)},
        "nights": nights,
        "timelines": {str(th): tr.nights for th, tr in trackers.items()},
        "final_rings": detail,
        "anchor_sweep_final_night": sweep,
        "global_peeling_from_replay": global_persist,
        "disk": {"final_night_bytes_freed": int(freed),
                 "note": "every night's graph and features are deleted once its "
                         "numbers are written; only manifests are kept"},
        "on_demand_latency_ms": {"samples": int(lat.size), "p50": round(float(np.percentile(lat, 50)), 3),
                                 "p95": round(float(np.percentile(lat, 95)), 3),
                                 "p99": round(float(np.percentile(lat, 99)), 3),
                                 "max": round(float(lat.max()), 3)},
        "summary": {
            "final_night": nights[-1]["final_night"],
            "persistence_at_0.3": {"share_of_final_rings_with_a_predecessor": last_night_03["share_with_a_predecessor"],
                                   "median_jaccard_of_continued": last_night_03["median_jaccard_of_continued"]},
            "persistence_at_0.5": {"share_of_final_rings_with_a_predecessor": last_night_05["share_with_a_predecessor"],
                                   "median_jaccard_of_continued": last_night_05["median_jaccard_of_continued"]},
            "days_to_detection": {"min": int(min(dtd)) if dtd else None,
                                  "median": float(np.median(dtd)) if dtd else None,
                                  "max": int(max(dtd)) if dtd else None,
                                  "share_seen_before_the_last_night": round(float(np.mean([x < len(days) for x in dtd])), 4) if dtd else None,
                                  "histogram": {str(k): int(sum(1 for x in dtd if x == k)) for k in range(1, len(days) + 1)}},
            "share_of_ring_spend_still_ahead_when_first_seen": round(total_ahead / total_spend, 4) if total_spend else None,
            "total_window_spend_inr": round(total_spend, 2),
            "spend_on_or_after_first_seen_inr": round(total_ahead, 2),
            "events_on_the_final_night_at_0.3": {k: last_night_03[k] for k in ("born", "continued", "merged", "split", "died", "merged_into")},
        },
    }
    return out


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "anchored.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    s = out["summary"]
    g = out["global_peeling_from_replay"] or {}
    print()
    print(f"final night, anchored top-25: precision {s['final_night']['ring_precision']} "
          f"at {s['final_night']['normal_flagged_per_fraud_caught']} per catch "
          f"(global: {g.get('final_night_precision')} at {g.get('final_night_normal_flagged_per_fraud_caught')})")
    p3, p5 = s["persistence_at_0.3"], s["persistence_at_0.5"]
    print(f"rings with a predecessor: {p3['share_of_final_rings_with_a_predecessor']:.1%} at θ=0.3, "
          f"{p5['share_of_final_rings_with_a_predecessor']:.1%} at θ=0.5 "
          f"(global: {g.get('share_with_a_predecessor_at_0.3')}, {g.get('share_with_a_predecessor_at_0.5')})")
    print(f"median Jaccard of continued rings: {p3['median_jaccard_of_continued']} "
          f"(global best overlap: {g.get('median_best_overlap_with_an_earlier_night')})")
    d = s["days_to_detection"]
    print(f"days to detection: median {d['median']}, range {d['min']}-{d['max']}, "
          f"{d['share_seen_before_the_last_night']:.1%} seen before the last night; histogram {d['histogram']}")
    print(f"ring spend still ahead when first seen: {s['share_of_ring_spend_still_ahead_when_first_seen']:.1%}")
    L = out["on_demand_latency_ms"]
    print(f"on-demand ring around an account: p50 {L['p50']} ms, p95 {L['p95']} ms over {L['samples']} anchors")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
