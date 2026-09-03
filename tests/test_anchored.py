"""Tests for anchored extraction and ring identity across nights.

The toy graphs are built by hand so that every expected answer can be checked
on paper, which is the point: the events the matcher assigns are the thing an
operations team would act on, and they have to be right on a case where I know
what right is.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbweaver.config import load_config
from orbweaver.rings import anchored as A
from orbweaver.rings.peel import build_csr

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
needs_run = pytest.mark.skipif(not (PROC / "anchored.json").exists(),
                               reason="run `make anchored` first")


def two_cliques(n_a=8, n_b=8, bridge=True):
    """Two cliques with one weak bridge; nodes 0..n_a-1 and n_a..n_a+n_b-1,
    plus a few outsiders (30+) below the score cut-off."""
    src, dst, w = [], [], []
    for grp in (range(0, n_a), range(n_a, n_a + n_b)):
        g = list(grp)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                src.append(g[i]); dst.append(g[j]); w.append(1.0)
    if bridge:
        src.append(0); dst.append(n_a); w.append(0.05)
    # outsiders below tau, attached to node 0
    for o in range(30, 34):
        src.append(0); dst.append(o); w.append(5.0)
    n = 40
    csr = build_csr(np.asarray(src), np.asarray(dst), np.asarray(w), n)
    scores = np.zeros(n); scores[: n_a + n_b] = 0.9
    in_ref = scores > 0.5
    return csr, scores, in_ref


def test_anchor_is_always_a_member_and_ring_lies_inside_reference():
    csr, scores, in_ref = two_cliques()
    for a in (0, 3, 8, 15):
        r = A.ring_around(csr, scores, in_ref, a, lambda_=1.0, k_min=3, k_max=50)
        assert r is not None
        assert a in r.members.tolist()
        assert in_ref[r.members].all(), "an outsider got in"


def test_ring_is_local_to_its_anchor():
    """Anchored at a node in clique A, the ring is clique A - not the union,
    which a global peel would happily return as one denser blob."""
    csr, scores, in_ref = two_cliques()
    r = A.ring_around(csr, scores, in_ref, 3, lambda_=1.0, k_min=3, k_max=50)
    assert set(r.members.tolist()) == set(range(0, 8))
    r = A.ring_around(csr, scores, in_ref, 12, lambda_=1.0, k_min=3, k_max=50)
    assert set(r.members.tolist()) == set(range(8, 16))


def test_outsiders_never_enter_even_when_heavily_linked():
    """The strict form: the ball is intersected with R before peeling, so a
    heavy edge to an account below the cut-off is not evidence."""
    csr, scores, in_ref = two_cliques()
    nodes = A.ball(csr, 0, in_ref)
    assert in_ref[nodes].all()
    assert 30 not in nodes.tolist()


def test_ball_cap_keeps_the_anchor_and_nearest_first():
    csr, scores, in_ref = two_cliques()
    nodes = A.ball(csr, 0, in_ref, cap=4)
    assert nodes.size == 4 and nodes[0] == 0
    assert in_ref[nodes].all()


def test_dedupe_is_idempotent_and_keeps_the_first():
    # No bridge: with one, anchor 0's two-hop ball reaches the whole of clique
    # B and the 16-node union is denser than either clique by the bridge's
    # weight, which is the objective doing its job and not what this tests.
    csr, scores, in_ref = two_cliques(bridge=False)
    anchors = np.array([0, 1, 2, 8, 9])
    found, uniq = A.extract_night(csr, scores, in_ref, anchors, lambda_=1.0, k_min=3, k_max=50)
    assert len(found) == 5 and len(uniq) == 2
    assert [r.anchor for r in uniq] == [0, 8]
    again = A.dedupe(uniq)
    assert [r.anchor for r in again] == [r.anchor for r in uniq]


def test_extraction_is_deterministic():
    csr, scores, in_ref = two_cliques()
    anchors = A.choose_anchors(scores, in_ref, 6)
    _, a = A.extract_night(csr, scores, in_ref, anchors, lambda_=1.0, k_min=3, k_max=50)
    _, b = A.extract_night(csr, scores, in_ref, anchors, lambda_=1.0, k_min=3, k_max=50)
    assert [r.anchor for r in a] == [r.anchor for r in b]
    assert all(np.array_equal(x.members, y.members) for x, y in zip(a, b))


def _ring(members, anchor=None):
    m = np.asarray(sorted(members), dtype=np.int64)
    return A.AnchoredRing(anchor=int(anchor if anchor is not None else m[0]), members=m,
                          density=1.0, internal_weight=1.0, score_mass=1.0,
                          ball_size=m.size, seconds=0.0)


def test_matcher_reproduces_all_five_events_on_a_three_night_toy():
    """Night 1: A and B. Night 2: A grows, B splits in two, C is born.
    Night 3: A and C merge, B's larger half continues, its smaller half dies."""
    tr = A.Tracker(theta=0.3)
    A1, B1 = _ring(range(1, 11)), _ring(range(20, 31))
    n1 = tr.observe(1, [A1, B1])
    assert [e["event"] for e in n1] == ["born", "born"]
    id_a, id_b = n1[0]["case_id"], n1[1]["case_id"]

    A2 = _ring(list(range(1, 11)) + [11])
    Bbig, Bsmall = _ring(range(20, 26)), _ring(range(26, 31))
    C2 = _ring(range(40, 51))
    n2 = tr.observe(2, [A2, Bbig, Bsmall, C2])
    assert [e["event"] for e in n2] == ["continued", "continued", "split", "born"]
    assert n2[0]["case_id"] == id_a and n2[0]["born"] == 1
    assert n2[1]["case_id"] == id_b, "the larger half keeps the case"
    assert n2[2]["split_from"] == id_b and n2[2]["born"] == 2
    id_c = n2[3]["case_id"]
    assert tr.nights[-1]["split"] == 1 and tr.nights[-1]["born"] == 1

    M3 = _ring(list(range(1, 12)) + list(range(40, 51)))
    B3 = _ring(range(20, 26))
    n3 = tr.observe(3, [M3, B3])
    assert n3[0]["event"] == "merged"
    assert set(n3[0]["predecessors"]) == {id_a, id_c}
    assert n3[0]["case_id"] == id_a, "ties on Jaccard go to the lower index"
    assert n3[0]["born"] == 1, "the merged ring carries the older timeline"
    assert n3[1]["event"] == "continued" and n3[1]["case_id"] == id_b
    last = tr.nights[-1]
    assert last["died"] == 1, "B's small half had no successor"
    assert last["merged_into"] == 1, "C was absorbed"


def test_threshold_changes_what_counts_as_the_same_ring():
    a, b = _ring(range(0, 10)), _ring(range(6, 16))     # Jaccard 4/16 = 0.25
    loose, strict = A.Tracker(0.2), A.Tracker(0.3)
    loose.observe(1, [a]); strict.observe(1, [a])
    assert loose.observe(2, [b])[0]["event"] == "continued"
    assert strict.observe(2, [b])[0]["event"] == "born"


def test_peel_pinned_respects_the_size_band():
    csr, scores, in_ref = two_cliques(n_a=12, n_b=3, bridge=False)
    r = A.ring_around(csr, scores, in_ref, 0, lambda_=1.0, k_min=3, k_max=6)
    assert 3 <= r.size <= 6 and 0 in r.members.tolist()


@needs_run
def test_final_night_rings_lie_inside_the_reference_set_and_contain_their_anchor():
    import pyarrow.parquet as pq
    out = json.loads((PROC / "anchored.json").read_text())
    scores = np.zeros(int(pq.read_table(PROC / "nodes.parquet").num_rows))
    t = pq.read_table(PROC / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()
    tau = out["operating_point"]["tau"]
    for r in out["final_rings"]:
        m = np.asarray(r["members"])
        assert r["anchor"] in m.tolist()
        assert (scores[m] > tau).all()
        assert out["operating_point"]["k_min"] <= m.size <= out["operating_point"]["k_max"]


@needs_run
def test_final_night_rings_reproduce_from_the_late_graph():
    """Recompute the recorded final-night rings around their recorded anchors
    from the standard late graph, which the replay asserts is the same graph
    as the last night's, and check the members match."""
    import pyarrow.parquet as pq
    out = json.loads((PROC / "anchored.json").read_text())
    op = out["operating_point"]
    n = int(pq.read_table(PROC / "nodes.parquet").num_rows)
    scores = np.zeros(n)
    t = pq.read_table(PROC / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()
    e = pq.read_table(PROC / "edges_week2_late.parquet", columns=["src", "dst", "weight"])
    src, dst = e["src"].to_numpy().astype(np.int64), e["dst"].to_numpy().astype(np.int64)
    w = e["weight"].to_numpy().astype(np.float64)
    in_ref = scores > op["tau"]
    m = in_ref[src] & in_ref[dst]
    csr = build_csr(src[m], dst[m], w[m], n)
    for r in out["final_rings"][:10]:
        again = A.ring_around(csr, scores, in_ref, r["anchor"], lambda_=op["lambda"],
                              k_min=op["k_min"], k_max=op["k_max"])
        assert again is not None
        assert again.members.tolist() == r["members"], f"ring around {r['anchor']} changed"


@needs_run
def test_days_to_detection_is_now_measurable_and_bounded():
    out = json.loads((PROC / "anchored.json").read_text())
    nights = out["window"]["nights"]
    for r in out["final_rings"]:
        assert 1 <= r["days_to_detection"] <= nights
        assert r["first_seen_night"] == r["days_to_detection"]
    s = out["summary"]["days_to_detection"]
    assert sum(s["histogram"].values()) == len(out["final_rings"])


@needs_run
def test_check_returns_a_live_ring_only_for_suspicious_accounts():
    """The console route computes the ring around an account on demand. It
    must refuse for an account below the cut-off rather than inventing a ball
    outside the reference set."""
    import numpy as np

    from orbweaver.console.check import CheckIndex, render_card

    ix = CheckIndex()
    out = json.loads((PROC / "anchored.json").read_text())
    assert int(ix.in_ref.sum()) == out["nights"][-1]["reference_set"], \
        "the console's reference set is not the one the extraction used"

    below = int(np.flatnonzero(~ix.in_ref)[0])
    r = ix.check(below)
    assert r["anchored_ring"] is None
    assert '<div class="card">' in render_card(r)

    anchor = out["final_rings"][0]["anchor"]
    r = ix.check(anchor)
    ar = r["anchored_ring"]
    assert ar is not None and ar["found"], "no live ring around a known anchor"
    assert ar["size"] >= out["operating_point"]["k_min"]
    assert ar["case"] is not None, "the live ring around an anchor lost its case"
    assert ar["case"]["case_id"] == out["final_rings"][0]["case_id"]
    assert '<div class="card">' in render_card(r)
