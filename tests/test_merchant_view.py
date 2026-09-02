"""Tests for the merchant-versus-platform comparison.

The claim being tested is that a relation only the platform can observe is
worth something measurable. That claim is only as good as the comparison
behind it, so most of these tests are about the comparison being fair rather
than about the machinery working.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbweaver.config import load_config

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
GAD = CFG.abs_path(CFG.paths.raw).parent / "gadbench"

needs_data = pytest.mark.skipif(
    not (GAD / "YelpChi.mat").exists(),
    reason="run `make download-gadbench` first",
)
needs_run = pytest.mark.skipif(
    not (PROC / "merchant_view.json").exists(),
    reason="run `make merchant-view` first",
)


@needs_data
def test_dropping_a_relation_removes_exactly_its_edges():
    """A dropped relation must be gone, and nothing else may move."""
    from eval.generalise import load_mat

    _, _, full, rels = load_mat("yelpchi", CFG)
    _, _, without, kept = load_mat("yelpchi", CFG, drop=("net_rur",))

    assert "net_rur" not in kept
    assert set(kept) == set(rels) - {"net_rur"}
    for r in kept:
        # the surviving relations must be untouched, edge for edge
        assert np.array_equal(full[r][0], without[r][0])
        assert np.array_equal(full[r][1], without[r][1])


@needs_data
def test_dropping_every_relation_is_refused():
    from eval.generalise import load_mat

    with pytest.raises(ValueError):
        load_mat("yelpchi", CFG, drop=("net_rur", "net_rtr", "net_rsr"))


def test_precision_at_budget_counts_in_ring_order():
    """A reviewer works down the queue, so the budget must fill in rank order
    and must not double-count an account that is in two rings."""
    from eval.generalise import precision_at_budget
    from orbweaver.rings.peel import Ring

    y = np.zeros(3000, dtype=np.int8)
    y[:200] = 1                                   # first 200 ids are fraud
    rings = [
        Ring(np.arange(0, 250), 0.0, 0.0, 0.0, 0.0),      # 200 fraud, 50 not
        Ring(np.arange(200, 500), 0.0, 0.0, 0.0, 0.0),    # overlaps, all normal
    ]
    out = precision_at_budget(rings, y, base=0.05)
    assert out["250"]["accounts"] == 250
    assert out["250"]["fraud"] == 200
    assert out["250"]["precision"] == pytest.approx(0.8)
    # 500 unique accounts, still only 200 fraud
    assert out["500"]["fraud"] == 200
    assert out["500"]["precision"] == pytest.approx(0.4)


@needs_run
def test_arms_are_compared_at_one_operating_point():
    """Letting each arm pick its own best cut-off would compare two different
    operating points and call the difference an effect of the graph."""
    d = json.loads((PROC / "merchant_view.json").read_text())
    y = d["datasets"]["yelpchi"]["arms"]
    assert y["platform"]["tau"] == y["merchant"]["tau"]


@needs_run
def test_platform_arm_matches_the_standard_generalisation_result():
    """The platform arm is the same run as the generalisation stage. If these
    ever disagree, one of the two has drifted."""
    mv = json.loads((PROC / "merchant_view.json").read_text())
    gen_path = PROC / "generalisation.json"
    if not gen_path.exists():
        pytest.skip("run `make generalise` first")
    gen = json.loads(gen_path.read_text())
    for name, block in mv["datasets"].items():
        platform = block["arms"]["platform"]
        g = gen["datasets"][name]
        assert platform["node_auprc"] == g["node_scoring_heldout"]["auprc"]
        cell = g["rings"][str(platform["tau"])]
        assert platform["ring_precision"] == cell["ring_precision"]
        assert platform["edges"] == g["edges"]


@needs_run
def test_json_shape():
    d = json.loads((PROC / "merchant_view.json").read_text())
    assert "datasets" in d and d["datasets"]
    for name, block in d["datasets"].items():
        assert "platform" in block["arms"]
        if "merchant" in block["arms"]:
            assert block["cross_business_relation"]
            assert block["at_equal_review_budget"]
            for v in block["at_equal_review_budget"].values():
                assert {"platform_precision", "merchant_precision",
                        "delta_precision"} <= set(v)
        else:
            assert block["leave_one_out"], f"{name} needs one arm or the other"
