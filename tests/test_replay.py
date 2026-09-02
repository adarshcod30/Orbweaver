"""Tests for the nightly replay.

The load-bearing one is `test_last_night_reproduces_the_standard_result`. The
replay rebuilds the graph from a day range rather than from the stored late
window, so if those two paths ever disagree the whole replay is measuring
something other than the pipeline it claims to be replaying.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from orbweaver.config import load_config

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)

needs_run = pytest.mark.skipif(
    not (PROC / "replay.json").exists(), reason="run `make replay` first")


@pytest.fixture(scope="module")
def replay():
    return json.loads((PROC / "replay.json").read_text())


@needs_run
def test_last_night_reproduces_the_standard_result(replay):
    """The final prefix is the whole scoring window, so it must land on the
    same precision as the standard run. If it does not, the replay is building
    a different graph from the same days."""
    ring = json.loads((PROC / "ring_report.json").read_text())
    best = ring["best_cell"]
    standard = ring["grid"][f"tau={best['tau']},lambda={best['lambda']}"]
    last = replay["snapshots"][-1]
    assert last["ring_precision"] == standard["ring_precision"]
    assert last["accounts_in_rings"] == standard["accounts_in_rings"]
    assert last["n_rings"] == standard["n_rings"]


@needs_run
def test_replay_runs_at_the_headline_operating_point(replay):
    op = replay["operating_point"]
    assert op["tau"] == CFG.rings.prune_tau_headline
    assert op["lambda"] == CFG.rings.lambda_headline


@needs_run
def test_each_night_sees_strictly_more_than_the_one_before(replay):
    """A prefix graph must grow: every night adds a day of orders and can only
    add edges. A night that shrank would mean the day filter is wrong."""
    edges = [s["edges"] for s in replay["snapshots"]]
    assert edges == sorted(edges)
    assert len(set(edges)) == len(edges), "two nights produced identical graphs"


@needs_run
def test_days_are_consecutive_and_cover_the_window(replay):
    days = [s["day"] for s in replay["snapshots"]]
    assert days == list(range(days[0], days[0] + len(days)))
    assert len(days) == replay["window"]["days"]


@needs_run
def test_no_ring_is_reported_detected_before_it_could_be(replay):
    for r in replay["detection"]:
        assert 1 <= r["days_to_detection"] <= replay["window"]["days"]
        assert 1 <= r["days_to_half_the_members"] <= replay["window"]["days"]


@needs_run
def test_overlap_is_recorded_even_when_no_match_is_found(replay):
    """An early version initialised the best-overlap at 1.0 and only wrote to
    it on a match, so rings that never matched were reported as a perfect
    overlap. The recorded values must be real measurements."""
    for r in replay["detection"]:
        vals = list(r["best_ring_overlap_by_day"].values())
        assert all(0.0 <= v <= 1.0 for v in vals)
        # the last night is the ring's own night, so it matches itself
        last = str(max(int(k) for k in r["best_ring_overlap_by_day"]))
        assert r["best_ring_overlap_by_day"][last] == 1.0
        if not r["matched_before_last_night"]:
            earlier = [v for k, v in r["best_ring_overlap_by_day"].items()
                       if k != last]
            assert all(v < replay["match_jaccard"] for v in earlier)


@needs_run
def test_spend_accounting_is_consistent(replay):
    for r in replay["detection"]:
        assert r["spend_on_or_after_detection_inr"] <= r["window_spend_inr"] + 1e-6
        if r["window_spend_inr"] > 0:
            assert 0.0 <= r["share_still_ahead_at_detection"] <= 1.0


@needs_run
def test_per_night_graphs_were_deleted(replay):
    """Each night's graph is ~350 MB. They must not survive the run."""
    leftovers = list(PROC.glob("edges_week2_late_upto_*.parquet"))
    leftovers += list(PROC.glob("features_week2_late_upto_*.parquet"))
    assert not leftovers, f"replay left {len(leftovers)} large files behind"
