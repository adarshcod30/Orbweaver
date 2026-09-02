"""The split is temporal, and these tests are what hold me to it.

No week-2 information may reach a training artefact. These must pass before I
report any metric. They are deliberately blunt: they check the data on disk,
not what the code intended.

The split is applied on `order_time`, never on which file a row came from.
`order_test.csv` ships 75 orders dated `1000-05-20`, a week-1 day; if the
split were "train file means train", those 75 rows would be a quiet leak in
the direction nobody thinks to check. They are dropped.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from orbweaver.config import load_config
from orbweaver.data.load_ppa import parse_day

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)

pytestmark = pytest.mark.skipif(
    not (PROC / "orders_week1.parquet").exists(),
    reason="run `make data` first",
)


def _days(week: int) -> np.ndarray:
    return np.unique(pq.read_table(PROC / f"orders_week{week}.parquet",
                                   columns=["day_ordinal"])["day_ordinal"].to_numpy())


def test_week_day_sets_are_disjoint():
    assert np.intersect1d(_days(1), _days(2)).size == 0


def test_every_week1_day_precedes_every_week2_day():
    assert _days(1).max() < _days(2).min()


def test_split_boundary_matches_config():
    w1_last = parse_day(CFG.data.week1_last_day).toordinal()
    w2_first = parse_day(CFG.data.week2_first_day).toordinal()
    assert _days(1).max() == w1_last
    assert _days(2).min() == w2_first
    assert w2_first - w1_last == 1


def test_no_week2_row_survived_into_week1():
    """The 75 boundary orders came from the test file on a week-1 day. They
    were dropped, so week 1 must hold exactly order_train.csv's row count."""
    m = json.loads((PROC / "orders_manifest.json").read_text())
    assert m["weeks"]["1"]["rows"] == m["files"]["order_train.csv"]["rows_read"]
    assert m["files"]["order_test.csv"]["boundary_rows_dropped"] > 0


def test_week2_rows_equal_test_file_minus_boundary():
    m = json.loads((PROC / "orders_manifest.json").read_text())
    f = m["files"]["order_test.csv"]
    assert m["weeks"]["2"]["rows"] == f["rows_read"] - f["boundary_rows_dropped"]


def test_all_orders_accounted_for():
    """Nothing silently vanished: every raw row is in a week or was a
    deliberately-dropped boundary row."""
    m = json.loads((PROC / "orders_manifest.json").read_text())
    read = sum(f["rows_read"] for f in m["files"].values())
    kept = sum(w["rows"] for w in m["weeks"].values())
    dropped = sum(f["boundary_rows_dropped"] for f in m["files"].values())
    assert read == kept + dropped


# --- the evaluation split itself -------------------------------------------
# The two order files are separately re-indexed, so the evaluation lives
# inside week 2 and has to earn both guarantees there. These check it does.


def _split():
    from eval.split import make_split
    return make_split(CFG)


def test_heldout_accounts_are_absent_from_training_and_calibration():
    s = _split()
    assert np.intersect1d(s.train, s.test).size == 0
    assert np.intersect1d(s.val, s.test).size == 0
    assert np.intersect1d(s.train_pool, s.test).size == 0


def test_train_and_val_are_disjoint():
    s = _split()
    assert np.intersect1d(s.train, s.val).size == 0


def test_heldout_is_stratified():
    """Held-out and training pools must share a fraud rate, or precision
    figures are not comparable between them."""
    s = _split()
    assert s.y(s.test).mean() == pytest.approx(s.y(s.train_pool).mean(), abs=0.005)


def test_split_covers_every_labelled_account_exactly_once():
    s = _split()
    union = np.concatenate([s.train, s.val, s.test])
    assert np.unique(union).size == union.size, "an account appears in two groups"
    labelled = np.flatnonzero(s.labels != -1)
    assert np.array_equal(np.sort(union), np.sort(labelled))


def test_split_is_deterministic_under_the_configured_seed():
    a, b = _split(), _split()
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.train, b.train)


@pytest.mark.skipif(not (PROC / "edges_week2_late_manifest.json").exists(),
                    reason="run `make windows` first")
def test_scoring_window_is_strictly_after_the_training_window():
    """The whole point of the two windows: no training feature may see an
    order that had not happened yet at decision time."""
    early = json.loads((PROC / "edges_week2_early_manifest.json").read_text())
    late = json.loads((PROC / "edges_week2_late_manifest.json").read_text())
    assert early["days"][1] < late["days"][0]


@pytest.mark.skipif(not (PROC / "features_week2_late.parquet").exists(),
                    reason="run `make windows` first")
def test_window_features_differ_from_each_other():
    """A guard against silently loading the same table twice, which would
    make the forward-in-time claim vacuous."""
    a = pq.read_table(PROC / "features_week2_early.parquet", columns=["n_orders"])
    b = pq.read_table(PROC / "features_week2_late.parquet", columns=["n_orders"])
    assert not np.array_equal(a["n_orders"].to_numpy(), b["n_orders"].to_numpy())
