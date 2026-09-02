"""Schema tests — the canonical parquet must match what `docs/data.md` says.

These assert the properties the rest of the pipeline relies on. If the OSF
release ever changes, these fail loudly instead of the pipeline producing
quietly wrong numbers.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from orbweaver.config import load_config

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)

# Ground truth measured by scripts/inspect_ppa.py and written up in
# docs/data.md. These are the released figures, not the paper's.
EXPECTED = {
    "nodes": 3_267_961,
    "labels": {1: 68_533, 0: 237_084, -1: 2_962_344},
    "author_edges": 10_012_449,
    "week1_rows": 22_456_547,
    "week2_rows": 21_478_629,
    "week1_users": 3_785_628,
    "week2_users": 3_267_961,
    "boundary_dropped": 75,
}

pytestmark = pytest.mark.skipif(
    not (PROC / "nodes.parquet").exists(),
    reason="run `make data` first",
)


def test_nodes_shape_and_labels():
    t = pq.read_table(PROC / "nodes.parquet")
    assert t.num_rows == EXPECTED["nodes"]
    labels, counts = np.unique(t["label"].to_numpy(), return_counts=True)
    got = dict(zip(labels.tolist(), counts.tolist()))
    assert got == EXPECTED["labels"]


def test_node_ids_are_contiguous_row_indices():
    """The pipeline indexes arrays by user_id directly; this must hold."""
    ids = pq.read_table(PROC / "nodes.parquet")["user_id"].to_numpy()
    assert ids[0] == 0
    assert ids[-1] == len(ids) - 1
    assert np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype))


def test_label_minus_one_is_unlabelled_not_normal():
    """-1 outnumbers both real classes ~10:1. Treating it as 'normal' would
    silently redefine every metric, so its size is asserted explicitly."""
    t = pq.read_table(PROC / "nodes.parquet")
    lab = t["label"].to_numpy()
    assert (lab == -1).sum() > 10 * (lab == 1).sum()
    assert set(np.unique(lab).tolist()) == {-1, 0, 1}


def test_author_edges_are_deduplicated_undirected_pairs():
    t = pq.read_table(PROC / "edges_authors.parquet")
    assert t.num_rows == EXPECTED["author_edges"]
    src, dst = t["src"].to_numpy(), t["dst"].to_numpy()
    assert (src != dst).all(), "edge.csv should contain no self-loops"
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    packed = lo.astype(np.int64) * (EXPECTED["nodes"] + 1) + hi
    assert np.unique(packed).size == t.num_rows, "duplicate undirected pairs"


def test_author_edge_columns_use_score_suffix():
    """The dataset readme calls these r1..r8; the files use r1_score..r8_score."""
    names = pq.read_schema(PROC / "edges_authors.parquet").names
    assert [f"r{i}_score" for i in range(1, 9)] == [n for n in names if n.endswith("_score")]


@pytest.mark.parametrize("week,rows,users", [
    (1, EXPECTED["week1_rows"], EXPECTED["week1_users"]),
    (2, EXPECTED["week2_rows"], EXPECTED["week2_users"]),
])
def test_order_weeks_shape(week, rows, users):
    t = pq.read_table(PROC / f"orders_week{week}.parquet")
    assert t.num_rows == rows
    assert np.unique(t["user_id"].to_numpy()).size == users


def test_empty_relations_are_absent_from_canonical_form():
    """r2, r4 and r5 are entirely empty in the raw orders and must not
    reappear as all-null columns downstream."""
    for week in (1, 2):
        names = pq.read_schema(PROC / f"orders_week{week}.parquet").names
        for rel in ("r2", "r4", "r5"):
            assert rel not in names
        for rel in CFG.data.buildable_relations:
            assert rel in names


def test_r8_is_not_fully_populated():
    """Regression test for the CRLF bug: r8 is the trailing column and parsed
    as '\\r' rather than null, reporting 100.00% present. It is 99.24%.
    See FAILURES.md, 2 September."""
    m = json.loads((PROC / "orders_manifest.json").read_text())
    w1 = m["weeks"]["1"]
    frac = w1["relation_non_null"]["r8"] / w1["rows"]
    assert 0.98 < frac < 0.995, f"r8 present on {frac:.4%} of week-1 orders"


def test_boundary_orders_were_dropped():
    m = json.loads((PROC / "orders_manifest.json").read_text())
    assert m["files"]["order_test.csv"]["boundary_rows_dropped"] == EXPECTED["boundary_dropped"]
