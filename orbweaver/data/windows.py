"""Split week 2 into an earlier and a later window, and build both.

The two order files are separately re-indexed (docs/data.md, finding E), so a
week-1-train / week-2-test split cannot be built - there is no key joining an
account across the boundary. Week 2 is the only slice where orders, graph and
labels share one id space, so the forward-in-time guarantee has to be
recovered inside it.

Week 2 covers eight days, `1000-05-21` to `1000-05-28`. This splits them in
half and builds a graph and a feature table for each window independently:

- `early`  05-21 … 05-24   features and graph for training
- `late`   05-25 … 05-28   features and graph for scoring held-out accounts

Both the order statistics and the graph come from the window, so nothing a
training feature sees had happened after the point of decision. Combined with
the account-disjoint split in `eval.split`, an evaluated account is one the
model has never seen, scored on behaviour that occurred after the behaviour it
was trained on.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.data.build_graph import build_graph
from orbweaver.features.node_features import build_features

EARLY, LATE = "early", "late"


def week2_windows(cfg: Config | None = None) -> dict[str, tuple[int, int]]:
    """Day-ordinal ranges for the two halves of week 2, inclusive."""
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    days = np.unique(pq.read_table(proc / "orders_week2.parquet",
                                   columns=["day_ordinal"])["day_ordinal"].to_numpy())
    half = days.size // 2
    return {EARLY: (int(days[0]), int(days[half - 1])),
            LATE: (int(days[half]), int(days[-1]))}


def build_windows(cfg: Config | None = None, *, force: bool = False) -> dict:
    cfg = cfg or load_config()
    windows = week2_windows(cfg)
    out = {}
    for tag, (lo, hi) in windows.items():
        build_graph(2, cfg, days=(lo, hi), tag=tag, force=force)
        build_features(2, cfg, days=(lo, hi), tag=tag, force=force)
        proc = cfg.abs_path(cfg.paths.processed)
        gm = json.loads((proc / f"edges_week2_{tag}_manifest.json").read_text())
        out[tag] = {"days": [lo, hi], "edges": gm["unique_edges"],
                    "nodes_in_edges": gm["distinct_nodes_in_edges"],
                    "raw_pairs": gm["raw_pairs_before_aggregation"]}
    return out


def main() -> None:
    from datetime import date
    cfg = load_config()
    stats = build_windows(cfg, force=True)
    for tag, s in stats.items():
        lo, hi = s["days"]
        print(f"{tag:6s} {date.fromordinal(lo)} -> {date.fromordinal(hi)}  "
              f"{s['raw_pairs']:>11,} raw pairs -> {s['edges']:>11,} edges over "
              f"{s['nodes_in_edges']:>9,} nodes")


if __name__ == "__main__":
    main()
