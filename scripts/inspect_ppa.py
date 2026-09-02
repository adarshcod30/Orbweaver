"""Measure the raw PPA files and write the results to a JSON of schema facts.

Everything in docs/data.md is read out of this script's output, so no
schema number in the repository is typed in by hand.

For each order file it profiles the date range, the user id range, and -
per relation - the number of distinct entities, the size of the largest,
and the number of user-pairs the relation would induce at several entity
caps. That last column is what the cap in config/default.yaml is chosen
from: uncapped, week 2 comes to 5.46 trillion pairs.

The order CSVs use CRLF line endings; pandas handles this, but naive
field splitting leaves a '\\r' in the trailing r8 column and makes it
look 100% populated. See FAILURES.md, 2 September.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/ppa")
ORDER_RELATIONS = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]
# Candidate caps on the number of distinct users sharing one entity. An
# entity shared by more users than this is too common to be evidence.
CAP_PROBES = [100, 500, 1_000, 10_000, 100_000]


def pair_count(sizes: np.ndarray) -> int:
    """User-pairs induced by entities of the given sizes, uncapped."""
    s = sizes.astype(np.float64)
    return int((s * (s - 1) / 2).sum())


def profile_relation(path: Path, rel: str) -> dict:
    """Entity-size profile for one relation column, loaded one column at a time."""
    df = pd.read_csv(path, usecols=["id", rel], dtype={"id": np.int64, rel: "float64"})
    df = df.dropna(subset=[rel])
    if df.empty:
        return {"relation": rel, "orders_with_relation": 0, "empty": True}

    # distinct (entity, user) pairs — a user ordering twice on one entity
    # must not count as two members of that entity.
    df = df.drop_duplicates()
    sizes = df.groupby(rel).size().to_numpy()

    out = {
        "relation": rel,
        "empty": False,
        "orders_with_relation": int(len(df)),
        "distinct_entities": int(len(sizes)),
        "entity_size_max": int(sizes.max()),
        "entity_size_mean": round(float(sizes.mean()), 3),
        "entity_size_median": int(np.median(sizes)),
        "entity_size_p99": int(np.percentile(sizes, 99)),
        "singleton_entities": int((sizes == 1).sum()),
        "uncapped_user_pairs": pair_count(sizes),
    }
    for cap in CAP_PROBES:
        kept = sizes[sizes <= cap]
        out[f"entities_over_{cap}"] = int((sizes > cap).sum())
        out[f"pairs_if_cap_{cap}"] = pair_count(kept)
    return out


def profile_orders(name: str, path: Path) -> dict:
    print(f"[{name}] profiling {path} ...", flush=True)
    dates = pd.read_csv(path, usecols=["order_time"])["order_time"]
    ids = pd.read_csv(path, usecols=["id"], dtype={"id": np.int64})["id"]
    res = {
        "file": str(path),
        "bytes": path.stat().st_size,
        "rows": int(len(ids)),
        "distinct_users": int(ids.nunique()),
        "user_id_min": int(ids.min()),
        "user_id_max": int(ids.max()),
        "date_counts": {str(k): int(v) for k, v in dates.value_counts().sort_index().items()},
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "relations": {},
    }
    del dates, ids
    for rel in ORDER_RELATIONS:
        res["relations"][rel] = profile_relation(path, rel)
        r = res["relations"][rel]
        if r.get("empty"):
            print(f"  {rel}: EMPTY", flush=True)
        else:
            print(
                f"  {rel}: {r['distinct_entities']:>10,} entities, "
                f"max size {r['entity_size_max']:>9,}, "
                f"uncapped pairs {r['uncapped_user_pairs']:>15,}, "
                f"pairs@500 {r['pairs_if_cap_500']:>12,}",
                flush=True,
            )
    return res


def profile_node(path: Path) -> dict:
    df = pd.read_csv(path)
    vec_cols = [c for c in df.columns if c.startswith("vec_")]
    uniq_vec = df[vec_cols].drop_duplicates()
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "columns": list(df.columns),
        "rows": int(len(df)),
        "id_min": int(df["id"].min()),
        "id_max": int(df["id"].max()),
        "id_is_contiguous": bool(df["id"].is_monotonic_increasing and df["id"].max() == len(df) - 1),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "n_vec_columns": len(vec_cols),
        "distinct_vec_rows": int(len(uniq_vec)),
        "vec_all_ones": bool((df[vec_cols].to_numpy() == 1.0).all()),
    }


def profile_edge(path: Path) -> dict:
    cols = ["src", "dst"] + [f"r{i}_score" for i in range(1, 9)]
    df = pd.read_csv(path, dtype={"src": np.int64, "dst": np.int64})
    score_cols = [c for c in cols if c.endswith("_score")]
    present = (df[score_cols] > 0).to_numpy()
    nodes = pd.unique(np.concatenate([df["src"].to_numpy(), df["dst"].to_numpy()]))
    undirected = pd.MultiIndex.from_arrays(
        [np.minimum(df["src"], df["dst"]), np.maximum(df["src"], df["dst"])]
    )
    res = {
        "file": str(path),
        "bytes": path.stat().st_size,
        "columns": list(df.columns),
        "rows": int(len(df)),
        "distinct_nodes_in_edges": int(len(nodes)),
        "self_loops": int((df["src"] == df["dst"]).sum()),
        "distinct_undirected_pairs": int(undirected.nunique()),
        "relations_per_edge": {
            str(k): int(v) for k, v in pd.Series(present.sum(axis=1)).value_counts().sort_index().items()
        },
        "per_relation": {},
    }
    for i, c in enumerate(score_cols):
        col = df[c].to_numpy()
        nz = col[col > 0]
        res["per_relation"][c] = {
            "nonzero_edges": int(len(nz)),
            "pct_of_edges": round(100 * len(nz) / len(df), 3),
            "min": float(nz.min()) if len(nz) else None,
            "max": float(nz.max()) if len(nz) else None,
            "mean": round(float(nz.mean()), 6) if len(nz) else None,
            "distinct_values": int(pd.Series(nz).nunique()),
        }
    return res


def main() -> int:
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat()}
    out["node_csv"] = profile_node(RAW / "node.csv")
    print("node.csv done", flush=True)
    out["edge_csv"] = profile_edge(RAW / "edge.csv")
    print("edge.csv done", flush=True)
    out["order_train"] = profile_orders("train", RAW / "order_train.csv")
    out["order_test"] = profile_orders("test", RAW / "order_test.csv")

    dest = Path("data/ppa_schema_facts.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
