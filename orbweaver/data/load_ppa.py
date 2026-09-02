"""Parse the raw PPA release into canonical parquet.

The raw files are described in `docs/data.md`, which I wrote from the files
themselves rather than from the paper. Three properties of the raw data drive
this module:

1. **CRLF line endings.** The trailing `r8` column parses as `"\\r"` rather
   than empty unless the reader strips it. pyarrow's CSV reader handles this
   correctly; a naive splitter does not. See FAILURES.md, 2 September.
2. **`order_time` is year-shifted to 1000** and has day resolution.
   `pandas.to_datetime` raises `OutOfBoundsDatetime` on it, because
   `datetime64[ns]` only spans 1677-2262. I parse with `datetime.date`
   and store an integer day ordinal instead, so nothing downstream needs a
   datetime type at all.
3. **`r2`, `r4` and `r5` are entirely empty** in both order files. They are
   dropped from the canonical form, and `assert_empty_relations` proves it
   on every load rather than assuming it.

The temporal split is applied **here, on `order_time`** - not on which file a
row came from. `order_test.csv` contains 75 orders dated `1000-05-20`, a
week-1 day; they are dropped rather than reassigned, so no row from the test
file can reach a training artefact by any path.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

ORDER_COLUMNS = ["order_time", "sku_id", "id"] + [f"r{i}" for i in range(1, 9)]
EMPTY_RELATIONS = ["r2", "r4", "r5"]


def parse_day(s: str) -> date:
    """Parse a PPA `order_time`. Year 1000 is out of range for datetime64[ns]."""
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _read_orders_csv(path: Path) -> pa.Table:
    """Read one raw order file. Relation ids arrive as '123.0' floats."""
    convert = pacsv.ConvertOptions(
        column_types={
            "order_time": pa.string(),
            "sku_id": pa.int64(),
            "id": pa.int64(),
            **{f"r{i}": pa.float64() for i in range(1, 9)},
        }
    )
    return pacsv.read_csv(path, convert_options=convert)


def assert_empty_relations(table: pa.Table, path: Path) -> None:
    """r2, r4 and r5 carry no values at all. Prove it on every load."""
    for rel in EMPTY_RELATIONS:
        n = table.num_rows - table[rel].null_count
        if n:
            raise ValueError(
                f"{path.name}: relation {rel} was expected to be entirely empty "
                f"(see docs/data.md) but has {n:,} values. "
                "The release has changed; re-run scripts/inspect_ppa.py."
            )


def load_orders(cfg: Config | None = None, *, force: bool = False) -> dict:
    """Raw order CSVs -> `orders_week1.parquet`, `orders_week2.parquet`.

    Returns a manifest dict, also written to `orders_manifest.json`.
    """
    cfg = cfg or load_config()
    raw = cfg.abs_path(cfg.paths.raw)
    out = cfg.abs_path(cfg.paths.processed)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "orders_manifest.json"

    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    week1_last = parse_day(cfg.data.week1_last_day)
    week2_first = parse_day(cfg.data.week2_first_day)
    if (week2_first - week1_last).days != 1:
        raise ValueError("week2_first_day must be the day after week1_last_day")

    keep_rel = cfg.data.buildable_relations
    frames: dict[int, list[pa.Table]] = {1: [], 2: []}
    per_file: dict[str, dict] = {}

    for name in ("order_train.csv", "order_test.csv"):
        path = raw / name
        table = _read_orders_csv(path)
        assert_empty_relations(table, path)

        # There are only ~9 distinct dates per file. Dictionary-encode so the
        # date is parsed once per distinct value, not once per row -
        # to_pylist() on 22M strings would materialise 22M Python objects.
        encoded = pc.dictionary_encode(table["order_time"].combine_chunks())
        vocab = encoded.dictionary.to_pylist()
        codes = encoded.indices.to_numpy(zero_copy_only=False)
        lut = np.array([parse_day(v).toordinal() for v in vocab], dtype=np.int64)
        day_arr = lut[codes]
        uniq = sorted(vocab)

        week1_mask = day_arr <= week1_last.toordinal()
        dropped = int(week1_mask.sum()) if name == "order_test.csv" else 0

        base = pa.table(
            {
                "user_id": table["id"].cast(pa.int32()),
                "sku_id": table["sku_id"].cast(pa.int32()),
                "day_ordinal": pa.array(day_arr, type=pa.int32()),
                **{r: table[r].cast(pa.int64()) for r in keep_rel},
            }
        )

        if name == "order_train.csv":
            # Every row is <= week1_last by construction; assert it.
            if not week1_mask.all():
                raise ValueError("order_train.csv contains rows after week1_last_day")
            frames[1].append(base)
            kept, week = base.num_rows, 1
        else:
            keep = pa.array(~week1_mask)
            base = base.filter(keep)
            frames[2].append(base)
            kept, week = base.num_rows, 2

        per_file[name] = {
            "rows_read": table.num_rows,
            "rows_kept": kept,
            "assigned_week": week,
            "boundary_rows_dropped": dropped,
            "date_min": uniq[0],
            "date_max": uniq[-1],
        }
        del table

    manifest = {"config_seed": cfg.seed, "files": per_file, "weeks": {}}
    for week, parts in frames.items():
        tbl = pa.concat_tables(parts)
        dest = out / f"orders_week{week}.parquet"
        pq.write_table(tbl, dest, compression="zstd")
        days = np.unique(tbl["day_ordinal"].to_numpy())
        manifest["weeks"][str(week)] = {
            "path": str(dest.relative_to(cfg.abs_path("."))),
            "rows": tbl.num_rows,
            "distinct_users": int(np.unique(tbl["user_id"].to_numpy()).size),
            "day_ordinal_min": int(days.min()),
            "day_ordinal_max": int(days.max()),
            "n_days": int(days.size),
            "dates": [date.fromordinal(int(d)).isoformat() for d in days],
            "bytes": dest.stat().st_size,
            "relation_non_null": {
                r: int(tbl.num_rows - tbl[r].null_count) for r in keep_rel
            },
        }
        del tbl

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_nodes(cfg: Config | None = None, *, force: bool = False) -> Path:
    """`node.csv` -> `nodes.parquet` (id, label). The vec_* columns are all
    1.0 in every row and carry no information (docs/data.md), so they are
    dropped rather than propagated."""
    cfg = cfg or load_config()
    out = cfg.abs_path(cfg.paths.processed)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "nodes.parquet"
    if dest.exists() and not force:
        return dest

    src = cfg.abs_path(cfg.paths.raw) / "node.csv"
    table = pacsv.read_csv(
        src,
        convert_options=pacsv.ConvertOptions(include_columns=["id", "label"]),
    )
    ids = table["id"].to_numpy()
    if not (ids == np.arange(len(ids), dtype=ids.dtype)).all():
        raise ValueError("node.csv ids are no longer contiguous 0..n-1")
    pq.write_table(
        pa.table({"user_id": table["id"].cast(pa.int32()),
                  "label": table["label"].cast(pa.int8())}),
        dest,
        compression="zstd",
    )
    return dest


def load_author_edges(cfg: Config | None = None, *, force: bool = False) -> Path:
    """`edge.csv` -> `edges_authors.parquet` (View B).

    Week 2 only, all 8 relations, the authors' own weights. Note the columns
    are `r1_score`..`r8_score`; the dataset readme's `r1`..`r8` is wrong.
    """
    cfg = cfg or load_config()
    out = cfg.abs_path(cfg.paths.processed)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "edges_authors.parquet"
    if dest.exists() and not force:
        return dest

    src = cfg.abs_path(cfg.paths.raw) / "edge.csv"
    score_cols = [f"r{i}_score" for i in range(1, 9)]
    table = pacsv.read_csv(
        src,
        convert_options=pacsv.ConvertOptions(
            column_types={"src": pa.int32(), "dst": pa.int32(),
                          **{c: pa.float32() for c in score_cols}}
        ),
    )
    pq.write_table(table, dest, compression="zstd")
    return dest


def main() -> None:
    cfg = load_config()
    print("nodes        ->", load_nodes(cfg, force=True))
    print("author edges ->", load_author_edges(cfg, force=True))
    m = load_orders(cfg, force=True)
    for week, w in m["weeks"].items():
        print(f"week {week}: {w['rows']:>12,} orders  {w['distinct_users']:>10,} users  "
              f"{w['dates'][0]} -> {w['dates'][-1]}")
    dropped = sum(f["boundary_rows_dropped"] for f in m["files"].values())
    print(f"boundary orders dropped: {dropped}")


if __name__ == "__main__":
    main()
