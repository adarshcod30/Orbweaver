"""A version of the console anyone can run without the 4 GB download.

Everything else here needs the raw dataset: four gigabytes from OSF, an hour
of processing, and 32 GB of free disk. That is a fair price for reproducing
the numbers and an absurd one for looking at the thing, so this ships a small
bundle of already-computed results in the repository and lets the console
serve them.

What is in the bundle, and what is deliberately not:

- the ring case files with their evidence, the anchored timelines and case
  ids, and the review policy's recommendations - all small JSON;
- a sample of accounts with everything `/check` answers with: the score, the
  label, neighbour counts, and ring membership. Every member of every ring the
  console shows is in it - the case files and the anchored rings with their
  case ids - and the rest of the sample is drawn at random so the page can
  answer for ordinary accounts too. The legitimate co-located clusters are
  summarised in the results rather than listed here, because the hostel test
  records what it found about them and not which accounts they contain;
- **not** the graph. Thirty-five million edges do not fit in a bundle meant to
  be cloned, so `/check` serves precomputed neighbour counts rather than
  recomputing a ring around an account. The full console does that live; the
  demo says so rather than pretending.

The console picks this up automatically when `data/processed` has no run in
it, and `ORBWEAVER_DEMO=1` forces it even when a full run is present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from orbweaver.config import Config, load_config

BUNDLE_DIR = "demo"
MAX_BYTES = 40 * 1024 * 1024
SAMPLE_ACCOUNTS = 50_000
RARE_ENTITY_MAX = 10


def bundle_path(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg.abs_path(".") / BUNDLE_DIR


def available(cfg: Config | None = None) -> bool:
    return (bundle_path(cfg) / "accounts.parquet").exists()


def should_use_demo(cfg: Config | None = None) -> bool:
    """Demo mode when asked for, or when there is no full run to serve."""
    cfg = cfg or load_config()
    if os.environ.get("ORBWEAVER_DEMO") == "1":
        return True
    proc = cfg.abs_path(cfg.paths.processed)
    full = (proc / "nodes.parquet").exists() and (proc / "edges_week2_late.parquet").exists()
    return not full and available(cfg)


# ------------------------------------------------------------- building --

def _trim_anchored(an: dict) -> dict:
    return {
        "method": an.get("method"), "operating_point": an.get("operating_point"),
        "design": an.get("design"), "window": an.get("window"),
        "summary": an.get("summary"), "timelines": an.get("timelines"),
        "global_peeling_from_replay": an.get("global_peeling_from_replay"),
        "anchor_sweep_final_night": an.get("anchor_sweep_final_night"),
        "on_demand_latency_ms": an.get("on_demand_latency_ms"),
        "nights": [{k: v for k, v in nd.items() if k != "queue"} for nd in an.get("nights", [])],
        "final_rings": [{k: v for k, v in r.items() if k != "members"}
                        | {"members": r.get("members", [])[:200]}
                        for r in an.get("final_rings", [])],
    }


def _trim_policy(po: dict) -> dict:
    return {k: po.get(k) for k in
            ("assumptions", "queue", "budgets_minutes", "final_night", "reviewer_accuracy",
             "churn_sweep", "night_by_night_at_120_minutes", "recommendations_at_120_minutes")}


def build(cfg: Config | None = None) -> dict:
    """Write the bundle from a completed run."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    dest = bundle_path(cfg)
    dest.mkdir(parents=True, exist_ok=True)

    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()
    n = labels.size
    scores = np.zeros(n, dtype=np.float32)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

    # Neighbour counts, ring membership and rare-link counts, precomputed once
    # so the bundle never needs the edge list.
    e = pq.read_table(proc / "edges_week2_late.parquet",
                      columns=["src", "dst", "min_entity_size"])
    src = e["src"].to_numpy().astype(np.int64)
    dst = e["dst"].to_numpy().astype(np.int64)
    esz = e["min_entity_size"].to_numpy().astype(np.int32)
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    rare = esz <= RARE_ENTITY_MAX
    rare_deg = (np.bincount(src[rare], minlength=n) + np.bincount(dst[rare], minlength=n))

    ring_of = np.full(n, -1, dtype=np.int32)
    rings_payload = {}
    rings_source = None
    for name in ("ring_report_deep.json", "ring_report.json"):
        p = proc / name
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        rings_payload = payload
        rings_source = name
        for i, case in enumerate(payload.get("case_files", [])):
            # The full membership, not the display sample - otherwise every
            # member past the twenty-fifth is told it is in no ring.
            m = np.asarray(case.get("members") or case.get("members_sample", []),
                           dtype=np.int64)
            if m.size:
                ring_of[m] = i
        break

    nbr_in_ring = np.zeros(n, dtype=np.int32)
    both = (ring_of[src] >= 0).astype(np.int32), (ring_of[dst] >= 0).astype(np.int32)
    np.add.at(nbr_in_ring, src, both[1])
    np.add.at(nbr_in_ring, dst, both[0])

    # Everyone interesting, then a random sample of everyone else.
    # Every account the console can show must be answerable, or /check returns
    # "not in the sample" for a ring the page is displaying. That means the
    # case files and the anchored rings, whose members carry the case ids.
    must = [np.flatnonzero(ring_of >= 0)]
    ap = proc / "anchored.json"
    if ap.exists():
        for r in json.loads(ap.read_text()).get("final_rings", []):
            if r.get("members"):
                must.append(np.asarray(r["members"], dtype=np.int64))
    must_have = np.unique(np.concatenate(must)) if must else np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(cfg.seed)
    room = max(0, SAMPLE_ACCOUNTS - must_have.size)
    pool = np.setdiff1d(np.flatnonzero(deg > 0), must_have, assume_unique=False)
    extra = rng.choice(pool, size=min(room, pool.size), replace=False) if room and pool.size else np.empty(0, np.int64)
    keep = np.sort(np.concatenate([must_have, extra]).astype(np.int64))

    table = pa.table({
        "user_id": pa.array(keep, pa.int32()),
        "score": pa.array(scores[keep], pa.float32()),
        "label": pa.array(labels[keep].astype(np.int8), pa.int8()),
        "neighbours": pa.array(deg[keep].astype(np.int32), pa.int32()),
        "neighbours_in_a_ring": pa.array(nbr_in_ring[keep], pa.int32()),
        "rare_links": pa.array(rare_deg[keep].astype(np.int32), pa.int32()),
        "ring": pa.array(ring_of[keep], pa.int32()),
    })
    pq.write_table(table, dest / "accounts.parquet", compression="zstd")

    (dest / "rings.json").write_text(json.dumps(rings_payload, separators=(",", ":")))
    for name, trim in (("anchored.json", _trim_anchored), ("policy.json", _trim_policy)):
        p = proc / name
        if p.exists():
            (dest / name).write_text(json.dumps(trim(json.loads(p.read_text())),
                                                separators=(",", ":"), default=float))
    for name in ("hostel_test.json", "replay.json"):
        p = proc / name
        if p.exists():
            (dest / name).write_text(json.dumps(json.loads(p.read_text()),
                                                separators=(",", ":")))

    op = proc / "offers.json"
    if op.exists():
        offers_full = json.loads(op.read_text())
        # The by-night detail is for the results section, not the console;
        # the bundle keeps only what /offers actually renders.
        trimmed = {k: v for k, v in offers_full.items() if k != "early_warning"}
        if offers_full.get("early_warning"):
            ew = offers_full["early_warning"]
            trimmed["early_warning"] = {k: v for k, v in ew.items() if k != "by_night"}
        (dest / "offers.json").write_text(json.dumps(trimmed, separators=(",", ":"), default=float))

    headline = {}
    hp = proc / "ring_report.json"
    if hp.exists():
        h = json.loads(hp.read_text()).get("best_cell") or {}
        hc = (json.loads(hp.read_text()).get("grid") or {}).get(
            f"tau={h.get('tau')},lambda={h.get('lambda')}", {})
        headline = {"ring_precision": h.get("ring_precision"),
                    "n_rings": hc.get("n_rings")}

    # how deep the pass in this bundle actually goes, so the console can say so
    bc = rings_payload.get("best_cell") or {}
    bcell = (rings_payload.get("grid") or {}).get(
        f"tau={bc.get('tau')},lambda={bc.get('lambda')}", {})
    bundle_pass = {"ring_precision": bc.get("ring_precision"),
                   "n_rings": bcell.get("n_rings"),
                   "case_files_shown": len(rings_payload.get("case_files", []))}

    files = sorted(f for f in dest.iterdir() if f.is_file() and f.name != "meta.json")
    total = sum(f.stat().st_size for f in files)
    anchored_members = int(np.isin(keep, must_have).sum())
    meta = {
        "accounts_in_bundle": int(keep.size),
        "accounts_in_a_ring": int((ring_of[keep] >= 0).sum()),
        "accounts_in_a_ring_or_an_anchored_ring": anchored_members,
        "total_accounts_in_the_run": int(n),
        "files": {f.name: f.stat().st_size for f in files},
        "bytes": int(total),
        "limit_bytes": MAX_BYTES,
        "rings_from": rings_source,
        "headline_pass": headline,
        "bundle_pass": bundle_pass,
        "note": ("Precomputed results, so anyone can look at the console without the "
                 "4 GB dataset. The graph itself is not in here - /check serves stored "
                 "neighbour counts rather than computing a ring live."),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    if total > MAX_BYTES:
        raise SystemExit(f"bundle is {total / 1e6:.1f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit")
    return meta


# -------------------------------------------------------------- serving --

class DemoIndex:
    """`/check` from the bundle. Same answers, minus what needs the graph."""

    def __init__(self, cfg: Config | None = None):
        import pyarrow.parquet as pq

        self.cfg = cfg = cfg or load_config()
        d = bundle_path(cfg)
        t = pq.read_table(d / "accounts.parquet")
        self.ids = t["user_id"].to_numpy().astype(np.int64)
        self.score = t["score"].to_numpy()
        self.label = t["label"].to_numpy()
        self.deg = t["neighbours"].to_numpy()
        self.deg_ring = t["neighbours_in_a_ring"].to_numpy()
        self.rare = t["rare_links"].to_numpy()
        self.ring = t["ring"].to_numpy()
        self.report = json.loads((d / "rings.json").read_text())
        self.cases = self.report.get("case_files", [])
        p = d / "anchored.json"
        self.anchored = json.loads(p.read_text()) if p.exists() else {}
        self.case_of: dict[int, dict] = {}
        for r in self.anchored.get("final_rings", []):
            for m in r.get("members", []):
                self.case_of[int(m)] = r
        self.assumption = (
            f"Rupee figures use an assumed Rs."
            f"{cfg.cost.assumed_avg_promo_value_inr:.0f} per promotion; this "
            "dataset ships no monetary amounts.")
        self.n = int(self.ids.max()) + 1 if self.ids.size else 0

    def check(self, account: int) -> dict:
        i = int(np.searchsorted(self.ids, account))
        if i >= self.ids.size or self.ids[i] != account:
            return {"account": int(account), "known": False,
                    "reason": ("not in the demo sample - it carries "
                               f"{self.ids.size:,} of {self.n:,} accounts, so most "
                               "ordinary accounts are absent")}
        lab = int(self.label[i])
        ring_i = int(self.ring[i])
        out = {
            "account": int(account), "known": True,
            "score": round(float(self.score[i]), 4),
            "label": {1: "known fraud", 0: "known good", -1: "unreviewed"}[lab],
            "neighbours": int(self.deg[i]),
            "neighbours_in_a_ring": int(self.deg_ring[i]),
            "rare_links": int(self.rare[i]),
            "in_a_ring": ring_i >= 0,
            "assumption": self.assumption,
            "demo": True,
        }
        if 0 <= ring_i < len(self.cases):
            c = self.cases[ring_i]
            out["ring"] = {
                "rank": c.get("rank"), "size": c.get("size"), "density": c.get("density"),
                "rupees_at_stake": c.get("rupees_at_stake"),
                "same_day_share": c.get("busiest_day_share"),
                "labels": c.get("labels", {}),
                "shared": [{"what": e.get("relation_label"), "coverage": e.get("coverage"),
                            "accounts_platform_wide": e.get("global_users_with_entity")}
                           for e in c.get("shared_entities", [])[:5]],
            }
        r = self.case_of.get(int(account))
        if r is not None:
            out["anchored_ring"] = {
                "found": True, "stored": True, "size": r.get("size"),
                "density": r.get("density"), "mean_member_score": r.get("mean_member_score"),
                "known_fraud": r.get("fraud"), "known_good": r.get("normal"),
                "case": {"case_id": r.get("case_id"), "event": r.get("event"),
                         "first_seen_night": r.get("first_seen_night"),
                         "of_nights": (self.anchored.get("window") or {}).get("nights")},
            }
        else:
            out["anchored_ring"] = {
                "found": False, "stored": True,
                "reason": ("the demo bundle carries no graph, so a ring cannot be "
                           "computed around an account here; the full console does "
                           "this live in under a millisecond"),
            }
        return out


def main() -> None:
    cfg = load_config()
    meta = build(cfg)
    print(f"bundle: {meta['bytes'] / 1e6:.1f} MB of a {MAX_BYTES / 1e6:.0f} MB limit")
    for name, size in sorted(meta["files"].items(), key=lambda kv: -kv[1]):
        print(f"  {size / 1e6:7.2f} MB  {name}")
    print(f"accounts: {meta['accounts_in_bundle']:,} of {meta['total_accounts_in_the_run']:,}"
          f" ({meta['accounts_in_a_ring_or_an_anchored_ring']:,} of them in a ring the "
          f"console shows)")
    print(f"wrote {bundle_path(cfg)}")


if __name__ == "__main__":
    main()
