"""Turn a set of account ids into a case file a human can act on.

A ring is not a result until someone can read why those accounts are together.
Everything in here is deterministic and derived from the orders - no model is
consulted. An analyst can check any line of it by hand.

For each ring this produces:

- **the shared entities**, with coverage (what share of members touch this
  entity) and how rare that entity is globally. A delivery record shared by
  nine of eleven members is evidence; a coupon type shared by all of them and
  by 97.5% of the platform is not, and the rarity column is what separates
  them.
- **temporal concentration** - the largest share of the ring's orders falling
  on a single day. PPA timestamps have day resolution, so this is a day-level
  statement and never an hours-long one.
- **rupees at stake**, from the members' promotion orders and a stated
  assumption about promotion value.
- **the cost of being wrong**, from members labelled normal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

# An entity has to be shared by at least this share of a ring before it is
# worth showing as evidence.
MIN_COVERAGE = 0.5
# Relation names as an analyst would read them.
RELATION_LABELS = {
    "r1": "order location",
    "r3": "delivery record",
    "r6": "promotion",
    "r7": "coupon type",
    "r8": "sales stimulation",
}


def entity_sizes(week: int, relation: str, cfg: Config,
                 tag: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Global (entity_id, distinct_user_count) for one relation, cached."""
    proc = cfg.abs_path(cfg.paths.processed)
    suffix = f"_{tag}" if tag else ""
    cache = proc / f"entity_sizes_week{week}{suffix}_{relation}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["entities"], z["sizes"]

    t = pq.read_table(proc / f"orders_week{week}.parquet",
                      columns=["user_id", relation])
    ent = t[relation].to_numpy(zero_copy_only=False).astype(np.float64)
    uid = t["user_id"].to_numpy()
    keep = ~np.isnan(ent)
    ent, uid = ent[keep].astype(np.int64), uid[keep]
    order = np.lexsort((uid, ent))
    ent, uid = ent[order], uid[order]
    uniq = np.empty(ent.size, dtype=bool)
    uniq[0] = True
    np.logical_or(ent[1:] != ent[:-1], uid[1:] != uid[:-1], out=uniq[1:])
    ent = ent[uniq]
    entities, sizes = np.unique(ent, return_counts=True)
    np.savez_compressed(cache, entities=entities, sizes=sizes)
    return entities, sizes


def _lookup_sizes(entities: np.ndarray, sizes: np.ndarray,
                  wanted: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(entities, wanted)
    idx = np.clip(idx, 0, entities.size - 1)
    found = entities[idx] == wanted
    return np.where(found, sizes[idx], 0)


def ring_evidence(members: np.ndarray, orders: dict[str, np.ndarray],
                  cfg: Config, size_lookup: dict) -> dict:
    """Shared entities, timing and money for one ring."""
    n = members.size
    member_set = np.zeros(int(orders["user_id"].max()) + 1, dtype=bool)
    member_set[members] = True
    rows = orders["user_id"]
    mine = member_set[rows]

    ev: dict = {"size": int(n), "shared_entities": []}

    for rel in cfg.data.buildable_relations:
        col = orders[rel][mine]
        users = rows[mine]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        e, u = col[ok].astype(np.int64), users[ok]
        # distinct (entity, member) pairs, so one member ordering ten times
        # on an entity counts once
        order = np.lexsort((u, e))
        e, u = e[order], u[order]
        uniq = np.empty(e.size, dtype=bool)
        uniq[0] = True
        np.logical_or(e[1:] != e[:-1], u[1:] != u[:-1], out=uniq[1:])
        e = e[uniq]
        ents, counts = np.unique(e, return_counts=True)
        cov = counts / n
        sel = cov >= MIN_COVERAGE
        if not sel.any():
            continue
        entities, sizes_g = size_lookup[rel]
        glob = _lookup_sizes(entities, sizes_g, ents[sel])
        for entity, c, g in zip(ents[sel], counts[sel], glob):
            ev["shared_entities"].append({
                "relation": rel,
                "relation_label": RELATION_LABELS.get(rel, rel),
                "entity_id": int(entity),
                "members_sharing": int(c),
                "coverage": round(float(c / n), 4),
                "global_users_with_entity": int(g),
                # Rarity is what makes coverage meaningful. Sharing an entity
                # that 3 million accounts share is not evidence of anything.
                "rarity_weight": round(float(1.0 / np.log(cfg.graph.rarity_base + g)), 4)
                if g > 0 else None,
            })

    ev["shared_entities"].sort(
        key=lambda d: (d["coverage"] * (d["rarity_weight"] or 0)), reverse=True)
    ev["shared_entities"] = ev["shared_entities"][:12]

    # Day-level concentration. The data has no finer resolution than this.
    days = orders["day_ordinal"][mine]
    if days.size:
        _, counts = np.unique(days, return_counts=True)
        ev["orders"] = int(days.size)
        ev["busiest_day_share"] = round(float(counts.max() / days.size), 4)
        ev["active_days"] = int(counts.size)
    else:
        ev["orders"] = 0
        ev["busiest_day_share"] = 0.0
        ev["active_days"] = 0

    promo = orders["r6"][mine]
    n_promo = int((~np.isnan(promo)).sum())
    ev["promo_orders"] = n_promo
    ev["rupees_at_stake"] = round(n_promo * cfg.cost.assumed_avg_promo_value_inr, 2)
    ev["rupees_at_stake_basis"] = (
        f"{n_promo} promotion orders x an assumed "
        f"Rs.{cfg.cost.assumed_avg_promo_value_inr:.0f} per promotion "
        f"(PPA ships no monetary amounts; this is an assumption)")
    return ev


def load_ring_orders(week: int, cfg: Config) -> dict[str, np.ndarray]:
    proc = cfg.abs_path(cfg.paths.processed)
    cols = ["user_id", "day_ordinal"] + cfg.data.buildable_relations
    t = pq.read_table(proc / f"orders_week{week}.parquet", columns=cols)
    return {c: t[c].to_numpy(zero_copy_only=False) for c in cols}


def build_size_lookup(week: int, cfg: Config) -> dict:
    return {rel: entity_sizes(week, rel, cfg) for rel in cfg.data.buildable_relations}
