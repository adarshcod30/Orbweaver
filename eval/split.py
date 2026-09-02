"""The split, and why it is the shape it is.

The plan was week 1 trains, week 2 tests. That is not buildable on this
release: the two order files are separately re-indexed from zero, so week-1
account 12,345 and week-2 account 12,345 are different people and no key joins
them (docs/data.md, finding E; FAILURES.md has how I found out). Labels live
in the week-2 id space, so week-1 features for a labelled account do not
exist.

Week 2 is the only slice where orders, graph and labels agree on an id space,
so the evaluation lives there and has to earn both guarantees inside it.

**Account-disjoint.** Labelled accounts are split, stratified by label, into a
training pool and a held-out set. Held-out accounts appear in neither training
nor calibration, so the model has never seen them. Without this the model
could score well by memorising which accounts are fraudulent instead of
learning what fraud looks like.

**Forward in time.** Training features and the graph behind them come from
`1000-05-21…05-24`; held-out accounts are scored on features and a graph built
from `1000-05-25…05-28` (see `orbweaver.data.windows`). Nothing a training
feature saw had happened after the point of decision.

Rings can straddle the account boundary. That is acceptable because no feature
reads a neighbour's label (`orbweaver.features.node_features`), so a held-out
account is scored on its own behaviour and structure, never on who its
labelled neighbours are. The graph itself is shared across the split, which
makes this transductive - the same situation a production system is in, where
you hold today's whole graph and must score one account inside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config

FRAUD, NORMAL, UNLABELLED = 1, 0, -1


@dataclass(frozen=True)
class Split:
    """Index sets over the week-2 user-id space."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray            # held out of training and calibration
    train_pool: np.ndarray      # train + val, for the seen/unseen comparison
    labels: np.ndarray
    n_users_week2: int

    def y(self, idx: np.ndarray) -> np.ndarray:
        lab = self.labels[idx]
        if (lab == UNLABELLED).any():
            raise ValueError("index set contains unlabelled accounts")
        return (lab == FRAUD).astype(np.int8)

    def summary(self) -> dict:
        def counts(idx):
            if idx.size == 0:
                return {"n": 0, "fraud": 0, "normal": 0, "fraud_rate": 0.0}
            y = self.y(idx)
            return {"n": int(idx.size), "fraud": int(y.sum()),
                    "normal": int((y == 0).sum()),
                    "fraud_rate": round(float(y.mean()), 6)}
        return {
            "train": counts(self.train),
            "val": counts(self.val),
            "test_heldout": counts(self.test),
            "train_pool_for_comparison": counts(self.train_pool),
            "active_accounts_week2": self.n_users_week2,
            "disjoint": bool(np.intersect1d(self.train_pool, self.test).size == 0),
        }


def active_users(week: int, cfg: Config) -> np.ndarray:
    proc = cfg.abs_path(cfg.paths.processed)
    uid = pq.read_table(proc / f"orders_week{week}.parquet",
                        columns=["user_id"])["user_id"].to_numpy()
    return np.unique(uid)


def make_split(cfg: Config | None = None) -> Split:
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)

    labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy().astype(np.int8)
    w2 = active_users(2, cfg)
    labelled = np.flatnonzero(labels != UNLABELLED)
    pool_all = np.intersect1d(w2, labelled, assume_unique=True)

    rng = np.random.default_rng(cfg.seed)

    # Stratify so the held-out set keeps the same fraud rate; otherwise the
    # precision figures are not comparable between the two groups.
    held_parts, keep_parts = [], []
    for cls in (FRAUD, NORMAL):
        members = pool_all[labels[pool_all] == cls]
        perm = rng.permutation(members.size)
        n_hold = int(round(cfg.scoring.heldout_fraction * members.size))
        held_parts.append(members[perm[:n_hold]])
        keep_parts.append(members[perm[n_hold:]])
    test = np.sort(np.concatenate(held_parts))
    train_pool = np.sort(np.concatenate(keep_parts))

    perm = rng.permutation(train_pool.size)
    n_val = int(round(cfg.scoring.val_fraction * train_pool.size))
    val = np.sort(train_pool[perm[:n_val]])
    train = np.sort(train_pool[perm[n_val:]])

    if np.intersect1d(train, test).size or np.intersect1d(val, test).size:
        raise RuntimeError("held-out accounts leaked into training or calibration")

    return Split(train=train, val=val, test=test, train_pool=train_pool,
                 labels=labels, n_users_week2=int(w2.size))


def main() -> None:
    import json
    print(json.dumps(make_split().summary(), indent=2))


if __name__ == "__main__":
    main()
