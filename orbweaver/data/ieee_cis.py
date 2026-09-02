"""The same method on transactions from an actual payment processor.

Everything else here runs on food-delivery orders. This runs the pipeline
unchanged on IEEE-CIS — 590,540 card transactions released by Vesta — where
the relations are the ones a processor really holds: the device, the e-mail
domains on both sides, the billing address, the browser. That is the closest
public analogue to what sits inside a payment aggregator.

**Three things have to be said before any number.**

*It is card fraud, not promotion abuse.* The mechanism is different: a stolen
card used across many merchants, rather than many accounts farming one offer.
The graph shape is the same — accounts linked by shared entities, fraud
concentrated in dense pockets — which is why the method transfers at all, but
this is not the same problem.

*The account is a proxy, not a person.* IEEE-CIS has no user id. The standard
approach, and the one used here, is a card fingerprint:
`card1|card2|card3|card5|addr1|addr2`. Two transactions with the same
fingerprint are probably the same card; they are certainly not guaranteed to
be the same person, and one person with two cards is two accounts here.

*Device edges are sparse.* The identity file covers 144,233 of 590,540
transactions — 24.4%. So the device and browser relations exist for a quarter
of the data and are missing, not zero, for the rest.

What this does have that PPA does not: **real timestamps over six months**, so
the split is genuinely forward in time as well as account-disjoint, and the
relations are payment-side rather than platform-side.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from orbweaver.config import Config, load_config

RAW = "data/raw/ieee_cis"

# The card fingerprint. Documented as a proxy everywhere it appears.
ACCOUNT_KEYS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

# Entities two accounts can share. Each is something a processor observes.
RELATIONS = {
    "device": "the same device",
    "email_payer": "the same payer e-mail domain",
    "email_recipient": "the same recipient e-mail domain",
    "address_distance": "the same billing address and distance band",
    "browser": "the same browser build",
}

TX_COLS = (["TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
            "ProductCD", "dist1", "P_emaildomain", "R_emaildomain"]
           + ACCOUNT_KEYS
           + [f"C{i}" for i in range(1, 15)]
           + [f"D{i}" for i in range(1, 16)])
ID_COLS = ["TransactionID", "DeviceInfo", "DeviceType", "id_31"]

# A fingerprint seen this many times or more is a shared or default value, not
# one card. The largest are things like a missing addr2 that everyone shares.
MAX_TX_PER_ACCOUNT = 5000


def load_raw(cfg: Config) -> pd.DataFrame:
    base = cfg.abs_path(".") / RAW
    tx = pd.read_csv(base / "train_transaction.csv", usecols=TX_COLS)
    ident = pd.read_csv(base / "train_identity.csv", usecols=ID_COLS)
    df = tx.merge(ident, on="TransactionID", how="left")
    df["day"] = (df["TransactionDT"] // 86400).astype(np.int32)
    return df


def account_proxy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Card fingerprint -> a dense account id.

    Deterministic: the same rows always give the same ids, because the
    fingerprints are sorted before they are numbered.
    """
    key = df[ACCOUNT_KEYS[0]].astype("string").fillna("na")
    for c in ACCOUNT_KEYS[1:]:
        key = key + "|" + df[c].astype("string").fillna("na")
    codes, uniques = pd.factorize(key, sort=True)
    return codes.astype(np.int64), np.asarray(uniques)


def relation_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    """One entity column per relation, as codes with missing left missing."""
    dist_band = pd.cut(df["dist1"], bins=[-1, 0, 5, 25, 100, 1e9],
                       labels=["0", "1-5", "6-25", "26-100", "100+"])
    out = {
        "device": df["DeviceInfo"].astype("string").fillna("")
                  + "|" + df["DeviceType"].astype("string").fillna(""),
        "email_payer": df["P_emaildomain"].astype("string"),
        "email_recipient": df["R_emaildomain"].astype("string"),
        "address_distance": (df["addr1"].astype("string").fillna("")
                             + "|" + dist_band.astype("string").fillna("")),
        "browser": df["id_31"].astype("string"),
    }
    cleaned = {}
    for name, col in out.items():
        col = col.replace({"": pd.NA, "|": pd.NA, "nan|nan": pd.NA})
        codes, _ = pd.factorize(col, sort=True)
        c = codes.astype(np.float64)
        c[codes < 0] = np.nan          # missing stays missing, never a value
        cleaned[name] = c
    return cleaned


def account_labels(df: pd.DataFrame, account: np.ndarray, n: int,
                   rule: float = 0.5) -> tuple[np.ndarray, dict]:
    """An account is fraudulent when this share of its transactions are.

    The 0.5 rule is the headline; the "any fraud at all" rule is reported
    beside it, because the choice moves the base rate and should be visible.
    """
    fraud = df["isFraud"].to_numpy()
    total = np.bincount(account, minlength=n).astype(np.float64)
    hits = np.bincount(account, weights=fraud, minlength=n)
    share = np.divide(hits, total, out=np.zeros(n), where=total > 0)
    labels = np.where(total > 0, (share >= rule).astype(np.int8), -1).astype(np.int8)
    sensitivity = {
        "rule_share_at_least_0.5": int((share >= 0.5).sum()),
        "rule_any_fraud_at_all": int((hits > 0).sum()),
        "accounts_with_transactions": int((total > 0).sum()),
    }
    return labels, sensitivity


def window_features(df: pd.DataFrame, account: np.ndarray, n: int,
                    day_lo: int, day_hi: int) -> tuple[np.ndarray, list[str]]:
    """Per-account behaviour inside one time window. No V columns."""
    m = (df["day"].to_numpy() >= day_lo) & (df["day"].to_numpy() <= day_hi)
    a = account[m]
    sub = df.loc[m]
    amt = sub["TransactionAmt"].to_numpy()
    day = sub["day"].to_numpy()

    count = np.bincount(a, minlength=n).astype(np.float64)
    safe = np.maximum(count, 1)
    cols, names = [], []

    def add(name, arr):
        cols.append(arr); names.append(name)

    add("n_transactions", count)
    add("amount_sum", np.bincount(a, weights=amt, minlength=n))
    add("amount_mean", np.bincount(a, weights=amt, minlength=n) / safe)
    amax = np.zeros(n); np.maximum.at(amax, a, amt)
    add("amount_max", amax)

    active = np.zeros(n)
    order = np.lexsort((day, a))
    aa, dd = a[order], day[order]
    if aa.size:
        new = np.empty(aa.size, dtype=bool)
        new[0] = True
        np.logical_or(aa[1:] != aa[:-1], dd[1:] != dd[:-1], out=new[1:])
        active = np.bincount(aa[new], minlength=n).astype(np.float64)
    add("active_days", active)
    add("transactions_per_active_day", count / np.maximum(active, 1))

    for col in ("ProductCD", "P_emaildomain", "DeviceInfo", "addr1"):
        codes, _ = pd.factorize(sub[col].astype("string"), sort=True)
        v = codes.astype(np.float64); v[codes < 0] = np.nan
        ok = ~np.isnan(v)
        if ok.any():
            pair = np.stack([a[ok], v[ok]])
            o = np.lexsort(pair[::-1])
            k, w = pair[0][o], pair[1][o]
            new = np.empty(k.size, dtype=bool)
            new[0] = True
            np.logical_or(k[1:] != k[:-1], w[1:] != w[:-1], out=new[1:])
            add(f"distinct_{col}", np.bincount(k[new].astype(np.int64),
                                               minlength=n).astype(np.float64))
        else:
            add(f"distinct_{col}", np.zeros(n))

    for c in [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]:
        v = sub[c].to_numpy(dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0)
        add(f"{c}_mean", np.bincount(a, weights=v, minlength=n) / safe)

    return np.column_stack(cols).astype(np.float32), names
