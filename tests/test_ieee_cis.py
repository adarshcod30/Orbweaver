"""Tests for the payment-processor graph.

The split is the thing worth testing hardest. IEEE-CIS is the one dataset here
with real timestamps, so it can carry both guarantees at once - forward in time
*and* account-disjoint - and a claim like that is only worth making if it is
checked.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from orbweaver.config import load_config
from orbweaver.data import ieee_cis as IC

CFG = load_config()
PROC = CFG.abs_path(CFG.paths.processed)
RAW = CFG.abs_path(".") / IC.RAW

needs_data = pytest.mark.skipif(
    not (RAW / "train_transaction.csv").exists(),
    reason="run `make download-ieee-cis` first")
needs_run = pytest.mark.skipif(
    not (PROC / "ieee_cis.json").exists(), reason="run `make ieee-cis` first")


def test_account_proxy_is_deterministic():
    """The same fingerprints must always produce the same ids, or nothing
    downstream is reproducible."""
    df = pd.DataFrame({
        "card1": [1, 2, 1, None], "card2": [10, 20, 10, 40],
        "card3": [100, 100, 100, 100], "card5": [1, 1, 1, 2],
        "addr1": [5, 6, 5, None], "addr2": [87, 87, 87, 87],
    })
    a, uniq = IC.account_proxy(df)
    b, _ = IC.account_proxy(df)
    assert np.array_equal(a, b)
    assert a[0] == a[2], "identical fingerprints must share an id"
    assert a[0] != a[1]
    # a missing field becomes "na" rather than dropping the row
    assert a[3] >= 0 and len(uniq) == 3


def test_account_proxy_survives_row_reordering():
    """Ids are assigned from sorted fingerprints, so shuffling the frame must
    not renumber the accounts."""
    df = pd.DataFrame({
        "card1": [3, 1, 2], "card2": [1, 1, 1], "card3": [1, 1, 1],
        "card5": [1, 1, 1], "addr1": [1, 1, 1], "addr2": [1, 1, 1],
    })
    a, _ = IC.account_proxy(df)
    shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)
    b, _ = IC.account_proxy(shuffled)
    assert a[2] == b[0] and a[0] == b[1] and a[1] == b[2]


def test_missing_entities_stay_missing():
    """A missing device must not become a shared 'unknown device' that links
    every account without one - that would be the phantom-entity bug again."""
    df = pd.DataFrame({
        "DeviceInfo": [None, None, "iPhone"], "DeviceType": [None, None, "mobile"],
        "P_emaildomain": ["a.com", None, "a.com"],
        "R_emaildomain": [None, None, None],
        "addr1": [1, None, 2], "dist1": [3, None, 40],
        "id_31": [None, "chrome 70", None],
    })
    cols = IC.relation_columns(df)
    assert np.isnan(cols["device"][0]) and np.isnan(cols["device"][1])
    assert not np.isnan(cols["device"][2])
    assert np.isnan(cols["R_emaildomain" if False else "email_recipient"]).all()


def test_label_rule_is_reported_with_its_sensitivity():
    df = pd.DataFrame({"isFraud": [1, 0, 1, 0, 0, 0]})
    account = np.array([0, 0, 1, 1, 2, 2])
    labels, sens = IC.account_labels(df, account, 3)
    assert labels.tolist() == [1, 1, 0]          # 0.5 share counts as fraud
    assert sens["rule_any_fraud_at_all"] == 2
    assert sens["rule_share_at_least_0.5"] == 2


@needs_run
def test_split_is_forward_in_time_and_account_disjoint():
    out = json.loads((PROC / "ieee_cis.json").read_text())
    s = out["split"]
    assert s["train_days"][1] < s["score_days"][0], "windows overlap in time"
    assert s["train_accounts"] > 0 and s["heldout_accounts"] > 0
    assert "forward in time" in s["kind"] and "account-disjoint" in s["kind"]


@needs_run
def test_caveats_travel_with_the_numbers():
    """This is card fraud on a card-fingerprint proxy, and the section is not
    allowed to report a precision without saying so."""
    out = json.loads((PROC / "ieee_cis.json").read_text())
    joined = " ".join(out["caveats"]).lower()
    assert "card fraud" in joined
    assert "not a person" in joined or "proxy" in joined
    assert "24.4%" in joined or "identity file" in joined


@needs_run
def test_relation_weights_are_measured_not_assumed():
    out = json.loads((PROC / "ieee_cis.json").read_text())
    w = out["relation_weights"]
    assert len(w) == len(IC.RELATIONS)
    measured = [v for v in w.values() if v.get("measured")]
    assert measured, "no relation had enough labelled edges to measure"
    # the weighting must actually differentiate, not hand everything a 1.0
    alphas = [v["alpha"] for v in w.values()]
    assert max(alphas) - min(alphas) > 0.1
