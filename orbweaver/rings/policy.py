"""What to do with the queue, given a budget.

The review queue orders rings by how bad they look. It says nothing about a
night on which one analyst has sixty minutes, and nothing about the cost of
holding a promotion for a group that turns out to be a hostel. Detection and
investigation are different constraints, and the second one is usually the
binding one.

So this turns the queue into a decision. For each ring there are three things
a team can do:

- **review** it, which costs analyst minutes and — assuming the analyst then
  acts correctly — stops the fraud in it without harming anyone legitimate;
- **auto-hold** the members' promotions, which costs no analyst time, stops the
  fraud, and harms every legitimate member by some fraction of their value;
- **ignore** it, which costs and stops nothing.

The review set is chosen by exact 0/1 knapsack against the night's budget, and
every ring that is not reviewed is auto-held if holding it is worth more than
ignoring it. Because reviewing is never worse than holding, the value an item
carries into the knapsack is the *gain from reviewing over the best thing you
could do without an analyst* — otherwise the optimiser would spend minutes on
rings that auto-holding already handles.

The framing is example-dependent cost-sensitive learning: a false positive
costs an administrative amount, a false negative costs the amount at stake, and
the number that matters is savings against doing nothing (Bahnsen, Aouada and
Ottersten, ESWA 2016; Elkan, IJCAI 2001, for the foundations).

**Every rupee here is an assumption.** PPA ships no monetary amounts at all.
Promotion value, customer value, what an analyst's minute is worth and how long
a case takes are all stated constants from `config/default.yaml`. They rank
policies against each other and mean nothing in absolute terms — and the
sweeps exist so a reader can see how much the ranking depends on them.

One honest weakness: the probability a ring is fraudulent is taken as the mean
of its members' calibrated scores. Member scores are calibrated; their mean is
not a calibrated ring-level probability, and the ring-confidence work showed
that ring-level calibration collapses on this data because nearly every
candidate above the cut-off is fraudulent. So `p` is a ranking signal being
used as a probability, and where that matters it is said again.
"""
from __future__ import annotations

import json

import numpy as np

from orbweaver.config import Config, load_config

BUDGETS_MIN = (30, 60, 120, 240)      # analyst minutes available per night
CHURN = (0.05, 0.10, 0.25)            # fraction of a wrongly held customer's value lost
CHURN_HEADLINE = 0.10
REVIEWER_ACCURACY = (1.00, 0.90)      # perfect is an upper bound; 0.9 is the honest one
QUEUE_DEPTH = 200


# ------------------------------------------------------------- the ring --

def ring_economics(members: np.ndarray, scores: np.ndarray, promo_value: np.ndarray,
                   ltv: np.ndarray, labels: np.ndarray, cfg: Config) -> dict:
    """Everything the policy needs about one ring.

    The decision quantities use no labels — they are what a team would have
    tonight. The realised quantities use labels and are only ever read
    afterwards, when scoring what the decision turned out to be worth.
    """
    s = scores[members]
    p = float(np.mean(s))
    minutes = cfg.cost.review_minutes_fixed + cfg.cost.review_minutes_per_member * members.size
    return {
        # known tonight
        "p": p,
        "value_at_stake_inr": float(promo_value[members].sum()),
        "legitimate_value_exposed_inr": float(((1.0 - s) * ltv[members]).sum()),
        "review_minutes": max(1, int(round(minutes))),
        "size": int(members.size),
        # known only afterwards
        "realised_fraud_value_inr": float(promo_value[members][labels[members] == 1].sum()),
        "realised_legitimate_value_inr": float(ltv[members][labels[members] == 0].sum()),
    }


def expected_values(r: dict, churn: float) -> tuple[float, float]:
    """Expected net rupees from reviewing, and from auto-holding."""
    ev_review = r["p"] * r["value_at_stake_inr"]
    ev_hold = ev_review - (1.0 - r["p"]) * r["legitimate_value_exposed_inr"] * churn
    return ev_review, ev_hold


# ---------------------------------------------------------- the knapsack --

def knapsack(values: list[float], weights: list[int], capacity: int) -> list[int]:
    """Exact 0/1 knapsack. Returns the chosen indices.

    Minutes are integers and there are at most a couple of hundred rings, so
    the table is small and there is no reason to approximate. Ties go to the
    lower index, which keeps the choice deterministic.
    """
    n = len(values)
    if n == 0 or capacity <= 0:
        return []
    table = np.zeros((n + 1, capacity + 1), dtype=np.float64)
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        table[i] = table[i - 1]
        if w <= capacity and v > 0:
            better = table[i - 1, :-w or None]
            cand = np.concatenate([np.full(w, -np.inf), better + v])
            table[i] = np.maximum(table[i - 1], cand)
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if table[i, c] > table[i - 1, c] + 1e-12:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return sorted(chosen)


# ------------------------------------------------------------ policies --

def _outcome(rings: list[dict], reviewed: set[int], held: set[int], *,
             churn: float, accuracy: float, cfg: Config) -> dict:
    """What a set of decisions turned out to be worth, scored on the labels.

    A reviewed ring: the analyst is right with probability `accuracy`, and when
    right stops the fraud in it and releases everyone legitimate. When wrong,
    they release the fraud and hold the legitimate members, so the mistake
    costs both ways. An auto-held ring stops all of its fraud and harms every
    legitimate member by `churn` of their value.
    """
    stopped = harmed = minutes = 0.0
    for i, r in enumerate(rings):
        if i in reviewed:
            minutes += r["review_minutes"]
            stopped += accuracy * r["realised_fraud_value_inr"]
            harmed += (1.0 - accuracy) * r["realised_legitimate_value_inr"] * churn
        elif i in held:
            stopped += r["realised_fraud_value_inr"]
            harmed += r["realised_legitimate_value_inr"] * churn
    cost_of_time = minutes * cfg.cost.assumed_reviewer_cost_per_minute_inr
    return {
        "rings_reviewed": len(reviewed), "rings_auto_held": len(held),
        "minutes_used": round(minutes, 1),
        "fraud_value_stopped_inr": round(stopped, 2),
        "legitimate_value_harmed_inr": round(harmed, 2),
        "analyst_cost_inr": round(cost_of_time, 2),
        "net_inr": round(stopped - harmed - cost_of_time, 2),
    }


def plan_capacity_aware(rings: list[dict], budget: int, churn: float) -> tuple[set[int], set[int]]:
    """Knapsack the review set, then auto-hold whatever is worth holding."""
    gains, weights = [], []
    for r in rings:
        ev_review, ev_hold = expected_values(r, churn)
        gains.append(max(0.0, ev_review - max(ev_hold, 0.0)))
        weights.append(r["review_minutes"])
    reviewed = set(knapsack(gains, weights, budget))
    held = {i for i, r in enumerate(rings)
            if i not in reviewed and expected_values(r, churn)[1] > 0}
    return reviewed, held


def plan_density_order(rings: list[dict], budget: int) -> tuple[set[int], set[int]]:
    """Work down the queue in density order until the minutes run out."""
    order = sorted(range(len(rings)), key=lambda i: (-rings[i]["density"], i))
    reviewed, spent = set(), 0
    for i in order:
        if spent + rings[i]["review_minutes"] <= budget:
            reviewed.add(i)
            spent += rings[i]["review_minutes"]
    return reviewed, set()


def plan_hold_everything(rings: list[dict]) -> tuple[set[int], set[int]]:
    return set(), set(range(len(rings)))


def plan_do_nothing(rings: list[dict]) -> tuple[set[int], set[int]]:
    return set(), set()


def savings(outcome: dict, total_fraud_value: float) -> float | None:
    """Bahnsen's savings: what the policy costs against doing nothing.

    Doing nothing costs all of the fraud. A policy costs the fraud it failed
    to stop, plus the legitimate value it harmed, plus the analyst time it
    spent. Savings is the fraction of the do-nothing cost avoided, so 0 means
    no better than doing nothing and 1 means every rupee saved for free.
    """
    if total_fraud_value <= 0:
        return None
    cost = ((total_fraud_value - outcome["fraud_value_stopped_inr"])
            + outcome["legitimate_value_harmed_inr"] + outcome["analyst_cost_inr"])
    return round(1.0 - cost / total_fraud_value, 4)


def evaluate_night(rings: list[dict], cfg: Config, *, churn: float = CHURN_HEADLINE,
                   accuracy: float = 1.0, budgets=BUDGETS_MIN) -> dict:
    """Every policy at every budget, for one night's queue."""
    total_fraud = sum(r["realised_fraud_value_inr"] for r in rings)
    rows = {}
    for b in budgets:
        by_policy = {}
        for name, plan in (
            ("capacity-aware", lambda: plan_capacity_aware(rings, b, churn)),
            ("density order until the budget is spent", lambda: plan_density_order(rings, b)),
            ("auto-hold everything", lambda: plan_hold_everything(rings)),
            ("do nothing", lambda: plan_do_nothing(rings)),
        ):
            rev, held = plan()
            out = _outcome(rings, rev, held, churn=churn, accuracy=accuracy, cfg=cfg)
            out["savings"] = savings(out, total_fraud)
            by_policy[name] = out
        rows[str(b)] = by_policy
    return {"budgets": rows, "total_fraud_value_inr": round(total_fraud, 2),
            "rings_in_queue": len(rings)}


# --------------------------------------------------------------- the run --

def load_inputs(cfg: Config):
    """Per-account score, promotion value in the scoring window, and value."""
    import pyarrow.parquet as pq

    from eval.metrics import ltv_proxy
    from eval.split import make_split
    from orbweaver.data.windows import EARLY, LATE, week2_windows

    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    labels = split.labels
    n = labels.size

    scores = np.zeros(n)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

    lo, hi = week2_windows(cfg)[LATE]
    o = pq.read_table(proc / "orders_week2.parquet",
                      columns=["user_id", "day_ordinal", "r6"])
    uid = o["user_id"].to_numpy()
    day = o["day_ordinal"].to_numpy()
    promo = o["r6"].to_numpy(zero_copy_only=False)
    m = ~np.isnan(promo) & (day >= lo) & (day <= hi) & (uid < n)
    promo_value = np.bincount(uid[m], minlength=n).astype(np.float64) \
        * cfg.cost.assumed_avg_promo_value_inr

    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    fu = f["user_id"].to_numpy(); keep = fu < n
    orders_n[fu[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)
    return labels, scores, promo_value, ltv, split


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    labels, scores, promo_value, ltv, split = load_inputs(cfg)

    src = proc / "anchored.json"
    if not src.exists():
        raise SystemExit("run `make anchored` first — the policy runs on its nightly queues")
    an = json.loads(src.read_text())

    def build(queue: list[dict]) -> list[dict]:
        out = []
        for q in queue[:QUEUE_DEPTH]:
            m = np.asarray(q["members"], dtype=np.int64)
            e = ring_economics(m, scores, promo_value, ltv, labels, cfg)
            e.update({"case_id": q.get("case_id"), "event": q.get("event"),
                      "first_seen_night": q.get("first_seen_night"),
                      "density": q.get("density"), "anchor": q.get("anchor")})
            out.append(e)
        return out

    nights = [{"night": nd["night"], "rings": build(nd.get("queue", []))} for nd in an["nights"]]
    final = nights[-1]["rings"]
    print(f"queue on the final night: {len(final)} rings, "
          f"₹{sum(r['value_at_stake_inr'] for r in final):,.0f} at stake, "
          f"₹{sum(r['realised_fraud_value_inr'] for r in final):,.0f} of it on known fraud")

    # ---- the headline night, every policy at every budget ----------------
    headline = evaluate_night(final, cfg, churn=CHURN_HEADLINE, accuracy=1.0)
    print(f"\n{'budget':>7s} {'policy':<40s} {'reviewed':>9s} {'held':>6s} "
          f"{'minutes':>8s} {'fraud Rs stopped':>17s} {'good Rs harmed':>15s} {'savings':>8s}")
    for b, by in headline["budgets"].items():
        for name, o in by.items():
            print(f"{b:>7} {name:<40s} {o['rings_reviewed']:>9} {o['rings_auto_held']:>6} "
                  f"{o['minutes_used']:>8.0f} {o['fraud_value_stopped_inr']:>17,.0f} "
                  f"{o['legitimate_value_harmed_inr']:>15,.0f} {o['savings']:>8}")

    # ---- reviewer accuracy ----------------------------------------------
    accuracy = {f"{a:.2f}": evaluate_night(final, cfg, churn=CHURN_HEADLINE, accuracy=a)
                for a in REVIEWER_ACCURACY}

    # ---- churn sweep, which should only move auto-hold decisions ---------
    churn_sweep = {}
    for c in CHURN:
        ev = evaluate_night(final, cfg, churn=c, accuracy=1.0)
        rev, held = plan_capacity_aware(final, 60, c)
        churn_sweep[f"{c:.2f}"] = {
            "at_60_minutes": ev["budgets"]["60"]["capacity-aware"],
            "rings_reviewed": len(rev), "rings_auto_held": len(held),
        }
    print(f"\n{'churn':>7s} {'reviewed@60':>12s} {'auto-held':>10s} "
          f"{'fraud Rs stopped':>17s} {'good Rs harmed':>15s}")
    for c, blk in churn_sweep.items():
        o = blk["at_60_minutes"]
        print(f"{c:>7} {blk['rings_reviewed']:>12} {blk['rings_auto_held']:>10} "
              f"{o['fraud_value_stopped_inr']:>17,.0f} {o['legitimate_value_harmed_inr']:>15,.0f}")

    # ---- night by night: one analyst, two hours -------------------------
    nightly, running = [], {"stopped": 0.0, "harmed": 0.0, "minutes": 0.0}
    for nd in nights:
        if not nd["rings"]:
            nightly.append({"night": nd["night"], "rings_in_queue": 0})
            continue
        rev, held = plan_capacity_aware(nd["rings"], 120, CHURN_HEADLINE)
        out = _outcome(nd["rings"], rev, held, churn=CHURN_HEADLINE, accuracy=1.0, cfg=cfg)
        out["savings"] = savings(out, sum(r["realised_fraud_value_inr"] for r in nd["rings"]))
        running["stopped"] += out["fraud_value_stopped_inr"]
        running["harmed"] += out["legitimate_value_harmed_inr"]
        running["minutes"] += out["minutes_used"]
        cases = sorted({r["case_id"] for i, r in enumerate(nd["rings"]) if i in rev
                        and r["case_id"] is not None})
        nightly.append({
            "night": nd["night"], "rings_in_queue": len(nd["rings"]),
            "cases_reviewed": cases[:50], "n_cases_reviewed": len(cases), **out,
            "cumulative_fraud_value_stopped_inr": round(running["stopped"], 2),
            "cumulative_legitimate_value_harmed_inr": round(running["harmed"], 2),
        })
    print(f"\none analyst, 120 minutes a night:")
    print(f"{'night':>6s} {'queue':>6s} {'reviewed':>9s} {'held':>6s} {'minutes':>8s} "
          f"{'fraud Rs stopped':>17s} {'cumulative':>13s}")
    for r in nightly:
        if not r.get("rings_in_queue"):
            continue
        print(f"{r['night']:>6} {r['rings_in_queue']:>6} {r['rings_reviewed']:>9} "
              f"{r['rings_auto_held']:>6} {r['minutes_used']:>8.0f} "
              f"{r['fraud_value_stopped_inr']:>17,.0f} "
              f"{r['cumulative_fraud_value_stopped_inr']:>13,.0f}")

    # ---- what a card shows ----------------------------------------------
    rev, held = plan_capacity_aware(final, 120, CHURN_HEADLINE)
    recommendations = []
    for i, r in enumerate(final):
        action = "review" if i in rev else ("auto-hold" if i in held else "ignore")
        ev_review, ev_hold = expected_values(r, CHURN_HEADLINE)
        recommendations.append({
            "case_id": r["case_id"], "anchor": r["anchor"], "size": r["size"],
            "action": action,
            "value_at_stake_inr": round(r["value_at_stake_inr"], 2),
            "legitimate_value_exposed_inr": round(r["legitimate_value_exposed_inr"], 2),
            "review_minutes": r["review_minutes"],
            "expected_net_if_reviewed_inr": round(ev_review, 2),
            "expected_net_if_auto_held_inr": round(ev_hold, 2),
        })

    return {
        "assumptions": {
            "promotion_value_inr": cfg.cost.assumed_avg_promo_value_inr,
            "customer_value_per_week1_order_inr": cfg.cost.assumed_avg_order_value_inr,
            "analyst_cost_per_minute_inr": cfg.cost.assumed_reviewer_cost_per_minute_inr,
            "review_minutes": f"{cfg.cost.review_minutes_fixed} + "
                              f"{cfg.cost.review_minutes_per_member} per member",
            "churn_headline": CHURN_HEADLINE,
            "note": ("PPA ships no monetary amounts, so every rupee here is a stated "
                     "constant. These rank policies against each other and mean nothing "
                     "in absolute terms."),
            "probability_caveat": ("A ring's probability of being fraudulent is the mean of "
                                   "its members' calibrated scores. Member scores are "
                                   "calibrated; their mean is not a calibrated ring-level "
                                   "probability, so this is a ranking signal used as a "
                                   "probability."),
        },
        "queue": {"ordered_by": "mean member score", "depth": QUEUE_DEPTH,
                  "source": "the anchored nightly queues, so a case id follows a ring"},
        "budgets_minutes": list(BUDGETS_MIN),
        "final_night": headline,
        "reviewer_accuracy": accuracy,
        "churn_sweep": churn_sweep,
        "night_by_night_at_120_minutes": nightly,
        "recommendations_at_120_minutes": recommendations,
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "policy.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
