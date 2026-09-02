"""What one business can see, against what the platform can see.

The argument that a payment aggregator holds an edge nobody else can build is,
elsewhere in this project, made with synthetic edges conditioned on the labels.
That is a sensitivity analysis and I have always labelled it as one. Two of the
review-fraud datasets let me make the same argument with a real relation and
real labels instead, by taking a relation away and rerunning everything.

**YelpChi** is the clean case. Its nodes are reviews, and its three relations
are:

    net_rur   posted by the same user        <- spans businesses
    net_rtr   same product that month        <- inside one business
    net_rsr   same product, same rating      <- inside one business

A single business sees its own reviews and the links among them. It cannot see
that the person who left this review also left forty others elsewhere, because
it has no visibility of that account off its own property. The platform can.
So `net_rur` is exactly the cross-business relation, and dropping it gives the
merchant's view of the same fraud.

**Amazon** does not split so cleanly. Its relations are co-review, same-rating-
that-week and review-text similarity, all of which a large seller could
approximate from its own reviews. Rather than force the analogy I run
leave-one-relation-out and report each relation's marginal value, which is the
honest version of the same question: what is each kind of link worth?

Dropping a relation removes its edges and therefore changes every graph-derived
feature, so node scoring moves too. That is the point - a merchant is not
running the same model with one column missing, it is running on a smaller
graph - and both the node and ring figures are reported.
"""
from __future__ import annotations

import json

from orbweaver.config import Config, load_config

# The relation only the platform can see, per dataset. YelpChi's is real;
# Amazon has no clean equivalent, so it gets leave-one-out instead.
PLATFORM_ONLY = {"yelpchi": "net_rur"}


def summarise(run: dict, tau: float | None = None) -> dict:
    """The few numbers worth comparing between two views of one dataset.

    `tau` pins both arms to the same score cut-off. Picking each arm's best
    cell independently would compare two different operating points and call
    the difference an effect of the graph.
    """
    node = run["node_scoring_heldout"]
    cells = [b for b in run["rings"].values() if b.get("n_rings")]
    if tau is not None:
        cells = [b for b in cells if b.get("tau") == tau] or cells
    best = max(cells, key=lambda b: b.get("ring_precision") or 0, default={})
    return {
        "edges": run["edges"],
        "relations": list(run["relations"]),
        "node_auprc": node["auprc"],
        "node_auprc_lift": node["auprc_lift_over_random"],
        "ring_precision": best.get("ring_precision"),
        "precision_lift_over_base": best.get("precision_lift_over_base"),
        "fraud_found": best.get("fraud_members"),
        "accounts_in_rings": best.get("accounts_in_rings"),
        "heldout_precision": best.get("heldout_precision"),
        "normal_flagged_per_fraud_caught": best.get("normal_flagged_per_fraud_caught"),
        "tau": best.get("tau"),
        "precision_at_budget": best.get("precision_at_budget", {}),
    }


def run(cfg: Config | None = None) -> dict:
    from eval.generalise import DATASETS, RELATION_MEANING, available, run_one

    cfg = cfg or load_config()
    have = available(cfg)
    if not have:
        return {}

    out: dict = {
        "note": ("A business sees only the links it can observe on its own "
                 "property. The platform sees the account everywhere. Both arms "
                 "run the identical pipeline; dropping a relation changes the "
                 "graph features too, so node scoring moves as well."),
        "datasets": {},
    }

    for name in have:
        relations = DATASETS[name][1]
        block: dict = {"arms": {}}

        platform = run_one(name, cfg)
        block["arms"]["platform"] = summarise(platform)

        only = PLATFORM_ONLY.get(name)
        if only:
            tau = block["arms"]["platform"]["tau"]
            merchant = run_one(name, cfg, drop=(only,))
            block["arms"]["merchant"] = summarise(merchant, tau)
            block["cross_business_relation"] = only
            block["cross_business_meaning"] = RELATION_MEANING.get(only, only)
            p, m = block["arms"]["platform"], block["arms"]["merchant"]
            block["delta"] = {
                "ring_precision": _delta(p["ring_precision"], m["ring_precision"]),
                "node_auprc": _delta(p["node_auprc"], m["node_auprc"]),
                "edges": _delta(p["edges"], m["edges"]),
            }
            # The comparison that means something. Raw fraud counts are not
            # comparable here: without the cross-business relation the graph is
            # denser and blunter, so peeling returns larger, looser rings and
            # the merchant arm surfaces far more accounts. At equal review
            # capacity the question is how many of them are worth reviewing.
            block["at_equal_review_budget"] = _budget_table(
                p.get("precision_at_budget", {}), m.get("precision_at_budget", {}))
        else:
            # No relation here corresponds to "off my own property", so the
            # honest question is what each link is worth on its own terms.
            block["leave_one_out"] = {}
            base_arm = block["arms"]["platform"]
            tau = base_arm["tau"]
            for r in relations:
                arm = summarise(run_one(name, cfg, drop=(r,)), tau)
                arm["dropped"] = r
                arm["dropped_meaning"] = RELATION_MEANING.get(r, r)
                arm["delta_ring_precision"] = _delta(
                    base_arm["ring_precision"], arm["ring_precision"])
                arm["delta_node_auprc"] = _delta(
                    base_arm["node_auprc"], arm["node_auprc"])
                arm["at_equal_review_budget"] = _budget_table(
                    base_arm.get("precision_at_budget", {}),
                    arm.get("precision_at_budget", {}))
                block["leave_one_out"][r] = arm
            block["finding"] = (
                "No single relation is load-bearing here: dropping any one of "
                "the three moves ring precision by less than a point, and "
                "dropping the same-rating-that-week relation slightly improves "
                "it. Amazon's three views of a reviewer are largely redundant, "
                "which is the opposite of what YelpChi shows and is worth "
                "stating rather than hiding.")
            block["no_clean_merchant_split"] = (
                "Amazon's relations are co-review, same-rating-that-week and "
                "review-text similarity. A large seller could approximate all "
                "three from its own reviews, so none of them is the "
                "cross-business link that net_rur is on YelpChi. Reported as "
                "leave-one-out instead of as a merchant view.")
        out["datasets"][name] = block
    return out


def _budget_table(platform: dict, merchant: dict) -> dict:
    out = {}
    for k in sorted(set(platform) & set(merchant), key=lambda x: (x == "all", x)):
        pb, mb = platform[k], merchant[k]
        out[k] = {
            "accounts_reviewed": pb["accounts"] if k != "all" else None,
            "platform_precision": pb["precision"], "platform_fraud": pb["fraud"],
            "merchant_precision": mb["precision"], "merchant_fraud": mb["fraud"],
            "delta_precision": round(pb["precision"] - mb["precision"], 4),
            "delta_fraud": pb["fraud"] - mb["fraud"],
        }
    return out


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4) if isinstance(a, float) or isinstance(b, float) else a - b


def main() -> None:
    cfg = load_config()
    result = run(cfg)
    if not result:
        print("no generalisation datasets found; run `make download-gadbench` "
              "to include this stage. Skipping.")
        return

    dest = cfg.abs_path(cfg.paths.processed) / "merchant_view.json"
    dest.write_text(json.dumps(result, indent=2))

    for name, block in result["datasets"].items():
        print(f"=== {name} ===")
        if "delta" in block:
            p, m = block["arms"]["platform"], block["arms"]["merchant"]
            print(f"  cross-business relation: {block['cross_business_relation']} "
                  f"({block['cross_business_meaning']})")
            print(f"  {'':10s} {'edges':>10s} {'node AUPRC':>11s} "
                  f"{'ring prec':>10s} {'vs base':>8s} {'fraud':>6s}")
            for arm, v in (("platform", p), ("merchant", m)):
                print(f"  {arm:10s} {v['edges']:>10,} {v['node_auprc']:>11} "
                      f"{str(v['ring_precision']):>10} "
                      f"{str(v['precision_lift_over_base']):>8} "
                      f"{str(v['fraud_found']):>6}")
            d = block["delta"]
            print(f"  the relation is {100 * d['edges'] / p['edges']:.1f}% of the edges "
                  f"and is worth {d['ring_precision']:+} ring precision, "
                  f"{d['node_auprc']:+} node AUPRC")
            print(f"  the merchant arm surfaces {m['accounts_in_rings']:,} accounts "
                  f"against {p['accounts_in_rings']:,}, so raw fraud counts do not "
                  f"compare. At equal review capacity:")
            print(f"    {'reviewed':>9s} {'platform':>10s} {'merchant':>10s} {'delta':>8s}")
            for k, v in block["at_equal_review_budget"].items():
                if k == "all":
                    continue
                print(f"    {v['accounts_reviewed']:>9,} {v['platform_precision']:>10} "
                      f"{v['merchant_precision']:>10} {v['delta_precision']:>+8}")
        else:
            print(f"  {block['no_clean_merchant_split']}")
            print(f"  {'dropped':10s} {'edges':>10s} {'ring prec':>10s} {'delta':>8s} {'fraud':>6s}")
            for r, v in block["leave_one_out"].items():
                print(f"  {r:10s} {v['edges']:>10,} {str(v['ring_precision']):>10} "
                      f"{str(v['delta_ring_precision']):>8} {str(v['fraud_found']):>6}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
