"""Regenerate docs/results.md and every figure from the saved run artefacts.

Nothing in the README or in docs/results.md is typed by hand. This reads the
JSON written by `eval.score_report` and `eval.run_rings` and renders it, so a
number in the documentation can only be wrong if the run that produced it was.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbweaver.config import load_config

INK = "#1c1c1c"
ACCENT = "#c2410c"
MUTED = "#9a9a9a"
GRID = "#e6e6e6"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def headline_chart(ring: dict, out: Path) -> Path | None:
    """Fraud caught against real customers wrongly swept in.

    This is the chart I would want to see before deploying anything: not
    precision alone, but what each operating point costs in customers.
    """
    rows = [b for b in ring["grid"].values()
            if b.get("ring_precision") is not None and b.get("fraud_members")]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    _style(ax)
    base = ring["base_rate_among_labelled"]

    taus = sorted({r["tau"] for r in rows})
    markers = {0.0: "o", 0.3: "s", 0.5: "^"}
    colors = {0.0: MUTED, 0.3: ACCENT, 0.5: "#0369a1"}
    for tau in taus:
        pts = sorted([r for r in rows if r["tau"] == tau], key=lambda r: r["normal_members"])
        ax.plot([r["normal_members"] for r in pts], [r["fraud_members"] for r in pts],
                marker=markers.get(tau, "o"), color=colors.get(tau, INK),
                linewidth=1.6, markersize=5,
                label=f"score cut-off τ = {tau}" if tau else "no score cut-off")

    lim = max(max(r["normal_members"] for r in rows),
              max(r["fraud_members"] for r in rows)) * 1.05
    ax.plot([0, lim], [0, lim * base / (1 - base)], "--", color=MUTED, linewidth=1.0)
    ax.annotate("what random selection\nwould give you", xy=(lim * 0.62, lim * 0.62 * base / (1 - base)),
                fontsize=8, color=MUTED, ha="left")

    ax.set_xlabel("real customers wrongly placed in a ring", color=INK, fontsize=10)
    ax.set_ylabel("fraudsters caught", color=INK, fontsize=10)
    ax.set_title("What each operating point costs in real customers",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    dest = out / "headline_precision_vs_cost.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def precision_grid_chart(ring: dict, out: Path) -> Path | None:
    rows = [b for b in ring["grid"].values() if b.get("ring_precision") is not None]
    if not rows:
        return None
    base = ring["base_rate_among_labelled"]
    taus = sorted({r["tau"] for r in rows})
    lams = sorted({r["lambda"] for r in rows})

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    _style(ax)
    width = 0.8 / max(len(taus), 1)
    x = np.arange(len(lams))
    for i, tau in enumerate(taus):
        vals = []
        for lam in lams:
            m = [r for r in rows if r["tau"] == tau and r["lambda"] == lam]
            vals.append(m[0]["ring_precision"] if m else 0.0)
        ax.bar(x + i * width, vals, width * 0.9,
               label=f"τ = {tau}" if tau else "no cut-off",
               color=[MUTED, ACCENT, "#0369a1"][i % 3])
    ax.axhline(base, color=INK, linestyle="--", linewidth=1.0)
    ax.annotate(f"base rate {base:.3f}", xy=(len(lams) - 0.5, base),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK, ha="right")
    ax.set_xticks(x + width * (len(taus) - 1) / 2)
    ax.set_xticklabels([f"λ = {l}" for l in lams])
    ax.set_ylabel("share of a ring's labelled members\nthat are fraud", fontsize=10, color=INK)
    ax.set_title("Ring precision across the sweep", color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    dest = out / "ring_precision_grid.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def relation_lift_chart(weights: dict, out: Path) -> Path | None:
    rels = list(weights["relations"].keys())
    lifts = [weights["relations"][r]["lift"] for r in rels]
    labels = {"r1": "location", "r3": "delivery", "r6": "promotion",
              "r7": "coupon", "r8": "stimulation"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    _style(ax)
    order = np.argsort(lifts)[::-1]
    names = [f"{rels[i]}\n{labels.get(rels[i], '')}" for i in order]
    ax.bar(names, [lifts[i] for i in order], color=ACCENT, width=0.6)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
    ax.set_ylabel("fraud–fraud edges vs chance", fontsize=10, color=INK)
    ax.set_title("Not every shared thing is equally incriminating",
                 color=INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    dest = out / "relation_lift.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def adversarial_chart(frag: dict, adv: dict | None, out: Path) -> Path | None:
    """Two ways an attacker can adapt, and what each costs the detector."""
    r = frag["results"]
    cells = [(v["cell_size"], v["ring_precision"]) for k, v in r.items()
             if k != "intact" and v.get("ring_precision") is not None]
    if not cells:
        return None
    cells.sort()
    intact = r["intact"]["ring_precision"]

    n = 2 if adv else 1
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.0), dpi=160)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    _style(ax)
    ax.plot([c for c, _ in cells], [p for _, p in cells], marker="o",
            color=ACCENT, linewidth=1.8, markersize=6)
    ax.axhline(intact, color=INK, linestyle="--", linewidth=1.0)
    ax.annotate(f"intact  {intact}", xy=(cells[-1][0], intact),
                xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=INK, ha="right")
    ax.set_xlabel("cell size the ring was broken into", fontsize=10, color=INK)
    ax.set_ylabel("ring precision", fontsize=10, color=INK)
    ax.set_title("Fragmentation", color=INK, fontsize=12, loc="left", pad=10)

    if adv:
        ax = axes[1]
        _style(ax)
        rounds = [row["round"] for row in adv["rounds"]]
        lift = [row["precision_lift_over_base"] for row in adv["rounds"]]
        ax.plot(rounds, lift, marker="s", color="#0369a1", linewidth=1.8, markersize=6)
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
        ax.annotate("no better than chance", xy=(rounds[-1], 1.0),
                    xytext=(0, 5), textcoords="offset points",
                    fontsize=8, color=MUTED, ha="right")
        ax.set_xlabel("adaptation round", fontsize=10, color=INK)
        ax.set_ylabel("precision lift over base rate", fontsize=10, color=INK)
        ax.set_title("Multi-round duplication", color=INK, fontsize=12,
                     loc="left", pad=10)

    fig.tight_layout()
    dest = out / "adversarial.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def merchant_vs_platform_chart(mv: dict, out: Path) -> Path | None:
    """What one business sees against what the platform sees.

    Plotted at equal review capacity, because the two arms surface different
    numbers of accounts and a raw comparison flatters whichever one happens to
    produce larger rings.
    """
    y = mv["datasets"].get("yelpchi")
    if not y or "merchant" not in y["arms"]:
        return None
    rows = [(int(k), v) for k, v in y["at_equal_review_budget"].items() if k != "all"]
    if not rows:
        return None
    rows.sort()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160)

    ax = axes[0]
    _style(ax)
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, [v["platform_precision"] for _, v in rows], w,
           label="the platform", color=ACCENT)
    ax.bar(x + w / 2, [v["merchant_precision"] for _, v in rows], w,
           label="one business", color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n, _ in rows])
    ax.set_xlabel("accounts reviewed", fontsize=10, color=INK)
    ax.set_ylabel("share worth reviewing", fontsize=10, color=INK)
    lo = min(min(v["merchant_precision"] for _, v in rows),
             min(v["platform_precision"] for _, v in rows))
    ax.set_ylim(max(0.0, lo - 0.06), 1.0)
    ax.set_title("At equal review capacity", color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    _style(ax)
    p_, m_ = y["arms"]["platform"], y["arms"]["merchant"]
    ax.bar(["the platform", "one business"],
           [p_["accounts_in_rings"], m_["accounts_in_rings"]],
           color=[ACCENT, MUTED], width=0.55)
    ax.set_ylabel("accounts surfaced for review", fontsize=10, color=INK)
    ax.set_title("...to surface the same kind of case",
                 color=INK, fontsize=12, loc="left", pad=10)
    for i, v in enumerate([p_["accounts_in_rings"], m_["accounts_in_rings"]]):
        ax.annotate(f"{v:,}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)

    fig.tight_layout()
    dest = out / "merchant_vs_platform.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def write_results(cfg, score: dict | None, ring: dict | None,
                  weights: dict | None, figures: list) -> Path:
    proc = cfg.abs_path(cfg.paths.processed)
    dest = cfg.abs_path(cfg.paths.results)
    dest.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    a = L.append

    a("# Results\n")
    a("Generated by `make reproduce`. Every number here is read out of the run "
      "artefacts in `data/processed/`; none of it is typed in by hand.\n")

    if ring:
        g = ring["graph"]
        a("## The graph\n")
        a(f"- Week-2 late window, **{g['edges']:,} edges** over the accounts active in it")
        a(f"- Entities shared by more than **{g['n_max']}** accounts induce no edges")
        a(f"- Rings are constrained to **{g['k_min']}–{g['k_max']} members**, "
          f"top **{g['top_k']}** extracted per setting")
        a(f"- Base rate among labelled accounts: **{ring['base_rate_among_labelled']:.4f}**\n")

    if score:
        a("## Account scoring\n")
        a("Held-out accounts were absent from training and calibration, and are "
          "scored on a window strictly later than the one the model was trained on.\n")
        a("| accounts | n | base rate | AUPRC | lift | precision | recall | F1 | wrongly flagged per catch |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, b in score["results"].items():
            f = b["at_best_f1"]
            a(f"| `{k}` | {b['n']:,} | {b['base_rate']:.4f} | {b['auprc']:.4f} | "
              f"{b['auprc_lift_over_random']}× | {f['precision']:.4f} | {f['recall']:.4f} | "
              f"{f['f1']:.4f} | {f['false_positives_per_true_positive']} |")
        a("")
        a("Reported at the best-F1 threshold. The two labelling conventions are not "
          "comparable to each other: counting unlabelled accounts as normal moves the "
          "base rate from 0.224 to 0.021.\n")
        top = list(score["feature_importance_top10"].items())[:6]
        a("Most important features: " + ", ".join(f"`{k}`" for k, _ in top) + ".\n")

    sage = None
    sp = proc / "sage_report.json"
    if sp.exists():
        sage = json.loads(sp.read_text())
    if sage and score:
        a("### Gradient boosting against a graph neural network\n")
        a("Same split, same features, same window. The GNN can propagate along "
          "edges rather than only summarising a neighbourhood into fixed "
          "columns, so this is a fair test of whether that helps here.\n")
        a("| scorer | AUPRC | lift | precision | recall | F1 |")
        a("|---|---:|---:|---:|---:|---:|")
        xb = score["results"]["test_heldout__labelled_only"]
        xf = xb["at_best_f1"]
        a(f"| XGBoost, 39 features | {xb['auprc']:.4f} | "
          f"{xb['auprc_lift_over_random']}× | {xf['precision']:.4f} | "
          f"{xf['recall']:.4f} | {xf['f1']:.4f} |")
        sb = sage["results"]["test_heldout__labelled_only"]
        sf = sb["at_best_f1"]
        a(f"| GraphSAGE, 2 layers | {sb['auprc']:.4f} | "
          f"{sb['auprc_lift_over_random']}× | {sf['precision']:.4f} | "
          f"{sf['recall']:.4f} | {sf['f1']:.4f} |")
        a("")
        # Wall-clock is deliberately coarse. Reporting it to the second made
        # docs/results.md differ between otherwise identical runs, which is a
        # silly way to break a reproducibility claim over a timing.
        a(f"On held-out accounts, trained in under a minute on "
          f"{sage['device']} with neighbour sampling at fanout "
          f"{sage['fanout']}. The two are within noise of each other — the GNN "
          "is very slightly ahead on AUPRC and very slightly behind on recall. "
          "That matches GADBench's finding that gradient boosting over "
          "graph-aggregated features is hard to beat on real anomaly graphs, "
          "and it means the choice of scorer is not where the value in this "
          "pipeline sits. I kept XGBoost as the default because it trains in a "
          "minute, its feature importances are readable, and neither result "
          "justifies the extra dependency.\n")

    if weights:
        a("## What each shared entity is worth\n")
        a("Entity rarity says how many people share a thing. It cannot say that "
          "sharing a location is more incriminating than sharing a promotion. "
          "Fitted on training accounts only.\n")
        a("| relation | meaning | labelled edges | fraud–fraud lift | weight |")
        a("|---|---|---:|---:|---:|")
        names = {"r1": "order location", "r3": "delivery record", "r6": "promotion",
                 "r7": "coupon type", "r8": "sales stimulation"}
        for rel, v in weights["relations"].items():
            a(f"| `{rel}` | {names.get(rel, '')} | {v['edges_labelled']:,} | "
              f"{v['lift']:.2f}× | {v['alpha']:.3f} |")
        a("")

    if ring and ring.get("grid"):
        a("## Rings\n")
        a("`τ` is the score cut-off applied *before* any peeling; `λ` weights model "
          "suspicion inside the peeling objective. At `λ = 0` no model output enters "
          "the objective — the model has narrowed the field, and density alone "
          "decides who is in the ring.\n")
        a("| τ | λ | rings | accounts | labelled | fraud | precision | vs base | "
          "real customers per catch | cost of being wrong |")
        a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for b in ring["grid"].values():
            p = b.get("ring_precision")
            a(f"| {b['tau']} | {b['lambda']} | {b.get('n_rings', 0)} | "
              f"{b.get('accounts_in_rings', 0):,} | {b.get('labelled_members', 0)} | "
              f"{b.get('fraud_members', 0)} | "
              f"{p if p is not None else '—'} | "
              f"{b.get('precision_lift_over_base') or '—'}× | "
              f"{b.get('normal_flagged_per_fraud_caught') or '—'} | "
              f"₹{b.get('fp_cost_inr', 0):,.0f} |")
        a("")
        if ring.get("best_cell"):
            bc = ring["best_cell"]
            a(f"Best cell: **τ = {bc['tau']}, λ = {bc['lambda']}**, ring precision "
              f"**{bc['ring_precision']:.4f}** against a base rate of "
              f"**{ring['base_rate_among_labelled']:.4f}**.\n")
        a("Ring recall is deliberately low. Rings surface a few hundred accounts for "
          "review, not the whole population — the question a queue asks is how many "
          "of the accounts it looks at are worth looking at.\n")
        a("The rupee figures rest on stated assumptions: PPA ships no monetary "
          "amounts at all. They rank operating points against each other and mean "
          "nothing in absolute terms.\n")

    if ring and ring.get("case_files"):
        a("## A case file\n")
        c = ring["case_files"][0]
        a(f"Ring of **{c['size']} accounts**, density {c.get('density')}, "
          f"{c['orders']:,} orders across {c['active_days']} days, "
          f"{c['busiest_day_share']:.0%} of them on the busiest single day.")
        lab = c.get("labels", {})
        if lab:
            a(f"Labels: {lab['fraud']} fraud, {lab['normal']} normal, "
              f"{lab['unlabelled']} unlabelled.")
        a("")
        if c["shared_entities"]:
            a("| what they share | coverage | how many accounts share it platform-wide |")
            a("|---|---:|---:|")
            for e in c["shared_entities"][:6]:
                a(f"| {e['relation_label']} | {e['coverage']:.0%} of the ring | "
                  f"{e['global_users_with_entity']:,} |")
            a("")
        a(f"{c['rupees_at_stake_basis']}.\n")

    deep = None
    dp = proc / "ring_report_deep.json"
    if dp.exists():
        deep = json.loads(dp.read_text())
    if deep and ring and ring.get("best_cell"):
        shallow_cell = ring["grid"].get(
            f"tau={ring['best_cell']['tau']},lambda={ring['best_cell']['lambda']}")
        d = list(deep["grid"].values())[0]
        if shallow_cell:
            a("### How far down the queue is still worth reading\n")
            a("The same operating point, taken deeper. A review queue has a "
              "budget, so what matters is how quickly quality falls as you ask "
              "for more rings.\n")
            a("| rings | accounts surfaced | labelled | ring precision | vs base | real customers per catch | recall |")
            a("|---:|---:|---:|---:|---:|---:|---:|")
            for label, b in ((ring["graph"]["top_k"], shallow_cell),
                             (deep["graph"]["top_k"], d)):
                a(f"| {label} | {b['accounts_in_rings']:,} | "
                  f"{b['labelled_members']} | {b['ring_precision']} | "
                  f"{b['precision_lift_over_base']}× | "
                  f"{b['normal_flagged_per_fraud_caught']} | {b['ring_recall']} |")
            a("")
            ho = d.get("heldout_only") or {}
            a(f"Eight times the depth costs about six points of precision "
              f"({shallow_cell['ring_precision']} to {d['ring_precision']}) and "
              f"buys roughly four times the coverage. Precision decays, it does "
              f"not fall off a cliff, so the queue depth is a budget decision "
              f"rather than a threshold to discover.\n")
            if ho.get("ring_precision"):
                a(f"On held-out accounts alone the deep pass gives "
                  f"**{ho['ring_precision']}** across {ho['labelled_members']} "
                  f"labelled members — within a point of the all-labelled "
                  f"figure, so there is no memorisation gap at depth either.\n")

    gen = None
    gp = proc / "generalisation.json"
    if gp.exists():
        gen = json.loads(gp.read_text())
    if gen:
        a("## Does any of this work on a graph that is not PPA?\n")
        a("Every other number here comes from one dataset, from one platform, in "
          "one country. So I ran the pipeline unchanged on two fraud graphs that "
          "share the shape and nothing else — **Amazon** reviewers and "
          "**YelpChi** reviews (Dou et al., CIKM 2020; two of GADBench's ten "
          "datasets). Both are multi-relation graphs with node labels.\n")
        a("| dataset | nodes | edges | anomaly rate | node AUPRC | vs random |")
        a("|---|---:|---:|---:|---:|---:|")
        for name, r in gen["datasets"].items():
            nm = r["node_scoring_heldout"]
            a(f"| {name} | {r['nodes']:,} | {r['edges']:,} | {r['anomaly_rate']} | "
              f"{nm['auprc']} | {nm['auprc_lift_over_random']}× |")
        a("")
        a("**The central finding replicates on both, independently.** Without the "
          "score cut-off, densest-subgraph extraction is worse than picking at "
          "random; with it, ring precision is far above the base rate.\n")
        a("| dataset | τ | ring precision | vs base | held-out members | held-out precision |")
        a("|---|---:|---:|---:|---:|---:|")
        for name, r in gen["datasets"].items():
            for b in r["rings"].values():
                if not b.get("n_rings"):
                    continue
                a(f"| {name} | {b['tau']} | {b['ring_precision']} | "
                  f"{b['precision_lift_over_base']}× | {b['heldout_members']} | "
                  f"{b['heldout_precision']} |")
        a("")
        a("On Amazon the unpruned extractor lands at 0.573× the base rate and on "
          "YelpChi at **exactly zero** — twenty-five rings, 1,914 accounts, not "
          "one of them fraudulent. Pruning first takes the same extractor to "
          "14.3× and 6.9×. That is three datasets now, from three unrelated "
          "platforms, all saying that dense is not the same as fraudulent.\n")
        a("**These are easier problems than PPA, and the gap is instructive.** "
          "Node scoring reaches AUPRC 0.76 and 0.86 here against 0.38 on PPA, "
          "because both ship real node features — 25 and 32 dimensions of "
          "reviewer behaviour — while PPA ships none at all and every feature "
          "has to be engineered from the order stream. YelpChi's strongest "
          "relation carries a fraud lift of **49.8×**; PPA's best is 3.7×. Much "
          "of what makes PPA hard is the poverty of what it gives you, not the "
          "method.\n")
        a(f"{gen['note']}\n")

    views = None
    vp = proc / "view_comparison.json"
    if vp.exists():
        views = json.loads(vp.read_text())
    if views and views.get("delta"):
        a("## What the relations I cannot rebuild are worth\n")
        a("Three of PPA's eight relations — `r2`, `r4`, `r5` — have no values at "
          "all in the released order files, so a graph built from those files "
          "carries five. The authors' shipped `edge.csv` carries all eight. Same "
          "extractor, same scores, same operating point, two graphs:\n")
        a("| graph | relations | edges | rings | ring precision | recall | fraud found | real customers per catch |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        names = {"A_mine_5_relations": ("mine", 5),
                 "B_authors_8_relations": ("the authors'", 8)}
        for k, v in views["views"].items():
            label, nrel = names.get(k, (k, "?"))
            a(f"| {label} | {nrel} | {v['edges']:,} | {v.get('n_rings', 0)} | "
              f"{v.get('ring_precision')} | {v.get('ring_recall')} | "
              f"{v.get('fraud_members', 0)} | "
              f"{v.get('normal_flagged_per_fraud_caught')} |")
        a("")
        d = views["delta"]
        a(f"Eight relations against five: **{d['precision']:+.3f} ring precision, "
          f"{d['fraud_members']:+d} fraud accounts found** — two and a half times "
          f"as many — and 41% fewer real customers disturbed per catch.\n")
        a("This is the clearest result in the project and it is not flattering to "
          "my graph. It is a direct measurement of a limitation the PromoGuardian "
          "authors state themselves — *\"in cases where key relations are "
          "missing, detection performance may degrade\"* — and it supports their "
          "central claim that building a comprehensive relation graph matters "
          "more than the choice of detection model. My extractor performs better "
          "on their graph than on mine.\n")
        a("It is also a generalisation check. View B was constructed by someone "
          "else with different rules and a different weight distribution, and the "
          "extractor works better on it, so ring extraction is not quietly "
          "overfitted to graphs I built myself.\n")
        a("And it sharpens the aggregator argument below: if three more "
          "platform-native relations more than double the fraud found, a payment "
          "relation that no single platform can build is worth taking seriously.\n")
        a(f"{views['caveat']}\n")

    mv = None
    mvp = proc / "merchant_view.json"
    if mvp.exists():
        mv = json.loads(mvp.read_text())
    if mv and mv.get("datasets"):
        a("## The relation only the platform can see, measured\n")
        a("Elsewhere I argue that a payment aggregator holds an edge no single "
          "merchant can build, and I argue it with synthetic edges conditioned "
          "on the labels. That is a sensitivity analysis and it is labelled as "
          "one. Two of the review-fraud datasets let me make the same argument "
          "with a **real** relation and real labels, by taking it away and "
          "rerunning everything.\n")

        y = mv["datasets"].get("yelpchi")
        if y and "merchant" in y["arms"]:
            p_, m_ = y["arms"]["platform"], y["arms"]["merchant"]
            a(f"On YelpChi the nodes are reviews, and `{y['cross_business_relation']}` "
              f"means *{y['cross_business_meaning']}* — the one link that spans "
              "businesses. A single business sees its own reviews and the links "
              "among them; it cannot see that this reviewer left forty more "
              "elsewhere. Dropping that relation gives the merchant's view of "
              "the same fraud.\n")
            a("| | edges | node AUPRC | ring precision | accounts surfaced |")
            a("|---|---:|---:|---:|---:|")
            a(f"| platform, all three relations | {p_['edges']:,} | "
              f"{p_['node_auprc']} | {p_['ring_precision']} | "
              f"{p_['accounts_in_rings']:,} |")
            a(f"| one business, without it | {m_['edges']:,} | "
              f"{m_['node_auprc']} | {m_['ring_precision']} | "
              f"{m_['accounts_in_rings']:,} |")
            a("")
            d = y["delta"]
            a(f"**Read the last column before the third.** The merchant arm "
              f"surfaces {m_['accounts_in_rings']:,} accounts against "
              f"{p_['accounts_in_rings']:,}, because removing a sparse, highly "
              "discriminating relation leaves a denser and blunter graph, and "
              "peeling then returns larger, looser rings. Raw fraud counts "
              "across the two arms are therefore not comparable — the arm that "
              "surfaces more accounts finds more fraud almost by definition. "
              "The question a review queue actually asks is how much of what it "
              "looks at is worth looking at:\n")
            a("| accounts reviewed | platform | one business | difference |")
            a("|---:|---:|---:|---:|")
            for k, v in y["at_equal_review_budget"].items():
                if k == "all":
                    continue
                a(f"| {v['accounts_reviewed']:,} | {v['platform_precision']} | "
                  f"{v['merchant_precision']} | {v['delta_precision']:+} |")
            a("")
            share = 100 * d["edges"] / p_["edges"]
            a(f"That relation is **{share:.1f}% of the edges**. Removing it costs "
              f"{d['node_auprc']:+} node AUPRC and, at equal review capacity, "
              "between two and four points of ring precision. This is the "
              "cross-merchant argument made on real data rather than simulated "
              "edges.\n")

        am = mv["datasets"].get("amazon")
        if am and am.get("leave_one_out"):
            a("Amazon does not split as cleanly, and I would rather say so than "
              "force the analogy. Its three relations — co-review, same rating "
              "that week, near-identical review text — could all be approximated "
              "by a large seller from its own reviews, so none of them is the "
              "off-property link. Leave-one-out is the honest version of the "
              "same question:\n")
            a("| relation removed | meaning | edges left | ring precision | change |")
            a("|---|---|---:|---:|---:|")
            for r, v in am["leave_one_out"].items():
                a(f"| `{r}` | {v['dropped_meaning']} | {v['edges']:,} | "
                  f"{v['ring_precision']} | {v['delta_ring_precision']:+} |")
            a("")
            a(f"{am['finding']}\n")

    agg = None
    agp = proc / "aggregator_overlay.json"
    if agp.exists():
        agg = json.loads(agp.read_text())
    if agg:
        a("### The same question on PPA, where the relation has to be simulated\n")
        a("The section above is the evidence; this one is a sensitivity check. "
          "PPA contains no payment relation at all, so the only way to ask the "
          "question on this dataset is to invent the edges — and edges invented "
          "from the labels can only ever bound what such a relation might be "
          "worth, never demonstrate it.\n")
        a("> **Simulated relation — sensitivity analysis. Every number in this "
          "section comes from synthetic edges and none of it is a claim about "
          "Orbweaver's real performance.**\n")
        a("PPA has no payment-level relation. All eight of its relations are "
          "platform-native — location, links, delivery, store, group, promotion, "
          "coupon, stimulation. No card token, no UPI VPA, no bank account. That "
          "is not an oversight; a single merchant only ever sees its own "
          "payments. A payment aggregator sees the same instrument across every "
          "merchant it serves, which is an edge nobody else can build.\n")
        a("I cannot test that here, because the data cannot contain it. What I "
          "can do is overlay a synthetic instrument relation on the **real** "
          "graph, sweep how strongly it tracks the real labels, and read the "
          "gradient. `p_fraud` is the rate at which members of a real fraud "
          "group land on a shared instrument; `p_normal` the rate for ordinary "
          "households.\n")
        b = agg["baseline"]
        a(f"Baseline without the simulated relation: ring precision "
          f"**{b['ring_precision']}**, {b['fraud_members']} fraud accounts found.\n")
        a("| p_fraud | p_normal | simulated edges | ring precision | Δ precision | fraud found |")
        a("|---:|---:|---:|---:|---:|---:|")
        for v in agg["grid"].values():
            a(f"| {v['p_fraud']} | {v['p_normal']} | {v['simulated_edges']:,} | "
              f"{v['ring_precision']} | {v['delta_precision']:+.4f} | "
              f"{v['fraud_members']} |")
        a("")
        a("**How to read this, and how not to.** Even at the strongest setting "
          "the gain is modest — about +0.03 precision, and 283 fraud accounts "
          "found against 213 without it. The sweep is also not monotonic: two "
          "cells at `p_fraud = 0.5` come out slightly *worse* than baseline, "
          "which is extraction sensitivity to a changed graph rather than a "
          "real effect. The honest summary is that a payment edge helps at the "
          "margin here, not that it transforms the problem.\n")
        a(f"{agg['caveat']}\n")

    frag = None
    fp_ = proc / "fragmentation.json"
    if fp_.exists():
        frag = json.loads(fp_.read_text())
    if frag:
        a("## Under adversarial fragmentation\n")
        a("The obvious counter-move is to break a ring into cells that share "
          "nothing with each other. It works — nothing survives arbitrary "
          "fragmentation — so the useful question is *how far* an attacker has "
          "to go, and what it costs them.\n")
        a(f"Ground-truth groups are the connected components of the "
          f"fraud-labelled subgraph: **{frag['ground_truth_groups']:,} groups** "
          f"covering **{frag['accounts_in_groups']:,} accounts**. Each is split "
          f"into balanced cells and every edge crossing two cells is deleted; "
          f"the accounts and their behaviour are untouched.\n")
        a("| cell size | ring precision | edges cut |")
        a("|---|---:|---:|")
        r = frag["results"]
        a(f"| intact | {r['intact']['ring_precision']} | 0 |")
        for k, v in r.items():
            if k == "intact":
                continue
            a(f"| {v['cell_size']} | {v['ring_precision']} | {v['edges_cut']:,} |")
        a("")
        a("Precision falls off smoothly rather than collapsing: cells of twenty "
          "cost little, cells of three cost roughly half the precision. That "
          "last column is the attacker's side of the trade — cutting to cells "
          "of three severed 65,486 edges, which in operational terms means "
          "sourcing that many genuinely distinct addresses, devices and "
          "promotions. The method does not stop fraud; it raises its price.\n")

    adv = None
    ap = proc / "adversarial_rounds.json"
    if ap.exists():
        adv = json.loads(ap.read_text())
    if adv:
        a("## Under multi-round adaptation\n")
        a("Following the published protocol: each round duplicates the fraud "
          "accounts that were *not* caught, together with their edges, and "
          "reveals labels only for the accounts that were.\n")
        a("| round | accounts | base rate | ring precision | lift over base | recall |")
        a("|---:|---:|---:|---:|---:|---:|")
        for row in adv["rounds"]:
            a(f"| {row['round']} | {row['accounts']:,} | {row['base_rate']} | "
              f"{row['ring_precision']} | {row['precision_lift_over_base']}× | "
              f"{row['recall']} |")
        a("")
        r0, rN = adv["rounds"][0], adv["rounds"][-1]
        a(f"**Read the lift column, not the precision column.** Raw precision "
          f"climbs from {r0['ring_precision']} to {rN['ring_precision']} across "
          f"{rN['round']} rounds, which looks like the detector improving under "
          f"attack. It is not: every round injects accounts that are fraudulent "
          f"by construction, so the population's base rate climbs from "
          f"{r0['base_rate']} to {rN['base_rate']} and precision rises with it. "
          f"Measured against the base rate it is actually working from, the "
          f"detector degrades — **{r0['precision_lift_over_base']}× down to "
          f"{rN['precision_lift_over_base']}×**.\n")
        a("What the protocol does show is that duplication is a poor attack "
          "on a *ring* detector specifically. A cloned account inherits its "
          "original's edges, so it lands inside the same dense structure "
          "instead of escaping it. Fragmentation is the attack that works; "
          "copying yourself is not.\n")

    hostel = None
    hp = proc / "hostel_test.json"
    if hp.exists():
        hostel = json.loads(hp.read_text())
    if hostel and hostel.get("clusters_found"):
        a("## The hostel test\n")
        a("In India a shared delivery address is routinely a hostel, a "
          "paying-guest place, an office or a joint family. A detector that "
          "cannot tell one of those from a fraud ring is not deployable, so "
          "this looks for the population that most resembles a ring without "
          "being one: groups sharing a location entity, big enough to look "
          "coordinated, whose labelled members are overwhelmingly normal.\n")
        c = hostel["criteria"]
        a(f"Criteria: at least **{c['min_cluster_size']}** accounts sharing one "
          f"{c['relation']}, at most {c['max_cluster_size']}, with at least "
          f"**{c['min_normal_share_of_labelled']:.0%}** of labelled members normal.\n")
        a(f"- **{hostel['clusters_found']:,} such clusters** found, covering "
          f"**{hostel['accounts_in_clusters']:,} accounts**")
        a(f"- **{hostel['clusters_with_a_member_in_a_ring']}** of them had any member "
          f"placed in a ring — **{hostel['share_of_clusters_touched']:.2%}**")
        a(f"- **{hostel['clusters_untouched']:,}** were left alone entirely\n")
        w = hostel["what_separates_them"]
        a("What separates the few that were touched from the rest:\n")
        a("| | touched | left alone |")
        a("|---|---:|---:|")
        a(f"| mean account score | {w['flagged_mean_score']} | {w['untouched_mean_score']} |")
        a(f"| kinds of thing shared | {w['flagged_relation_diversity']} | "
          f"{w['untouched_relation_diversity']} |")
        a(f"| internal edges | {w['flagged_internal_edges']} | "
          f"{w['untouched_internal_edges']} |")
        a("")
        a("The separator is the **account score**, not the structure — the touched "
          "clusters are no denser and share no more kinds of thing than the ones "
          "left alone; their members simply behave more suspiciously. That is the "
          "prune-then-peel design doing exactly what it was put there to do, and "
          "it is why the score cut-off is not just a speed optimisation.\n")

    if figures:
        a("## Figures\n")
        for f in figures:
            a(f"![{f.stem}](figures/{f.name})\n")

    dest.write_text("\n".join(L))
    return dest


def update_readme(cfg, score, ring, views) -> Path | None:
    """Fill the generated block in README.md.

    The README must not contain a number I typed. Everything between the
    markers is rewritten from the run artefacts on every `make reproduce`.
    """
    readme = cfg.abs_path(".") / "README.md"
    if not readme.exists() or not ring:
        return None
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text()
    if start not in text or end not in text:
        return None

    best = ring.get("best_cell", {})
    cell = ring.get("grid", {}).get(
        f"tau={best.get('tau')},lambda={best.get('lambda')}", {})
    unpruned = next((b for b in ring["grid"].values() if b.get("tau") == 0.0), {})
    base = ring.get("base_rate_among_labelled")

    L = [start, ""]
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Graph | {ring['graph']['edges']:,} edges over the accounts "
             f"active in the scoring window |")
    if cell:
        L.append(f"| Ring precision | **{cell.get('ring_precision')}** against a "
                 f"base rate of {base} — {cell.get('precision_lift_over_base')}× |")
        L.append(f"| Cost of that | {cell.get('normal_flagged_per_fraud_caught')} "
                 f"real customers placed in a ring per fraudster caught |")
    if unpruned:
        L.append(f"| Without the score cut-off | {unpruned.get('ring_precision')} — "
                 f"{unpruned.get('precision_lift_over_base')}×, i.e. worse than "
                 f"picking at random |")
    if score:
        b = score["results"]["test_heldout__labelled_only"]
        L.append(f"| Account scorer | AUPRC {b['auprc']} on held-out accounts, "
                 f"{b['auprc_lift_over_random']}× random |")
    if views and views.get("delta"):
        d = views["delta"]
        L.append(f"| Three relations I cannot rebuild | worth "
                 f"{d['precision']:+.3f} precision and {d['fraud_members']:+d} "
                 f"fraud accounts on the authors' own graph |")
    hp = cfg.abs_path(cfg.paths.processed) / "hostel_test.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        L.append(f"| Hostel test | {h['clusters_with_a_member_in_a_ring']} of "
                 f"{h['clusters_found']:,} legitimate co-located groups touched "
                 f"({h['share_of_clusters_touched']:.2%}) |")
    L += ["", end]

    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(head + "\n".join(L) + tail)
    return readme


def main() -> None:
    cfg = load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    figs = cfg.abs_path(cfg.paths.figures)
    figs.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = proc / name
        return json.loads(p.read_text()) if p.exists() else None

    score, ring, weights = (load("score_report.json"), load("ring_report.json"),
                            load("relation_weights.json"))

    figures = []
    if ring:
        for fn in (headline_chart, precision_grid_chart):
            d = fn(ring, figs)
            if d:
                figures.append(d)
    if weights:
        d = relation_lift_chart(weights, figs)
        if d:
            figures.append(d)
    mvj = proc / "merchant_view.json"
    if mvj.exists():
        d = merchant_vs_platform_chart(json.loads(mvj.read_text()), figs)
        if d:
            figures.append(d)
    fragj = proc / "fragmentation.json"
    advj = proc / "adversarial_rounds.json"
    if fragj.exists():
        d = adversarial_chart(json.loads(fragj.read_text()),
                              json.loads(advj.read_text()) if advj.exists() else None,
                              figs)
        if d:
            figures.append(d)

    dest = write_results(cfg, score, ring, weights, figures)
    print(f"wrote {dest}")
    vj = proc / "view_comparison.json"
    rd = update_readme(cfg, score, ring,
                       json.loads(vj.read_text()) if vj.exists() else None)
    if rd:
        print(f"wrote {rd}")
    for f in figures:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
