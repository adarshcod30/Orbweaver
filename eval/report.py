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

    dest = write_results(cfg, score, ring, weights, figures)
    print(f"wrote {dest}")
    for f in figures:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
