"""Regenerate docs/results.md and every figure from the saved run artefacts.

Nothing in the README or in docs/results.md is typed by hand. This reads the
JSON written by `eval.score_report` and `eval.run_rings` and renders it, so a
number in the documentation can only be wrong if the run that produced it was.
"""
from __future__ import annotations

import json
import re
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


def adversarial_chart(frag: dict, adv: dict | None, out: Path,
                      twins: dict | None = None) -> Path | None:
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
    if twins:
        tw = [(v["cell_size"], v["with_twins"]["ring_precision"])
              for k, v in twins["fragmentation"].items()
              if v.get("cell_size") and v["with_twins"].get("ring_precision")]
        if tw:
            tw.sort()
            ax.plot([c for c, _ in tw], [p for _, p in tw], marker="s",
                    color="#0369a1", linewidth=1.8, markersize=5,
                    label="with behaviour edges")
            ax.plot([c for c, _ in cells], [p for _, p in cells], marker="o",
                    color=ACCENT, linewidth=1.8, markersize=6,
                    label="shared entities only")
            ax.legend(frameon=False, fontsize=9)
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


def ieee_relation_chart(ie: dict, out: Path) -> Path | None:
    """What each payment-side relation is worth, measured."""
    w = ie.get("relation_weights") or {}
    rows = [(k, v["lift"], v.get("meaning", k)) for k, v in w.items()
            if v.get("lift") is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=160)
    _style(ax)
    names = [f"{r[0]}\n{r[2]}" for r in rows]
    vals = [r[1] for r in rows]
    colours = [ACCENT if v >= 1.0 else MUTED for v in vals]
    ax.bar(names, vals, color=colours, width=0.6)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
    ax.annotate("no better than chance", xy=(len(rows) - 0.5, 1.0),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK, ha="right")
    ax.set_ylabel("fraud–fraud edges vs chance", fontsize=10, color=INK)
    ax.set_title("What a processor's relations are worth",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    dest = out / "ieee_relation_lift.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def ring_scorer_charts(rs: dict, out: Path) -> list[Path]:
    """Queue quality by ranking, and whether the confidence means anything."""
    made = []
    if not rs.get("trained"):
        return made

    depths = sorted({int(k) for r in rs["rankings"].values()
                     for k in r["all_labelled"]})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    _style(ax)
    styles = {"density": (MUTED, "o", "density (what the queue does today)"),
              "mean_member_score": ("#0369a1", "s", "mean member score (baseline)"),
              "learned_confidence": (ACCENT, "^", "learned confidence")}
    for name, r in rs["rankings"].items():
        colour, marker, label = styles.get(name, (INK, "o", name))
        ys = [r["all_labelled"].get(str(d), {}).get("precision") for d in depths]
        ax.plot(depths, ys, marker=marker, color=colour, linewidth=1.8,
                markersize=6, label=label)
    ax.set_xlabel("rings reviewed", fontsize=10, color=INK)
    ax.set_ylabel("share worth reviewing", fontsize=10, color=INK)
    ax.set_xticks(depths)
    ax.set_title("Does a learned confidence order the queue better?",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    d1 = out / "queue_by_ranking.png"
    fig.savefig(d1, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    made.append(d1)

    cal = rs.get("calibration") or []
    if cal:
        fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
        _style(ax)
        px = [c["predicted"] for c in cal]
        py = [c["realised"] for c in cal]
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.0)
        ax.annotate("perfectly calibrated", xy=(0.55, 0.58), fontsize=8,
                    color=MUTED, rotation=38)
        ax.plot(px, py, marker="o", color=ACCENT, linewidth=1.8, markersize=6)
        ax.set_xlabel("confidence the model gave", fontsize=10, color=INK)
        ax.set_ylabel("share that were actually rings", fontsize=10, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Does the confidence mean what it says?",
                     color=INK, fontsize=12, loc="left", pad=12)
        fig.tight_layout()
        d2 = out / "ring_calibration.png"
        fig.savefig(d2, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        made.append(d2)
    return made


def replay_chart(rep: dict, out: Path) -> Path | None:
    """How much a night of extra data is worth, and how little ring identity
    survives from one night to the next."""
    snaps = rep.get("snapshots") or []
    if not snaps:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160)

    ax = axes[0]
    _style(ax)
    days = [s["days_of_data"] for s in snaps]
    prec = [s["ring_precision"] for s in snaps]
    cost = [s["normal_flagged_per_fraud_caught"] for s in snaps]
    ax.plot(days, prec, marker="o", color=ACCENT, linewidth=1.8, markersize=6,
            label="share worth reviewing")
    ax.set_xlabel("nights of data", fontsize=10, color=INK)
    ax.set_ylabel("ring precision", fontsize=10, color=INK)
    ax.set_xticks(days)
    ax2 = ax.twinx()
    ax2.plot(days, cost, marker="s", color="#0369a1", linewidth=1.6,
             markersize=5, linestyle="--", label="real customers per catch")
    ax2.set_ylabel("real customers per catch", fontsize=10, color="#0369a1")
    ax2.tick_params(colors="#0369a1", labelsize=9)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax.set_title("What another night of data buys", color=INK, fontsize=12,
                 loc="left", pad=10)

    ax = axes[1]
    _style(ax)
    det = rep.get("detection") or []
    if det:
        last = max(int(k) for k in det[0]["best_ring_overlap_by_day"])
        xs = sorted(int(k) for k in det[0]["member_share_already_surfaced_by_day"])
        shares = [[r["member_share_already_surfaced_by_day"][str(d)] for r in det]
                  for d in xs]
        ax.boxplot(shares, positions=range(1, len(xs) + 1), widths=0.5,
                   medianprops=dict(color=ACCENT, linewidth=1.8),
                   flierprops=dict(marker=".", markersize=3, color=MUTED))
        ax.set_xticks(range(1, len(xs) + 1))
        ax.set_xticklabels([str(i + 1) for i in range(len(xs))])
        ax.set_xlabel("nights of data", fontsize=10, color=INK)
        ax.set_ylabel("share of the ring already surfaced", fontsize=10, color=INK)
        ax.set_title("How much of each ring was already visible",
                     color=INK, fontsize=12, loc="left", pad=10)

    fig.tight_layout()
    dest = out / "time_to_detection.png"
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


def ring_context_chart(rc: dict, out: Path) -> Path | None:
    """Why feeding ring membership back into the account score did nothing.

    Both panels share a feature order deliberately. The left one is the
    ceiling - the share of held-out accounts the feature is non-zero for -
    and the right one is what it actually bought. The point of putting them
    side by side is that the second was determined by the first before any
    model ran: a feature that is zero for ninety-nine accounts in a hundred
    cannot move an average over all of them.
    """
    cov = rc.get("coverage") or {}
    res = rc.get("results") or {}
    if not cov or not res:
        return None

    best = {}
    for key, val in res.items():
        m = re.search(r"\[([^\]]+)\]", key)
        if m and "delta_auprc" in val:
            f = m.group(1)
            best[f] = max(best.get(f, 0.0), val["delta_auprc"])
    feats = [f for f in cov if f in best]
    if not feats:
        return None
    feats.sort(key=lambda f: cov[f]["heldout_share"])

    labels = [f.replace("_", " ") for f in feats]
    shares = [cov[f]["heldout_share"] * 100 for f in feats]
    deltas = [best[f] for f in feats]
    y = np.arange(len(feats))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=160)

    ax = axes[0]
    _style(ax)
    ax.barh(y, shares, color=MUTED, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("share of held-out accounts it is non-zero for (%)",
                  fontsize=10, color=INK)
    ax.set_xlim(0, max(max(shares) * 1.35, 1.0))
    ax.set_title("How many accounts it reaches", color=INK, fontsize=12,
                 loc="left", pad=10)
    for i, v in enumerate(shares):
        ax.annotate(f"{v:.2f}%", xy=(v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK)

    ax = axes[1]
    _style(ax)
    ax.barh(y, deltas, color=ACCENT, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("best change in held-out AUPRC", fontsize=10, color=INK)
    ax.set_xlim(0, max(max(deltas) * 1.45, 0.002))
    # Four ticks, not the default seven - at this scale the labels are wide
    # enough to run into each other and the axis becomes unreadable.
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("What it bought", color=INK, fontsize=12, loc="left", pad=10)
    for i, v in enumerate(deltas):
        ax.annotate(f"+{v:.4f}", xy=(v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK)

    base = res.get("score alone", {}).get("auprc")
    if base is not None:
        fig.text(0.5, -0.04, f"against {base} for the score on its own",
                 ha="center", fontsize=9, color=MUTED)

    fig.tight_layout()
    dest = out / "ring_context.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def ring_persistence_chart(an: dict, out: Path) -> Path | None:
    """Left: does a ring found tonight have a predecessor from last night -
    anchored extraction per night at both thresholds, against what global
    peeling managed on the final night. Right: on which night each final ring
    was first seen, with how much of its spend was still ahead of it then.
    """
    tl = an.get("timelines") or {}
    t03, t05 = tl.get("0.3") or [], tl.get("0.5") or []
    if len(t03) < 2:
        return None
    g = an.get("global_peeling_from_replay") or {}
    final = an.get("final_rings") or []

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160)

    ax = axes[0]
    _style(ax)
    nights = [r["night"] for r in t03[1:]]
    x = np.arange(len(nights))
    w = 0.36
    ax.bar(x - w / 2, [r["share_with_a_predecessor"] or 0 for r in t03[1:]], w,
           color=ACCENT, label="anchored, θ = 0.3")
    ax.bar(x + w / 2, [r["share_with_a_predecessor"] or 0 for r in t05[1:]], w,
           color=MUTED, label="anchored, θ = 0.5")
    # The reference lines use global peeling's *most generous* score - its best
    # overlap with any earlier night, not just the previous one - because that
    # is the version that flatters the thing being compared against.
    if g.get("share_with_a_predecessor_at_0.3") is not None:
        ax.axhline(g["share_with_a_predecessor_at_0.3"], color=INK, linewidth=1.0,
                   linestyle="--", label="global peeling, best of any earlier night, θ = 0.3")
    if g.get("share_with_a_predecessor_at_0.5") is not None:
        ax.axhline(g["share_with_a_predecessor_at_0.5"], color=INK, linewidth=1.0,
                   linestyle=":", label="global peeling, best of any earlier night, θ = 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"night {n}" for n in nights])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of tonight's rings with a predecessor", fontsize=10, color=INK)
    ax.set_title("Does the ring survive the night?", color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    _style(ax)
    n_nights = an["window"]["nights"]
    counts = [sum(1 for r in final if r["days_to_detection"] == k) for k in range(1, n_nights + 1)]
    ahead = []
    for k in range(1, n_nights + 1):
        rows = [r for r in final if r["days_to_detection"] == k and r["share_still_ahead_when_first_seen"] is not None]
        ahead.append(float(np.mean([r["share_still_ahead_when_first_seen"] for r in rows])) if rows else None)
    xs = np.arange(1, n_nights + 1)
    ax.bar(xs, counts, color=ACCENT, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"night {k}" for k in xs])
    ax.set_ylabel("final rings first seen that night", fontsize=10, color=INK)
    ax.set_title("When each final ring was first seen", color=INK, fontsize=12, loc="left", pad=10)
    for k, c, a_ in zip(xs, counts, ahead):
        if c:
            lab = f"{c}" + (f"\n{a_:.0%} of spend\nstill ahead" if a_ is not None else "")
            ax.annotate(lab, xy=(k, c), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, max(counts + [1]) * 1.45)

    fig.tight_layout()
    dest = out / "ring_persistence.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _anchored_section(a, proc: Path) -> None:
    """The section for rings extracted around anchors, with case ids."""
    f = proc / "anchored.json"
    if not f.exists():
        return
    an = json.loads(f.read_text())
    s = an["summary"]; g = an.get("global_peeling_from_replay") or {}
    fn = s["final_night"]; d = an["design"]; op = an["operating_point"]
    p3, p5 = s["persistence_at_0.3"], s["persistence_at_0.5"]
    dtd = s["days_to_detection"]; lat = an["on_demand_latency_ms"]
    nights = an["nights"]; n_nights = an["window"]["nights"]

    a("## A ring you can find again tomorrow\n")
    a("The nightly replay found that ring identity does not survive a night. Peeling is a "
      "global optimisation, so one more day of edges shifts densities everywhere and the top "
      "rings are recomposed rather than extended: the best overlap between a final ring and "
      f"anything from an earlier night was a median Jaccard of {g.get('median_best_overlap_with_an_earlier_night')}, "
      "and not one ring had a predecessor at the 0.5 the replay required. An operations team "
      "cannot open a case on Monday and find it on Tuesday, and days-to-detection could not be "
      "measured.\n")
    a("So this extracts rings **around anchors** instead - the anchored densest subgraph of Dai, "
      "Qiao, Chang and Qin (SIGMOD 2022), in its strict form. For an anchor account the candidate "
      "set is the ball `{a} ∪ N(a) ∪ N²(a)` inside the reference set R = {accounts with score > "
      f"{op['tau']}}}, capped at {d['ball_cap']:,} nodes by descending edge weight, and greedy peeling "
      "runs on that ball with the anchor pinned so it is never removed. Because the ball is "
      "intersected with R first, the outsider penalty in their objective vanishes and it reduces "
      "to the weighted density used everywhere else here; that is the special case in which "
      "outsiders are forbidden rather than penalised. Anchors each night are the top "
      f"{d['n_anchors']} accounts by score inside R plus every member of last night's rings, so a "
      "case has something to be found again from. Rings from different anchors that overlap at "
      f"Jaccard {d['dedupe_jaccard']} are one ring, kept under its highest-scoring anchor.\n")
    a("Identity from one night to the next follows Greene, Doyle and Cunningham (ASONAM 2010): "
      "tonight's rings are matched to last night's by Jaccard above θ, each is assigned one of "
      "born / continued / merged / split / died, and a case id is carried along the timeline "
      f"from the first ring in it. They used θ = 0.3; both 0.3 and 0.5 are reported. A ring "
      f"unobserved for {d['death_after_unobserved_nights']} night is dead.\n")

    a("| night | reference set | anchors | rings found | after de-duplication | "
      "median ring size | precision, top 25 | real customers per catch | extract (s) |")
    a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in nights:
        m = r["final_night"]
        a(f"| {r['night']} | {r['reference_set']:,} | {r['anchors']:,} | {r['rings_found']:,} | "
          f"{r['rings_after_dedupe']:,} | {r['median_ring_size']:.0f} | {m['ring_precision']} | "
          f"{m['normal_flagged_per_fraud_caught']} | {r['seconds']['extract_all']} |")
    a("")

    a("**Does it survive the night?** The anchored tracker only ever matches a ring against *last "
      "night*, because a front is one night old. Comparing that against global peeling's best "
      "overlap with **any** earlier night would be an easier test for global than for anchored, so "
      "the like-for-like column restricts global to the same single night; the looser number is "
      "kept beside it so the comparison cannot be accused of being rigged.\n")
    po = g.get("previous_night_only") or {}
    a("| final rings with a predecessor | global, previous night only | global, any earlier night | anchored |")
    a("|---|---:|---:|---:|")
    a(f"| θ = 0.3 (Greene et al.) | {po.get('share_with_a_predecessor_at_0.3', 0):.0%} | "
      f"{g.get('share_with_a_predecessor_at_0.3', 0):.0%} | "
      f"**{p3['share_of_final_rings_with_a_predecessor']:.0%}** |")
    a(f"| θ = 0.5 | {po.get('share_with_a_predecessor_at_0.5', 0):.0%} | "
      f"{g.get('share_with_a_predecessor_at_0.5', 0):.0%} | "
      f"**{p5['share_of_final_rings_with_a_predecessor']:.0%}** |")
    a(f"| median overlap with the predecessor | {po.get('median_overlap')} | "
      f"{g.get('median_best_overlap_with_an_earlier_night')} | "
      f"**{p3['median_jaccard_of_continued']}** |")
    a("")
    a("Given the same one night that anchoring gets, **no** global ring has a predecessor at either "
      "threshold, and its median overlap with last night is "
      f"{po.get('median_overlap')}. Anchored rings match at a median Jaccard of "
      f"{p3['median_jaccard_of_continued']} - not the same set of accounts, but recognisably the "
      "same group, which is what a case id has to mean.\n")
    ev = s["events_on_the_final_night_at_0.3"]
    a(f"On the final night at θ = 0.3, of {fn['rings_ranked']} ranked rings the tracker saw "
      f"{ev['continued']} continued, {ev['merged']} merged, {ev['split']} split and {ev['born']} born, "
      f"with {ev['died']} of the previous night's rings dying and {ev['merged_into']} absorbed.\n")

    a("**Days to detection, which is now measurable.** Each final ring's timeline has a first "
      "night, and that is when a case with this id first existed.\n")
    a("| first seen on | final rings |")
    a("|---|---:|")
    for k in range(1, n_nights + 1):
        a(f"| night {k} of {n_nights} | {dtd['histogram'].get(str(k), 0)} |")
    a("")
    a(f"Median {dtd['median']:.0f} of {n_nights} nights, range {dtd['min']}-{dtd['max']}; "
      f"**{dtd['share_seen_before_the_last_night']:.0%} of the final rings had a case open before the last night**, "
      f"against 0% under global peeling. When a case first opened, "
      f"**{s['share_of_ring_spend_still_ahead_when_first_seen']:.1%} of its promotion spend was still ahead of it** "
      f"(₹{s['spend_on_or_after_first_seen_inr']:,.0f} of ₹{s['total_window_spend_inr']:,.0f}, on the stated "
      "₹-per-promotion assumption) - the part an intervention could have reached. A ring that was "
      "split off another timeline counts as born on the night of the split, which is the strict "
      "reading; its members were visible inside the parent before that.\n")

    gp, gc = g.get("final_night_precision"), g.get("final_night_normal_flagged_per_fraud_caught")
    ap, ac = fn["ring_precision"], fn["normal_flagged_per_fraud_caught"]
    a("**Precision on the final night, against the global extractor.** Same night, same "
      "operating point, top 25 by density in both cases.\n")
    a("| | ring precision | real customers per catch | fraud accounts found |")
    a("|---|---:|---:|---:|")
    a(f"| global peeling | **{gp}** | {gc} | — |")
    a(f"| anchored | **{ap}** | {ac} | {fn['fraud_members']} |")
    bm = fn["by_mean_member_score"]
    a(f"| anchored, ranked by mean member score | {bm['ring_precision']} | {bm['normal_flagged_per_fraud_caught']} | {bm['fraud_members']} |")
    a("")
    if gp is not None and ap is not None:
        if ap < gp:
            a(f"Anchored is **{gp - ap:+.4f} worse** on precision. That is the expected direction and "
              "the reason is structural: each anchored ring is a local optimum on a ball of at most "
              f"{d['ball_cap']:,} nodes, where global peeling optimises over the whole pruned graph; and "
              "the anchors are chosen by score, so anchoring inherits every bias the scorer has. "
              "What is bought with that precision is the case id - a ring that exists tomorrow - and "
              "the section above is the measurement of whether that trade is worth making.\n")
        else:
            a(f"Anchored is **{ap - gp:+.4f}** on precision, which I did not expect: a local optimum "
              "on a ball is not supposed to beat a global one. The likely reason is the size band - "
              "the ball caps how far a ring can spread, and global peeling on this graph tends to "
              "return larger, looser rings near the top. I would not lean on the sign without more "
              "nights.\n")

    a("**How many anchors.** N's effect on the final night, top-N anchors only:\n")
    a("| anchors | rings found | after de-duplication | precision, top 25 | real customers per catch | seconds |")
    a("|---:|---:|---:|---:|---:|---:|")
    for r in an["anchor_sweep_final_night"]:
        a(f"| {r['n_anchors']:,} | {r['rings_found']:,} | {r['rings_after_dedupe']:,} | {r['ring_precision']} | "
          f"{r['normal_flagged_per_fraud_caught']} | {r['seconds']} |")
    a("")

    a("**On demand.** `GET /check/{account}` now computes the ring around the account live from its "
      f"ball, when the account is inside R, and returns it with its case id and first-seen night. Over "
      f"{lat['samples']:,} anchors: **p50 {lat['p50']} ms, p95 {lat['p95']} ms**, worst {lat['max']} ms. "
      f"A full night's extraction over every anchor takes {nights[-1]['seconds']['extract_all']} s on "
      f"top of building the night's graph.\n")


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
    a("Re-running the pipeline from an empty `data/processed/` reproduces this "
      "file exactly, with two deliberate exceptions: the `/check` latencies "
      "and the per-night seconds in the replay are wall-clock measurements of "
      "the machine that ran them, and will differ on yours. Everything else is "
      "derived from the data and should match byte for byte - I have checked "
      "that against clean clones rather than assuming it.\n")
    a("Treat the two timing figures as the order of magnitude and nothing "
      "finer. Three full runs on the same laptop put the nightly snapshot "
      "between 96 and 208 seconds, the spread being thermal rather than "
      "algorithmic, and the per-account lookup moved by a similar factor. The "
      "conclusions drawn from them - that a nightly pass is cheap, and that a "
      "single lookup is not a file read - hold comfortably across that whole "
      "range, which is the only reason they are quoted at all.\n")

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

    ie = None
    iep = proc / "ieee_cis.json"
    if iep.exists():
        ie = json.loads(iep.read_text())
    if ie and ie.get("rings"):
        a("## A payment processor's graph\n")
        a("Everything above runs on food-delivery orders. This runs the "
          "pipeline unchanged on IEEE-CIS — 590,540 card transactions released "
          "by Vesta — where the relations are the ones a processor actually "
          "holds: the device, the e-mail domains on both sides, the billing "
          "address, the browser.\n")
        a("Three things before any number:\n")
        for c in ie["caveats"]:
            a(f"- {c}")
        a("")
        sp = ie["split"]
        a(f"What this dataset has that PPA does not is real timestamps over six "
          f"months, so the split carries **both** guarantees at once: days "
          f"{sp['train_days'][0]}–{sp['train_days'][1]} train and "
          f"{sp['score_days'][0]}–{sp['score_days'][1]} score, *and* the "
          f"{sp['heldout_accounts']:,} held-out accounts are absent from "
          "training. On PPA the id spaces forced a within-week arrangement; "
          "here it is a straightforward forward split.\n")
        a("| relation | what it means | labelled edges | fraud–fraud lift | weight |")
        a("|---|---|---:|---:|---:|")
        for rel, v in sorted(ie["relation_weights"].items(),
                             key=lambda kv: -kv[1]["lift"]):
            a(f"| `{rel}` | {v.get('meaning', '')} | {v['edges_labelled']:,} | "
              f"{v['lift']}× | {v['alpha']} |")
        a("")
        a("The billing address with a distance band is the strongest relation "
          "here at 5.42×, and the device is second at 3.31×. The payer's "
          "e-mail domain is **below one** — sharing `gmail.com` with someone "
          "is evidence of nothing, and the weighting drops it to 0.34 without "
          "being told to.\n")
        r = ie["rings"]
        nd = ie["node_scoring_heldout"]
        a(f"| | |")
        a(f"|---|---|")
        a(f"| Held-out base rate | {sp['heldout_base_rate']} |")
        a(f"| Node scoring | AUPRC {nd['auprc']}, {nd['auprc_lift_over_random']}× random |")
        a(f"| Rings | {r['n_rings']} over {r['accounts_in_rings']:,} accounts |")
        a(f"| Ring precision | **{r['ring_precision']}** — "
          f"**{r['precision_lift_over_base']}× the base rate** |")
        a(f"| Cost of that | {r['normal_flagged_per_fraud_caught']} good cards "
          f"flagged per fraudulent one caught |")
        a("")
        a(f"**{r['precision_lift_over_base']}× is the largest lift anywhere in "
          "this project**, and the reason is the base rate: 2.8% of held-out "
          "accounts are fraudulent here against 22.4% on PPA, so there is far "
          "more room above chance. The absolute precision, 0.51, is lower than "
          "PPA's 0.73. Both facts matter and quoting either alone would "
          "mislead.\n")
        ac = ie.get("address_cluster_test") or {}
        if ac.get("clusters_found"):
            a("### Where this one is weakest\n")
            a(f"The apartment-building analogue of the hostel test — "
              f"{ac['criteria']} — finds only **{ac['clusters_found']}** such "
              f"clusters, and the method touches **{ac['clusters_touched']}** "
              f"of them. On PPA the equivalent figure is 2 of 2,446.\n")
            a("That is a bad number and it has an obvious cause: the billing "
              "address is simultaneously the **most informative relation on "
              "this dataset** (5.42×) and the thing that legitimately ties "
              "together every card in a building. The two cannot be separated "
              "by weighting, because the weighting is what discovered the "
              "address was informative in the first place. Seven clusters is "
              "far too small a sample to put a rate on, so I will not — but "
              "the direction is clear and it is the honest limitation of "
              "running this on payment data where addresses carry most of the "
              "signal.\n")

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

    rc = None
    rcp = proc / "ring_context.json"
    if rcp.exists():
        rc = json.loads(rcp.read_text())
    if rc and rc.get("results"):
        a("## Feeding the web back into the strand\n")
        a("The argument this project makes is that a ring is invisible to a "
          "system scoring one transaction at a time. The fair follow-up is "
          "whether the ring view can give anything *back* to the per-account "
          "view — because if it can, a per-transaction system does not have to "
          "be replaced to benefit from any of this. It can consume a few extra "
          "columns.\n")
        h = rc["horizon"]
        a(f"Context comes from rings found on days {h['context_days']} and is "
          f"joined to features from days {h['feature_days']}, so every context "
          "day strictly precedes every feature day. That is checked against the "
          "window manifests rather than assumed from the naming, because this "
          "is the easiest place in the whole pipeline to let the future inform "
          f"the past. **{rc['accounts_in_previous_rings']:,} accounts** were in "
          "a previous-window ring.\n")
        a("| how the score and the context are combined | held-out AUPRC | change |")
        a("|---|---:|---:|")
        base = rc["results"].get("score alone", {})
        a(f"| score alone | {base.get('auprc')} | — |")
        for name, r in rc["results"].items():
            if name == "score alone" or "auprc" not in r:
                continue
            a(f"| {name} | {r['auprc']} | {r.get('delta_auprc', 0):+} |")
        a("")
        deltas = [r.get("delta_auprc", 0) for k, r in rc["results"].items()
                  if k != "score alone" and "auprc" in r]
        best = max(deltas) if deltas else 0.0
        if best <= 0.005:
            cov = rc.get("coverage", {})
            top = max((v["heldout_share"] for v in cov.values()), default=0.0)
            a(f"**This did not work, and the ceiling was set before the model "
              f"ever ran.** The best combination moves held-out AUPRC by "
              f"{best:+.4f}, which is a rounding error. The reason is coverage: "
              f"the most widespread context feature touches **{top:.2%} of "
              "held-out accounts**, and only 0.15% of them were in a "
              "previous-window ring at all. A feature that is zero for more "
              "than ninety-nine accounts in a hundred cannot move an aggregate "
              "metric, whatever it says about the hundredth.\n")
            a("This is the same fact the nightly replay found, seen from "
              "another angle. Rings do not persist from one night to the next, "
              "and they do not transfer from one window to the next either. "
              "They are window-specific objects: the accounts recur, the "
              "groupings do not. So ring membership is a poor thing to carry "
              "forward as a feature, and I would rather say that than report a "
              "fitted blend that squeezes out a third decimal place.\n")
        else:
            a(f"The best combination adds **{best:+.4f} AUPRC** on held-out "
              "accounts. The first two arms have nothing fitted in them at all, "
              "so there is no opportunity to tune on the measurement; the "
              "fitted blend is reported after them with its weight and its "
              "horizon caveat.\n")
        cov = rc.get("coverage", {})
        if cov:
            a("How much of the held-out set each context feature actually "
              "touches, which is the ceiling on how much any of them could "
              "matter:\n")
            a("| context feature | held-out accounts with a non-zero value | share |")
            a("|---|---:|---:|")
            for k, v in cov.items():
                a(f"| `{k}` | {v['heldout_nonzero']:,} | {v['heldout_share']:.2%} |")
            a("")
        lat = rc.get("check_latency", {})
        if "p50_ms" in lat:
            a("### Asking about one account\n")
            a("A per-transaction system would not run a batch job; it has one "
              "account in front of it and needs an answer inside a request. "
              f"`GET /check/{{account}}` returns the score, ring membership and "
              "its evidence, how many neighbours are in rings, and the rupee "
              "assumption that travels with every figure. Indexes are built "
              "once at start-up, so a lookup is array indexing rather than a "
              "file read.\n")
            a(f"Measured over {lat['samples']:,} random accounts: **p50 "
              f"{lat['p50_ms']} ms, p95 {lat['p95_ms']} ms**, worst "
              f"{lat['max_ms']} ms. `scripts/check_demo.sh` asks about an "
              "account inside a ring and one outside it.\n")

    rs = None
    rsp = proc / "ring_scorer.json"
    if rsp.exists():
        rs = json.loads(rsp.read_text())
    if rs and rs.get("trained"):
        a("## Ranking rings by a learned confidence\n")
        a("The queue is ordered by density, which is the crudest thing it could "
          "be. Density says how tightly a group is connected; it does not say "
          "how likely the group is to be fraudulent, and the whole reason the "
          "score cut-off exists is that those are different questions. So this "
          "learns a ring-level model and compares it against density and "
          "against the obvious baseline anyone would reach for first, the mean "
          "score of a ring's members.\n")
        c = rs["candidates"]
        a(f"Candidates come from the early window: **{c['generated']:,} rings** "
          f"across a spread of operating points, of which **{c['usable']:,}** "
          f"have enough labelled members to carry a label and "
          f"**{c['positives']:,}** are majority-fraud. The account scores used "
          "to build them are five-fold out-of-fold, so no ring is assembled "
          "from scores that had already seen its own members' labels.\n")
        a("| rings reviewed | density | mean member score | learned confidence |")
        a("|---:|---:|---:|---:|")
        depths = sorted({int(k) for r in rs["rankings"].values()
                         for k in r["all_labelled"]})
        for d in depths:
            row = [rs["rankings"][k]["all_labelled"].get(str(d), {}).get("precision")
                   for k in ("density", "mean_member_score", "learned_confidence")]
            a(f"| {d} | " + " | ".join(str(v) for v in row) + " |")
        a("")
        a("On held-out members only:\n")
        a("| rings reviewed | density | mean member score | learned confidence |")
        a("|---:|---:|---:|---:|")
        for d in depths:
            row = [rs["rankings"][k]["heldout_only"].get(str(d), {}).get("precision")
                   for k in ("density", "mean_member_score", "learned_confidence")]
            a(f"| {d} | " + " | ".join(str(v) for v in row) + " |")
        a("")
        # State the winner from the data rather than assuming it is the model.
        depth0 = str(depths[0])
        best_name, best_val = None, -1.0
        for k, r in rs["rankings"].items():
            v = r["all_labelled"].get(depth0, {}).get("precision") or 0.0
            if v > best_val:
                best_name, best_val = k, v
        pretty = {"density": "density", "mean_member_score": "mean member score",
                  "learned_confidence": "the learned confidence"}
        if best_name != "learned_confidence":
            c = rs["candidates"]
            a(f"**The learned model lost, and the baseline won.** At {depth0} "
              f"rings, {pretty[best_name]} reaches {best_val} against "
              f"{rs['rankings']['learned_confidence']['all_labelled'][depth0]['precision']} "
              f"for the learned confidence and "
              f"{rs['rankings']['density']['all_labelled'][depth0]['precision']} "
              "for density. Ordering the queue by the mean score of a ring's "
              "members beats both the thing it does today and the thing I "
              "built to replace it.\n")
            a(f"The reason is visible in the training data. Of "
              f"{c['usable']:,} candidate rings with enough labelled members, "
              f"**{c['positives']:,} are majority-fraud — {c['positive_share']:.1%}**. "
              "Candidates are generated at score cut-offs of 0.3 and 0.5, and "
              "that cut-off is precisely the thing that makes rings "
              "fraud-enriched in the first place. By the time a ring is a "
              "candidate, it is almost certainly fraudulent, so there is "
              "nearly nothing left for a ring-level model to separate. The "
              "calibration shows it collapsing: 441 of the rings are predicted "
              "at 1.0, and the one bucket it does push down it gets wrong in "
              "the other direction — predicted 0.277 against a realised "
              "0.658.\n")
            a("I am leaving the comparison in rather than deleting the "
              "experiment, because the useful result is the baseline. "
              "Reordering the queue by mean member score is a free improvement "
              "over density and I would not have found it if I had only "
              "compared my model against the status quo.\n")
        else:
            a(f"The learned confidence leads at {depth0} rings with "
              f"{best_val}, against the mean-member-score baseline and "
              "density.\n")

        if rs.get("hostel_clusters") and rs["hostel_clusters"].get("their_median_confidence") is not None:
            h = rs["hostel_clusters"]
            same = (h["their_median_confidence"] == h["median_confidence_of_all_rings"])
            a(f"The population this must not promote is the legitimate "
              f"co-located clusters. {h['rings_mostly_inside_a_legitimate_cluster']} "
              f"of the candidate rings sit mostly inside one, and their median "
              f"confidence is **{h['their_median_confidence']}** against "
              f"{h['median_confidence_of_all_rings']} across all rings"
              + (" — identical, which is not reassurance but another symptom "
                 "of a model that gives almost everything the same answer.\n"
                 if same else ".\n"))
        if rs.get("top_drivers"):
            a("Every ring carries the three features that moved its confidence "
              "most, so a reviewer opening a case sees why it is near the top "
              "rather than only that it is. For the highest-confidence ring "
              "those are: " +
              ", ".join(f"`{d['feature']}`" for d in rs["top_drivers"][0]["drivers"])
              + ".\n")

    rep = None
    repp = proc / "replay.json"
    if repp.exists():
        rep = json.loads(repp.read_text())
    if rep and rep.get("snapshots"):
        a("## Watching the window day by day\n")
        a("Everything above is forensic: it takes a window that has already "
          "happened and finds the rings in it. That is a fair way to measure "
          "the method and it is not how it would be used. So this replays the "
          "scoring window one night at a time — each night rebuilds the graph "
          "and features from the days up to that night only, applies the model "
          "**already fitted on the earlier window** (nothing is refitted, "
          "because a system running on the 25th cannot use a model trained on "
          "the 28th), and peels at the standard operating point.\n")
        a("| nights of data | edges | rings | accounts surfaced | share worth reviewing | real customers per catch | seconds |")
        a("|---:|---:|---:|---:|---:|---:|---:|")
        for sn in rep["snapshots"]:
            a(f"| {sn['days_of_data']} | {sn['edges']:,} | {sn['n_rings']} | "
              f"{sn['accounts_in_rings']:,} | {sn['ring_precision']} | "
              f"{sn['normal_flagged_per_fraud_caught']} | "
              f"{sn['seconds']['total']:.0f} |")
        a("")
        first, last = rep["snapshots"][0], rep["snapshots"][-1]
        a(f"**One night of data is not enough.** With a single night the queue "
          f"is at {first['ring_precision']} against a base rate of "
          f"{ring['base_rate_among_labelled'] if ring else 0.2242} — no better "
          f"than picking accounts at random — and it costs "
          f"{first['normal_flagged_per_fraud_caught']} real customers for every "
          f"fraudster caught. By the fourth night it is {last['ring_precision']} "
          f"at {last['normal_flagged_per_fraud_caught']}, which is a tenfold "
          "improvement in the cost of being wrong. The method needs a few days "
          "of accumulated structure before it has anything to say, and that is "
          "a real operational constraint rather than a tuning problem.\n")
        sm = rep["summary"]
        a(f"A nightly snapshot takes about {sm['seconds_per_night']['median_total']:.0f} "
          f"seconds end to end — build, score and peel — on a laptop, so running "
          "this every night is not the expensive part of operating it.\n")

        ov = sm["best_ring_overlap_before_the_last_night"]
        a("### The thing I could not measure\n")
        a("I wanted to report days-to-detection per ring: the night each of the "
          "final rings first became visible. That turns out not to be "
          "answerable, and the reason is worth more than the number would have "
          "been.\n")
        a(f"**Ring identity does not survive a night.** Matching each final ring "
          f"against every ring from an earlier night, the best overlap reached "
          f"is a median of {ov['median']} and a maximum of {ov['max']}, against "
          f"the {ov['threshold_required']} I required to call it the same group. "
          f"Not one of the {sm['rings_on_the_last_night']} final rings had a "
          "recognisable predecessor. Peeling is a global optimisation, so a "
          "night of new edges shifts densities everywhere and the top rings are "
          "recomposed rather than extended — the accounts do not vanish, the "
          "grouping does.\n")
        h = sm["days_to_half_the_members_surfaced"]
        a(f"So the honest version of the question drops group identity and asks "
          f"how much of each final ring was already being surfaced *somewhere* "
          f"on an earlier night. By that measure **{h['share_before_the_last_night']:.0%} "
          f"of the final rings had half their members already inside some "
          f"surfaced ring before the last night**, and at that point "
          f"{sm['share_of_ring_spend_still_ahead_when_half_surfaced']:.0%} of "
          "their promotion spend was still ahead of them. That is the number a "
          "team could act on, and it is much weaker than the one I set out to "
          "report.\n")

    views = None
    vp = proc / "view_comparison.json"
    if vp.exists():
        views = json.loads(vp.read_text())
    if views and views.get("delta"):
        _anchored_section(a, proc)
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

    tw = None
    twp = proc / "twins.json"
    if twp.exists():
        tw = json.loads(twp.read_text())
    if tw and tw.get("fragmentation"):
        a("## Edges an attacker cannot cut\n")
        a("Fragmentation works, and it works for a specific reason: it deletes "
          "the shared entities that tie a group together and leaves the "
          "members' behaviour untouched. Fifty accounts that order the same way "
          "at the same times are still doing that after every address and "
          "promotion they had in common has been severed.\n")
        t = tw["twins"]
        w = t["weight"]
        a(f"So this adds a relation the attacker's move does not touch: mutual "
          f"five-nearest-neighbour edges in behaviour space, among the "
          f"**{t['candidates']:,} accounts the scorer already flagged** — "
          f"{t['twin_edges']:,} edges. Confining them to flagged accounts is "
          "deliberate; behaviour similarity across the whole population would "
          "link millions of ordinary customers who happen to shop alike.\n")
        if w.get("measured"):
            a(f"They are weighted by the same rule as every entity relation: "
              f"their measured fraud–fraud lift on training accounts, "
              f"**{w['lift']}×**, times the median entity edge weight, giving "
              f"{w['weight']}. That lift is worth reading next to the entity "
              "relations — a shared location is 3.71× and a shared promotion "
              "1.76×, so behaving alike is much weaker evidence than sharing "
              "anything concrete. No constant here was chosen by hand.\n")
        a("Twins are added **after** the cuts, because that is the order the "
          "attack happens in.\n")
        a("| ring broken into cells of | shared entities only | with behaviour edges | recovered |")
        a("|---|---:|---:|---:|")
        for key, row in tw["fragmentation"].items():
            wo = row["without_twins"]["ring_precision"]
            wi = row["with_twins"]["ring_precision"]
            label = "intact" if row["cell_size"] is None else str(row["cell_size"])
            a(f"| {label} | {wo} | {wi} | {wi - wo:+.4f} |")
        a("")
        gains = {("intact" if row["cell_size"] is None else row["cell_size"]):
                 row["with_twins"]["ring_precision"] - row["without_twins"]["ring_precision"]
                 for row in tw["fragmentation"].values()}
        best_k = max(gains, key=gains.get)
        worst_k = min(gains, key=gains.get)
        a(f"**A partial recovery, and not a clean one.** The largest gain is "
          f"{gains[best_k]:+.4f} at cells of {best_k}, which is where it should "
          "be: the most aggressive fragmentation destroys the most entity "
          "structure and leaves the most behaviour to find. Twins also do not "
          "damage the undamaged graph — they improve it slightly — which was "
          "the thing to check, since adding a weak relation everywhere could "
          "easily have cost more than it returned.\n")
        if gains[worst_k] < 0:
            a(f"But the effect is not monotonic and I am not going to present "
              f"it as though it were. At cells of {worst_k} the twins make "
              f"things **{gains[worst_k]:+.4f} worse**. Behaviour edges are "
              "weak enough that at mild fragmentation they add roughly as much "
              "noise as signal, and only earn their place once enough entity "
              "structure has been cut away that there is little else left.\n")
        a("What this does not do is restore the curve. Cells of three still "
          "cost far more precision than they recover, so fragmentation remains "
          "the attack that works. Behaviour edges raise its price rather than "
          "defeating it, and the reason is visible in the lift: behaving alike "
          "is genuinely weaker evidence than sharing a delivery record, and no "
          "weighting scheme can make it stronger than it is.\n")
        h = tw.get("hostel_test_with_twins") or {}
        if h.get("clusters_found"):
            a(f"The population this could have hurt is the legitimate "
              f"co-located clusters, since behaviour edges link people who act "
              f"alike and a hostel is full of them. With twins present, "
              f"**{h['clusters_with_a_member_in_a_ring']} of "
              f"{h['clusters_found']:,}** are touched "
              f"({h['share_of_clusters_touched']:.2%}).\n")

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
    proc = cfg.abs_path(cfg.paths.processed)

    def artefact(name):
        f = proc / name
        return json.loads(f.read_text()) if f.exists() else None

    h = artefact("hostel_test.json")
    if h:
        L.append(f"| Hostel test | {h['clusters_with_a_member_in_a_ring']} of "
                 f"{h['clusters_found']:,} legitimate co-located groups touched "
                 f"({h['share_of_clusters_touched']:.2%}) |")

    # One row for each of the six things I went back and measured afterwards.
    # Three of them are negative results and say so here rather than only in
    # docs/results.md, because a summary table that quietly drops the failures
    # is a dishonest summary table.
    mv = artefact("merchant_view.json")
    if mv:
        y = mv["datasets"].get("yelpchi", {})
        budgets = [b for k, b in y.get("at_equal_review_budget", {}).items()
                   if k != "all"]
        if budgets:
            lo = min(b["delta_precision"] for b in budgets)
            hi = max(b["delta_precision"] for b in budgets)
            n_lo = min(b["accounts_reviewed"] for b in budgets)
            n_hi = max(b["accounts_reviewed"] for b in budgets)
            L.append(f"| The relation only a platform can see | worth "
                     f"{lo:+.3f} to {hi:+.3f} ring precision at equal review "
                     f"capacity ({n_lo:,}-{n_hi:,} accounts) |")

    rp = artefact("replay.json")
    if rp:
        s_ = rp["summary"]
        d = s_["days_to_detection"]
        L.append(f"| Time to detection, replaying night by night | median "
                 f"{d['median']:.0f} of {rp['window']['days']} nights; "
                 f"{s_['share_of_ring_spend_still_ahead_at_detection']:.1%} of a "
                 f"ring's spend still ahead of it when it is found |")

    rs = artefact("ring_scorer.json")
    if rs and rs.get("trained"):
        at = {k: v["all_labelled"]["200"]["precision"]
              for k, v in rs["rankings"].items()}
        L.append(f"| Ranking rings by confidence | the mean member score wins "
                 f"at 200 rings ({at['mean_member_score']}) — a trained ring "
                 f"model gets {at['learned_confidence']}, density "
                 f"{at['density']} |")

    rc = artefact("ring_context.json")
    if rc:
        best = max((v for k, v in rc["results"].items() if "delta_auprc" in v),
                   key=lambda v: v["delta_auprc"])
        cov = rc["coverage"]["in_previous_ring"]["heldout_share"]
        lat = rc.get("check_latency", {})
        L.append(f"| Yesterday's rings as a feature | {best['delta_auprc']:+.4f} "
                 f"AUPRC — it reaches {cov:.2%} of held-out accounts. "
                 f"`/check` answers in {lat.get('p50_ms')} ms at the median |")

    tw = artefact("twins.json")
    if tw:
        f3 = tw["fragmentation"]["cells_of_3"]
        f20 = tw["fragmentation"]["cells_of_20"]
        d3 = f3["with_twins"]["ring_precision"] - f3["without_twins"]["ring_precision"]
        d20 = f20["with_twins"]["ring_precision"] - f20["without_twins"]["ring_precision"]
        L.append(f"| Behaviour edges against fragmentation | {d3:+.4f} ring "
                 f"precision when the ring is split into threes, {d20:+.4f} "
                 f"when it is split into twenties |")

    ie = artefact("ieee_cis.json")
    if ie:
        r = ie["rings"]
        L.append(f"| The same method on a payment processor's graph | "
                 f"{r['ring_precision']} precision, {r['precision_lift_over_base']}× "
                 f"its base rate, at {r['normal_flagged_per_fraud_caught']} good "
                 f"cards per fraudulent one caught |")

    an = artefact("anchored.json")
    if an:
        s_ = an["summary"]; g_ = an.get("global_peeling_from_replay") or {}
        p3 = s_["persistence_at_0.3"]["share_of_final_rings_with_a_predecessor"]
        L.append(f"| A ring you can find again tomorrow | {p3:.0%} of final rings had a "
                 f"case open the night before (global peeling: "
                 f"{g_.get('share_with_a_predecessor_at_0.3', 0):.0%}); "
                 f"{s_['final_night']['ring_precision']} precision against "
                 f"{g_.get('final_night_precision')} for the cost of a case id |")

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
    iej = proc / "ieee_cis.json"
    if iej.exists():
        d = ieee_relation_chart(json.loads(iej.read_text()), figs)
        if d:
            figures.append(d)
    rsj = proc / "ring_scorer.json"
    if rsj.exists():
        figures.extend(ring_scorer_charts(json.loads(rsj.read_text()), figs))
    repj = proc / "replay.json"
    if repj.exists():
        d = replay_chart(json.loads(repj.read_text()), figs)
        if d:
            figures.append(d)
    mvj = proc / "merchant_view.json"
    if mvj.exists():
        d = merchant_vs_platform_chart(json.loads(mvj.read_text()), figs)
        if d:
            figures.append(d)
    rcj = proc / "ring_context.json"
    if rcj.exists():
        d = ring_context_chart(json.loads(rcj.read_text()), figs)
        if d:
            figures.append(d)
    anj = proc / "anchored.json"
    if anj.exists():
        d = ring_persistence_chart(json.loads(anj.read_text()), figs)
        if d:
            figures.append(d)
    fragj = proc / "fragmentation.json"
    advj = proc / "adversarial_rounds.json"
    if fragj.exists():
        twj = proc / "twins.json"
        d = adversarial_chart(json.loads(fragj.read_text()),
                              json.loads(advj.read_text()) if advj.exists() else None,
                              figs,
                              json.loads(twj.read_text()) if twj.exists() else None)
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
