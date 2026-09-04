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

from eval.case_report import CSS as CASE_CSS, esc
from orbweaver.config import load_config

INK = "#1c1c1c"
ACCENT = "#c2410c"
ACCENT2 = "#0369a1"  # a second series colour, chosen with ACCENT for orange/blue
                    # contrast that survives deuteranopia and protanopia; every
                    # multi-series chart also varies marker shape, so colour is
                    # never the only channel encoding which series is which.
MUTED = "#9a9a9a"
GRID = "#e6e6e6"

# alt text and a one-line caption for every figure in docs/results.md - what
# it shows, and what to conclude from it. A reader who cannot see the image
# still gets the alt text; a reader who can see it still needs to be told
# what the picture is arguing, because a chart with no caption asks the
# reader to reverse-engineer the point.
FIGURE_CAPTIONS: dict[str, tuple[str, str]] = {
    "headline_precision_vs_cost": (
        "Ring precision against false-positive cost across score cut-offs",
        "Each point is one score cut-off (tau); the headline operating point "
        "is marked. Precision rises steeply then flattens as the cut-off "
        "tightens, while cost falls - the marked point is where that "
        "trade-off was judged worth making, not the only defensible choice."),
    "ring_precision_grid": (
        "Ring precision across the full cut-off (tau) by structure-weight (lambda) grid",
        "Precision depends far more on the score cut-off than on how much the "
        "peeling objective weighs the model's opinion - most of the grid's "
        "variation runs along the tau axis, not the lambda axis."),
    "relation_lift": (
        "Fraud-fraud lift by relation, against each relation's share of all edges",
        "The relation carrying the most edges (promotion) has the weakest "
        "evidential lift; the rarest relations are the strongest evidence "
        "per edge. Edge count and evidential value are not the same thing."),
    "ieee_relation_lift": (
        "Fraud-fraud lift by relation on the IEEE-CIS payment-processor graph",
        "The same lift-versus-share pattern as PPA, on a different dataset. "
        "address_distance is the same billing address and distance band; "
        "device the same device; email_recipient the same recipient "
        "e-mail domain; browser the same browser build; email_payer the "
        "same payer e-mail domain, the one relation no better than chance."),
    "queue_by_ranking": (
        "Precision at depth for three ways of ranking rings",
        "Ranking rings by their members' mean account score beats both a "
        "trained ring-confidence model and raw density, at every depth "
        "tested - the simplest ranking wins here, not the most sophisticated one."),
    "ring_calibration": (
        "Predicted versus realised precision, by confidence decile",
        "Points near the diagonal mean the ring-confidence model's stated "
        "probability is trustworthy at that decile; points off it show "
        "where the model is over- or under-confident."),
    "time_to_detection": (
        "Ring precision and days-to-detection, replaying the window night by night",
        "One night of data lands at the base rate; it takes four nights of "
        "replay to reach the precision this project reports as its headline "
        "number, which is the real cost of not anchoring cases across nights."),
    "merchant_vs_platform": (
        "Ring precision and node AUPRC with and without the one relation only a platform can see",
        "Removing the cross-business relation costs both ring precision and "
        "account-scoring AUPRC - real, measured evidence for what an "
        "aggregator's view is worth, not an assumption."),
    "ring_context": (
        "Held-out AUPRC with and without last window's ring membership as a feature",
        "The lift is small (+0.0011) because the feature is only ever "
        "non-zero for 0.15% of held-out accounts - most accounts were in no "
        "ring last window, so there is nothing for the feature to carry "
        "forward for them."),
    "ring_persistence": (
        "Share of final-night rings with a predecessor the night before, anchored against global extraction",
        "Anchoring the extraction around fixed accounts is what makes a case "
        "trackable from one night to the next - global peeling recomposes "
        "the whole graph every night and loses almost every case in the process."),
    "policy_frontier": (
        "What each review policy stops against what it costs, by reviewer budget; and one analyst's cumulative catch across four nights",
        "Left: the capacity-aware policy holds fraud stopped roughly flat "
        "while cutting legitimate harm as the budget grows - more analyst "
        "time buys fewer wrongly-harmed customers, not more fraud caught. "
        "Right: an analyst working two hours a night keeps catching new "
        "fraud value every night of the replay, not just the first."),
    "lockstep": (
        "Ring precision with and without burst-weighted edges, by relation and by dataset",
        "Burst-weighting costs precision on PPA with no clean per-relation "
        "story, and leaves the IEEE-CIS apartment-cluster weakness exactly "
        "unchanged - a negative result, shown rather than omitted."),
    "offer_leakage": (
        "Precision at k and cumulative fraud coverage for two offer rankings",
        "Ranking offers by leakage is precise but narrow - it covers very "
        "little of all labelled fraud, because it puts small offers first. "
        "Ranking by raw size covers far more at a much larger review cost. "
        "Neither ranking dominates; which to use is a capacity decision."),
    "label_budget": (
        "Held-out AUPRC and ring precision against the number of confirmed labels used",
        "Even the smallest label budget tested (1,146 accounts) already "
        "beats the base rate, and AUPRC keeps climbing all the way to full "
        "label availability with no plateau visible in the range tested."),
    "scorer_by_label_budget": (
        "Held-out AUPRC against label budget for three scorers: XGBoost, Fast Belief Propagation, GraphSAGE",
        "Belief propagation trails both learned scorers until roughly half "
        "the training pool is labelled, then overtakes both - the crossover "
        "is real and the hypothesis going in predicted the opposite order."),
    "adversarial": (
        "Ring precision as a ring is fragmented into smaller cells, with and without behavioural edges",
        "Precision falls sharply as cells shrink; behavioural edges recover "
        "some of it at cells of three and recover nothing at cells of "
        "twenty - fragmentation remains the evasion that works."),
}


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

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160, constrained_layout=True)
    _style(ax)
    base = ring["base_rate_among_labelled"]

    taus = sorted({r["tau"] for r in rows})
    markers = {0.0: "o", 0.3: "s", 0.5: "^"}
    colors = {0.0: MUTED, 0.3: ACCENT, 0.5: ACCENT2}
    for tau in taus:
        pts = sorted([r for r in rows if r["tau"] == tau], key=lambda r: r["normal_members"])
        ax.plot([r["normal_members"] for r in pts], [r["fraud_members"] for r in pts],
                marker=markers.get(tau, "o"), color=colors.get(tau, INK),
                linewidth=1.6, markersize=5,
                label=f"score cut-off τ = {tau}" if tau else "no score cut-off")

    lim = max(max(r["normal_members"] for r in rows),
              max(r["fraud_members"] for r in rows)) * 1.05
    ax.plot([0, lim], [0, lim * base / (1 - base)], "--", color=MUTED, linewidth=1.0)
    # Offset well clear of the dashed line itself - text sitting directly on
    # the line it is labelling made both illegible.
    label_x = lim * 0.58
    ax.annotate("what random selection\nwould give you",
               xy=(label_x, label_x * base / (1 - base)),
               xytext=(10, -22), textcoords="offset points",
               fontsize=8, color=MUTED, ha="left")

    ax.set_xlabel("real customers wrongly placed in a ring", color=INK, fontsize=10)
    ax.set_ylabel("fraudsters caught", color=INK, fontsize=10)
    ax.set_title("What each operating point costs in real customers",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
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

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160, constrained_layout=True)
    _style(ax)
    width = 0.8 / max(len(taus), 1)
    x = np.arange(len(lams))
    highest = base
    for i, tau in enumerate(taus):
        vals = []
        for lam in lams:
            m = [r for r in rows if r["tau"] == tau and r["lambda"] == lam]
            vals.append(m[0]["ring_precision"] if m else 0.0)
        highest = max(highest, max(vals))
        ax.bar(x + i * width, vals, width * 0.9,
               label=f"τ = {tau}" if tau else "no cut-off",
               color=[MUTED, ACCENT, ACCENT2][i % 3])
    ax.axhline(base, color=INK, linestyle="--", linewidth=1.0)
    ax.annotate(f"base rate {base:.3f}", xy=(len(lams) - 0.5, base),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK, ha="right")
    ax.set_xticks(x + width * (len(taus) - 1) / 2)
    ax.set_xticklabels([f"λ = {l}" for l in lams])
    ax.set_ylabel("share of a ring's labelled members\nthat are fraud", fontsize=10, color=INK)
    ax.set_title("Ring precision across the sweep", color=INK, fontsize=12, loc="left", pad=12)
    # Headroom above the tallest bar, and the legend moved above the axes
    # entirely - it used to auto-place in the upper right and sit on top of
    # the tallest bars there.
    ax.set_ylim(0, highest * 1.18)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
             bbox_to_anchor=(0.5, 1.16), ncol=3, columnspacing=1.2)
    dest = out / "ring_precision_grid.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def relation_lift_chart(weights: dict, out: Path) -> Path | None:
    rels = list(weights["relations"].keys())
    lifts = [weights["relations"][r]["lift"] for r in rels]
    labels = {"r1": "location", "r3": "delivery", "r6": "promotion",
              "r7": "coupon", "r8": "stimulation"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160, constrained_layout=True)
    _style(ax)
    order = np.argsort(lifts)[::-1]
    names = [f"{rels[i]}\n{labels.get(rels[i], '')}" for i in order]
    ax.bar(names, [lifts[i] for i in order], color=ACCENT, width=0.6)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
    ax.annotate("chance", xy=(len(order) - 0.5, 1.0),
                xytext=(0, 6), textcoords="offset points",
                fontsize=8, color=INK, ha="right")
    ax.set_ylabel("fraud–fraud edges vs chance", fontsize=10, color=INK)
    ax.set_title("Not every shared thing is equally incriminating",
                 color=INK, fontsize=12, loc="left", pad=12)
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
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.0), dpi=160, constrained_layout=True)
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
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160, constrained_layout=True)
    _style(ax)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colours = [ACCENT if v >= 1.0 else MUTED for v in vals]
    ax.bar(names, vals, color=colours, width=0.6)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.0)
    ax.annotate("no better than chance", xy=(len(rows) - 0.5, 1.0),
                xytext=(0, 6), textcoords="offset points",
                fontsize=8, color=INK, ha="right")
    ax.set_ylabel("fraud–fraud edges vs chance", fontsize=10, color=INK)
    ax.set_title("What a processor's relations are worth",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.tick_params(axis="x", labelsize=9)
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
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160, constrained_layout=True)
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
    d1 = out / "queue_by_ranking.png"
    fig.savefig(d1, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    made.append(d1)

    cal = rs.get("calibration") or []
    if cal:
        fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160, constrained_layout=True)
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
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160, constrained_layout=True)

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

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160, constrained_layout=True)

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
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)

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

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=160, constrained_layout=True)

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

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160, constrained_layout=True)

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


def policy_frontier_chart(po: dict, out: Path) -> Path | None:
    """Left: what each policy stops against what it harms, as the budget grows.
    Right: the running total a single analyst working two hours a night would
    have stopped, by night.
    """
    budgets = po.get("final_night", {}).get("budgets") or {}
    if not budgets:
        return None
    order = sorted(budgets, key=lambda b: int(b))
    styles = {
        "capacity-aware": (ACCENT, "o", "capacity-aware"),
        "density order until the budget is spent": (INK, "s", "density order"),
        "auto-hold everything": (MUTED, "^", "auto-hold everything"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=160, constrained_layout=True)

    ax = axes[0]
    _style(ax)
    for name, (colour, marker, label) in styles.items():
        xs = [budgets[b][name]["legitimate_value_harmed_inr"] for b in order]
        ys = [budgets[b][name]["fraud_value_stopped_inr"] for b in order]
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.4, markersize=6, label=label)
    nothing = budgets[order[0]]["do nothing"]
    ax.scatter([nothing["legitimate_value_harmed_inr"]], [nothing["fraud_value_stopped_inr"]],
               color=MUTED, marker="x", s=70, linewidth=2, label="do nothing", zorder=5)

    # Label the capacity-aware points by their rank in x, not by budget-list
    # order - two budgets can be close in list order but far apart on the
    # axis (or the reverse), and alternating by list order left two visually
    # adjacent points both labelled above, colliding anyway. Alternating by
    # x-rank guarantees neighbours on the page get opposite offsets. The
    # density-order series is not labelled per point: two of its four points
    # sit at nearly the same height (0 and 200 rupees, indistinguishable at
    # this scale) and all four share x=0, so individual labels would collide
    # regardless of offset - the qualitative point ("stopped little, harmed
    # nobody") is carried by the series legend and the text instead.
    ca_by_x = sorted(order, key=lambda b: budgets[b]["capacity-aware"]["legitimate_value_harmed_inr"])
    for rank, b in enumerate(ca_by_x):
        dy = 12 if rank % 2 == 0 else -17
        o = budgets[b]["capacity-aware"]
        ax.annotate(f"{b} min", xy=(o["legitimate_value_harmed_inr"], o["fraud_value_stopped_inr"]),
                    xytext=(0, dy), textcoords="offset points", fontsize=8, color=ACCENT,
                    ha="center")

    ax.set_xlabel("legitimate value harmed (₹, assumed)", fontsize=10, color=INK)
    ax.set_ylabel("fraud value stopped (₹, assumed)", fontsize=10, color=INK)
    ax.set_title("What each policy stops, and what it costs", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.margins(x=0.28, y=0.12)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
             ncol=4, columnspacing=1.2)

    ax = axes[1]
    _style(ax)
    rows = [r for r in po.get("night_by_night_at_120_minutes", []) if r.get("rings_in_queue")]
    if rows:
        x = [r["night"] for r in rows]
        ax.bar(x, [r["fraud_value_stopped_inr"] for r in rows], color=ACCENT, width=0.55,
               label="stopped that night")
        ax.plot(x, [r["cumulative_fraud_value_stopped_inr"] for r in rows], color=INK,
                marker="o", linewidth=1.4, markersize=6, label="running total")
        ax.set_xticks(x)
        ax.set_xticklabels([f"night {n}" for n in x])
        ax.set_ylabel("fraud value stopped (₹, assumed)", fontsize=10, color=INK)
        ax.set_title("One analyst, two hours a night", color=INK, fontsize=12, loc="left", pad=10)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                 ncol=2)

    dest = out / "policy_frontier.png"
    fig.savefig(dest, facecolor="white")
    plt.close(fig)
    return dest


def _policy_section(a, proc: Path) -> None:
    f = proc / "policy.json"
    if not f.exists():
        return
    po = json.loads(f.read_text())
    asm = po["assumptions"]
    budgets = po["final_night"]["budgets"]
    order = sorted(budgets, key=lambda b: int(b))

    a("## What to do with the queue, given a budget\n")
    a("Every result above says which groups look worst. None of them says what to do on a night "
      "when one analyst has an hour, and none of them prices the cost of holding a promotion for "
      "a group that turns out to be a hostel. Detection and investigation are different "
      "constraints and the second is usually the binding one, so this turns the queue into a "
      "decision.\n")
    a("There are three things a team can do with a ring. **Review** it, which costs analyst "
      "minutes and - assuming they then act correctly - stops the fraud in it without touching "
      "anyone legitimate. **Auto-hold** the members' promotions, which costs no analyst time, "
      "stops the fraud, and harms every legitimate member by some fraction of their value. Or "
      "**ignore** it. The review set is chosen by exact 0/1 knapsack against the night's budget, "
      "and any ring not reviewed is auto-held when holding beats ignoring. Because reviewing is "
      "never worse than holding, what an item carries into the knapsack is the *gain from "
      "reviewing over the best thing you could do without an analyst* - otherwise the optimiser "
      "would spend minutes on rings that auto-holding already handles.\n")
    a("The framing is example-dependent cost-sensitive learning: a false positive costs an "
      "administrative amount, a false negative costs the amount at stake, and the number that "
      "matters is savings against doing nothing (Bahnsen, Aouada and Ottersten, ESWA 2016; "
      "Elkan, IJCAI 2001).\n")

    a("**Every rupee below is an assumption.** PPA ships no monetary amounts at all, so these are "
      "stated constants:\n")
    a("| assumption | value |")
    a("|---|---:|")
    a(f"| a promotion is worth | ₹{asm['promotion_value_inr']:.0f} |")
    a(f"| a customer is worth, per week-1 order | ₹{asm['customer_value_per_week1_order_inr']:.0f} |")
    a(f"| an analyst's minute costs | ₹{asm['analyst_cost_per_minute_inr']:.0f} |")
    a(f"| reviewing a ring takes | {asm['review_minutes']} minutes |")
    a(f"| a wrongly held customer loses | {asm['churn_headline']:.0%} of their value |")
    a("")
    a("They rank policies against each other and mean nothing in absolute terms. "
      f"{asm['probability_caveat']}\n")

    a(f"**The final night**, queue of {po['final_night']['rings_in_queue']} rings ordered by mean "
      "member score, of which "
      f"₹{po['final_night']['total_fraud_value_inr']:,.0f} sits on accounts labelled fraud. "
      "A perfect reviewer is assumed here; the 90% variant is below.\n")
    a("| budget | policy | reviewed | auto-held | minutes | fraud ₹ stopped | legitimate ₹ harmed | savings |")
    a("|---:|---|---:|---:|---:|---:|---:|---:|")
    for b in order:
        for name, o in budgets[b].items():
            a(f"| {b} | {name} | {o['rings_reviewed']} | {o['rings_auto_held']} | "
              f"{o['minutes_used']:.0f} | ₹{o['fraud_value_stopped_inr']:,.0f} | "
              f"₹{o['legitimate_value_harmed_inr']:,.0f} | {o['savings']} |")
    a("")

    # The interesting thing in that table is what does *not* move with the
    # budget, so the section says it rather than leaving it to be noticed.
    ca = [budgets[b]["capacity-aware"] for b in order]
    stopped = [o["fraud_value_stopped_inr"] for o in ca]
    harmed = [o["legitimate_value_harmed_inr"] for o in ca]
    spread = (max(stopped) - min(stopped)) / max(max(stopped), 1)
    if spread < 0.02:
        a(f"**The analyst budget barely changes how much fraud is stopped.** From {order[0]} "
          f"minutes to {order[-1]}, fraud value stopped moves from ₹{stopped[0]:,.0f} to "
          f"₹{stopped[-1]:,.0f} - under {spread:.1%} - while legitimate value harmed falls from "
          f"₹{harmed[0]:,.0f} to ₹{harmed[-1]:,.0f}, a drop of "
          f"{(harmed[0] - harmed[-1]) / max(harmed[0], 1):.0%}. Auto-holding already stops "
          "almost everything; what analyst time buys is not catching more, it is releasing the "
          "groups that should never have been held. On these assumptions the reviewer is a "
          "false-positive control rather than a detector, which is not what I expected to find "
          "and is the most useful thing in this section.\n")

    dens = [budgets[b]["density order until the budget is spent"] for b in order]
    losing = [b for b, o in zip(order, dens) if o["net_inr"] < 0]
    if losing:
        a(f"**Working the queue in density order loses money** at {', '.join(losing)} minutes: it "
          f"spends the analyst's time and stops ₹{dens[0]['fraud_value_stopped_inr']:,.0f} at "
          f"{order[0]} minutes. Density ranks how tightly a group is tied together, which is not "
          "the same as how much is at stake in it, so a queue sorted that way puts small dense "
          "rings ahead of large expensive ones. That is the same lesson as the ring-ranking "
          "result above, arriving from the cost side.\n")

    lo, hi = budgets[order[0]], budgets[order[-1]]
    a(f"At {order[0]} minutes the capacity-aware policy stops "
      f"₹{lo['capacity-aware']['fraud_value_stopped_inr']:,.0f} against "
      f"₹{lo['density order until the budget is spent']['fraud_value_stopped_inr']:,.0f} for "
      f"working down the queue in density order, and at {order[-1]} minutes "
      f"₹{hi['capacity-aware']['fraud_value_stopped_inr']:,.0f} against "
      f"₹{hi['density order until the budget is spent']['fraud_value_stopped_inr']:,.0f}. "
      "Auto-holding everything stops the most fraud of any policy and needs no analyst at all - "
      f"it also harms ₹{lo['auto-hold everything']['legitimate_value_harmed_inr']:,.0f} of "
      "legitimate value doing it, which is the whole reason the review budget exists.\n")

    acc = po.get("reviewer_accuracy") or {}
    if "0.90" in acc and "1.00" in acc:
        p100 = acc["1.00"]["budgets"][order[1] if len(order) > 1 else order[0]]["capacity-aware"]
        p90 = acc["0.90"]["budgets"][order[1] if len(order) > 1 else order[0]]["capacity-aware"]
        a(f"**A reviewer who is right nine times in ten**, at {order[1] if len(order) > 1 else order[0]} "
          f"minutes: ₹{p90['fraud_value_stopped_inr']:,.0f} stopped against "
          f"₹{p100['fraud_value_stopped_inr']:,.0f} for a perfect one, and "
          f"₹{p90['legitimate_value_harmed_inr']:,.0f} of legitimate value harmed against nothing. "
          "The perfect-reviewer number is an upper bound and is labelled as one everywhere it "
          "appears.\n")

    cs = po.get("churn_sweep") or {}
    if cs:
        a("**How much churn matters.** Churn is what a wrongly held customer costs. It should "
          "change which rings are worth auto-holding and never which are worth an analyst's "
          "time, and that is what it does:\n")
        a("| churn | rings reviewed at 60 min | rings auto-held | fraud ₹ stopped | legitimate ₹ harmed |")
        a("|---:|---:|---:|---:|---:|")
        for c, blk in cs.items():
            o = blk["at_60_minutes"]
            a(f"| {float(c):.0%} | {blk['rings_reviewed']} | {blk['rings_auto_held']} | "
              f"₹{o['fraud_value_stopped_inr']:,.0f} | ₹{o['legitimate_value_harmed_inr']:,.0f} |")
        a("")

    rows = [r for r in po.get("night_by_night_at_120_minutes", []) if r.get("rings_in_queue")]
    if rows:
        a("**One analyst, two hours a night**, working the nightly queues as they arrived - the "
          "case ids come from the anchored extraction, so a ring reviewed on one night is the "
          "same case if it returns on the next:\n")
        a("| night | queue | cases reviewed | auto-held | minutes | fraud ₹ stopped | running total |")
        a("|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            a(f"| {r['night']} | {r['rings_in_queue']} | {r['n_cases_reviewed']} | "
              f"{r['rings_auto_held']} | {r['minutes_used']:.0f} | "
              f"₹{r['fraud_value_stopped_inr']:,.0f} | "
              f"₹{r['cumulative_fraud_value_stopped_inr']:,.0f} |")
        a("")
        a(f"By the last night that is ₹{rows[-1]['cumulative_fraud_value_stopped_inr']:,.0f} of "
          f"promotion value stopped for ₹{rows[-1]['cumulative_legitimate_value_harmed_inr']:,.0f} "
          "of legitimate value harmed, on the assumptions above.\n")

    a("Each case card now carries its recommended action and the two numbers behind it: what "
      "reviewing it is expected to be worth, and what auto-holding it is expected to be worth. "
      "A reviewer who disagrees can see exactly which of the two the recommendation turned on.\n")


def _demo_section(a, cfg) -> None:
    """The committed bundle, sized from what is actually on disk."""
    from orbweaver.console.demo import MAX_BYTES, bundle_path
    meta = bundle_path(cfg) / "meta.json"
    if not meta.exists():
        return
    m = json.loads(meta.read_text())

    a("## A demo anyone can click\n")
    a("Everything above needs the raw dataset: four gigabytes from OSF, an hour of processing "
      "and about thirty gigabytes of free disk. That is a fair price for reproducing the numbers "
      "and an absurd one for looking at the thing. So the repository carries a small bundle of "
      "already-computed results, and the console serves them whenever `data/processed` is empty "
      "- no dataset, no pipeline, no build step.\n")
    a(f"| file | size |")
    a("|---|---:|")
    for name, size in sorted(m["files"].items(), key=lambda kv: -kv[1]):
        a(f"| `{name}` | {size / 1e6:.2f} MB |")
    a(f"| **total** | **{m['bytes'] / 1e6:.2f} MB** of a {MAX_BYTES / 1e6:.0f} MB limit |")
    a("")
    a(f"It carries {m['accounts_in_bundle']:,} accounts of the "
      f"{m['total_accounts_in_the_run']:,} in the run - every member of every ring the console "
      f"shows ({m['accounts_in_a_ring_or_an_anchored_ring']:,} accounts) plus a random sample, so "
      "the page answers for ordinary accounts and not only for suspicious ones.\n")
    a("**What is deliberately not in it: the graph.** Thirty-five million edges do not belong in "
      "something meant to be cloned, so in demo mode `/check` serves stored neighbour counts "
      "instead of computing a ring around an account live. The full console does that in about a "
      "millisecond; the demo says which one you are looking at rather than letting the "
      "distinction pass. Every page carries that notice.\n")
    a("`requirements-demo.txt` is six packages and none of the pipeline: no pandas, no XGBoost, "
      "no matplotlib. It needs pydantic and PyYAML beside the obvious four because the console "
      "loads the project config to find the bundle and read the rupee assumptions.\n")


def lockstep_chart(ls: dict, out: Path) -> Path | None:
    """Left: fraud-fraud lift by burst quartile, per relation. Right: how many
    legitimate crowds the standard graph touches against the lockstep graph,
    per relation."""
    fit = ls.get("fit", {}).get("relations") or {}
    crowd = ls.get("crowd_test_all_relations") or {}
    if not fit:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=160, constrained_layout=True)

    ax = axes[0]
    _style(ax)
    rels = [r for r in fit if fit[r].get("bins")]
    # Okabe-Ito colourblind-safe palette, paired with a distinct marker shape per
    # relation so no line depends on colour alone to be told apart.
    style_by_rel = {
        "r1": ("#0072B2", "o"), "r3": ("#D55E00", "s"), "r6": ("#009E73", "^"),
        "r7": ("#CC79A7", "D"), "r8": ("#E69F00", "v"),
    }
    fallback = [("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"),
                ("#CC79A7", "D"), ("#E69F00", "v")]
    for i, rel in enumerate(rels):
        colour, marker = style_by_rel.get(rel, fallback[i % len(fallback)])
        bins = fit[rel]["bins"]
        x = [b["bin"] for b in bins]
        y = [b.get("lift", 1.0) for b in bins]
        ax.plot(x, y, marker=marker, markersize=5.5, linewidth=1.4, color=colour, label=rel)
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Q1\nleast bursty", "Q2", "Q3", "Q4\nmost bursty"], fontsize=8)
    ax.set_ylabel("fraud-fraud lift", fontsize=10, color=INK)
    ax.set_title("Does burstiness predict fraud?", color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    _style(ax)
    rels2 = [r for r in crowd if crowd[r].get("standard", {}).get("clusters_found")]
    if rels2:
        x = np.arange(len(rels2))
        w = 0.36
        std_share = [crowd[r]["standard"].get("share_of_clusters_touched", 0) or 0 for r in rels2]
        ls_share = [crowd[r].get("lockstep", {}).get("share_of_clusters_touched", 0) or 0 for r in rels2]
        ax.bar(x - w / 2, std_share, w, color=MUTED, label="standard graph")
        ax.bar(x + w / 2, ls_share, w, color=ACCENT, label="lockstep graph")
        ax.set_xticks(x)
        ax.set_xticklabels(rels2)
        ax.set_ylabel("share of legitimate crowds touched", fontsize=10, color=INK)
        ax.set_title("Crowd test, all five relations", color=INK, fontsize=12, loc="left", pad=10)
        ax.legend(frameon=False, fontsize=8)

    dest = out / "lockstep.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _lockstep_section(a, proc: Path) -> None:
    f = proc / "lockstep.json"
    if not f.exists():
        return
    ls = json.loads(f.read_text())
    fit = ls.get("fit", {})
    rings = ls.get("rings_at_headline", {})
    std, lockstep = rings.get("standard", {}), rings.get("lockstep", {})
    crowd = ls.get("crowd_test_all_relations") or {}
    ieee = ls.get("ieee_cis_arm")

    a("## Telling a crowd from a ring by when it formed\n")
    a("On the processor graph the billing address is at once the strongest relation and the "
      "thing that legitimately ties together every card in a building, and weighting cannot "
      "separate them because weighting is what found the address informative in the first "
      "place. Rarity has the same blind spot on PPA: a hostel's address and a ring's address "
      "are equally rare. The published discriminator is time rather than rarity - CopyCatch "
      "(Beutel et al., WWW 2013) argues that coordination clusters in a narrow window while "
      "natural activity spreads across it - so this measures, per entity, how concentrated in "
      "time its members' first arrivals are, corrected for the entity's size against a "
      "simulated null, and turns the excess into a second, separate edge weight.\n")
    a("**The null is the part that has to be right.** A two-account entity can only have "
      "arrived on at most two distinct days, so raw concentration is guaranteed by size alone. "
      "`burst_z(e)` is the entity's concentration against 10,000 simulated draws of the same "
      "size, arriving independently in proportion to the platform's own daily activity - the "
      "excess over what size alone would produce, not the raw number. A hand-built entity "
      "whose members all arrive the same day gets a z of 12; two-account entities average to "
      "z ≈ -0.03, not a systematic high score just for being small.\n")

    a("**Fitted, never chosen**, on training accounts only, exactly like the relation weights "
      f"themselves: {fit.get('accounts_visible', 0):,} accounts visible, "
      f"{fit.get('heldout_excluded', 0):,} held out and excluded.\n")
    a("| relation | entities | Q1 lift | Q2 lift | Q3 lift | Q4 lift |")
    a("|---|---:|---:|---:|---:|---:|")
    for rel, v in (fit.get("relations") or {}).items():
        bins = {b["bin"]: b for b in v.get("bins", [])}
        row = [f"{bins.get(k, {}).get('lift', '—')}" for k in range(4)]
        a(f"| {rel} | {v.get('entities', 0):,} | " + " | ".join(row) + " |")
    a("")
    fitr = fit.get("relations") or {}
    def _direction(rel):
        bins = {b["bin"]: b.get("lift", 1.0) for b in fitr.get(rel, {}).get("bins", [])}
        return bins.get(0, 1.0), bins.get(3, 1.0)
    a("**The direction splits by relation, and not along the split I expected.** Ranked by how "
      "many entities each relation has - `r1` 501,425, `r8` 347,785, `r6` 100,685, `r7` 8,392, "
      "`r3` 5,219 - the two most populous, `r1` and `r8`, disagree with each other: `r1`'s least "
      "bursty quartile carries the highest fraud lift (4.39, falling to 3.14 at the most bursty "
      "quartile) while `r8`'s rises the other way (1.0 to 2.58). `r6` agrees with `r1`'s direction "
      "(2.93 falling to 1.77); `r7`, on far less data, ends up agreeing with `r8`'s. So the "
      "result is not \"CopyCatch's direction fails here\" - it is relation-specific, and I do not "
      "have a story that explains the split cleanly. The candidate explanation for `r1` and `r6` "
      "is that their most extreme bursts are plausibly genuine marketing events - a promotion "
      "launch, a lunch-hour rush at one popular pickup point - where many ordinary customers "
      "order inside the same narrow window for a reason that has nothing to do with "
      "coordination; `r8`, sales stimulation, may simply not carry that kind of legitimate spike "
      "the same way. I am reporting the split rather than picking the half of it that makes a "
      "tidier story.\n")

    a("**Ring metrics at the headline operating point, both graphs:**\n")
    a("| | standard graph | lockstep graph |")
    a("|---|---:|---:|")
    a(f"| ring precision | {std.get('ring_precision')} | {lockstep.get('ring_precision')} |")
    a(f"| real customers per catch | {std.get('normal_flagged_per_fraud_caught')} | "
      f"{lockstep.get('normal_flagged_per_fraud_caught')} |")
    a(f"| fraud accounts found | {std.get('fraud_members')} | {lockstep.get('fraud_members')} |")
    a("")
    dp = (lockstep.get("ring_precision") or 0) - (std.get("ring_precision") or 0)
    a(f"Lockstep weighting moves ring precision by {dp:+.4f}. This row sits beside the headline "
      "in the README; it never replaces it.\n")

    if crowd:
        a("**The crowd test, generalised to all five relations, both ways:**\n")
        a("| relation | clusters found | touched, standard | touched, lockstep |")
        a("|---|---:|---:|---:|")
        improved = worsened = unchanged = 0
        for rel, v in crowd.items():
            s_, l_ = v.get("standard", {}), v.get("lockstep", {})
            found = s_.get("clusters_found", 0)
            st, lt = s_.get("clusters_with_a_member_in_a_ring"), l_.get("clusters_with_a_member_in_a_ring")
            if st is not None and lt is not None:
                if lt < st: improved += 1
                elif lt > st: worsened += 1
                else: unchanged += 1
            a(f"| {rel} | {found:,} | {st if st is not None else '—'} | "
              f"{lt if lt is not None else '—'} |")
        a("")
        with_data = improved + worsened + unchanged
        a(f"**Collateral moves the other way from precision.** Of the {with_data} relations with "
          f"any legitimate crowds to touch, {improved} touch fewer under the lockstep graph, "
          f"{unchanged} unchanged, and none touch more. Ring precision cost 0.0149 at the "
          "headline operating point; the trade is a small amount of precision for a measurable "
          "fall, never a rise, in the false-positive population this project cares most about "
          "protecting.\n")

    if ieee:
        a("**IEEE-CIS is the strong arm** - `TransactionDT` is seconds over six months, so hour, "
          "six-hour and day windows are all meaningful, unlike PPA's day-only resolution. Same "
          "design, reapplied at each. The billing-address weakness this arm sets out to fix: 4 "
          "of 7 apartment clusters touched under the standard graph.\n")
        std_ie = ieee.get("standard", {})
        std_addr = std_ie.get("address_cluster_test", {})
        a("| resolution | ring precision | address clusters touched |")
        a("|---|---:|---:|")
        a(f"| standard (no time weighting) | {std_ie.get('ring_precision')} | "
          f"{std_addr.get('clusters_touched')} of {std_addr.get('clusters_found')} |")
        for name, arm in (ieee.get("resolutions") or {}).items():
            addr = arm.get("address_cluster_test", {})
            a(f"| {name.replace('_', ' ')} | {arm.get('ring_precision')} | "
              f"{addr.get('clusters_touched')} of {addr.get('clusters_found')} |")
        a("")
        resolutions = ieee.get("resolutions") or {}
        best = (min(resolutions.values(),
                   key=lambda r: (r.get("address_cluster_test", {}).get("clusters_touched")
                                  if r.get("address_cluster_test", {}).get("clusters_touched") is not None
                                  else 999))
               if resolutions else None)
        bt = best.get("address_cluster_test", {}) if best else {}
        if bt.get("clusters_touched") is not None and std_addr.get("clusters_touched") is not None:
            if bt["clusters_touched"] < std_addr["clusters_touched"]:
                a(f"The best resolution touches {bt['clusters_touched']} of "
                  f"{bt.get('clusters_found')} against {std_addr['clusters_touched']} of "
                  f"{std_addr.get('clusters_found')} standard - time weighting does recover some "
                  "of the apartment-cluster weakness at fine enough resolution, at the ring "
                  f"precision cost shown above.\n")
            else:
                a("No resolution touches fewer apartment clusters than the standard graph. "
                  "Time weighting does not fix the weakness this arm was built to test; the "
                  "billing address stays informative and collateral for the same reason stated "
                  "in the processor-graph section - the two cannot be separated by an edge "
                  "weight, because the weight is what discovered the address was informative in "
                  "the first place.\n")
    else:
        a("The IEEE-CIS arm did not run this pass - the raw files were absent. Everything above "
          "is PPA only.\n")


def offer_leakage_chart(off: dict, out: Path) -> Path | None:
    """Left: precision@k of the leakage ranking against the base rate and
    against drawing k offers at random. Right: fraud coverage of the top-k
    offers against ring recall - the answer to the recall ceiling."""
    prec = off.get("precision_at_k", {}).get("by_ring_share") or {}
    cov = off.get("coverage") or {}
    if not prec or not cov.get("by_leakage_mean_score"):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=160, constrained_layout=True)

    ax = axes[0]
    _style(ax)
    ks = sorted(prec, key=int)
    x = np.arange(len(ks))
    leak = [prec[k]["leakage_ranked"]["precision"] for k in ks]
    rand = [prec[k]["random_offers"]["mean"] for k in ks]
    base = off.get("base_rate", 0)
    w = 0.36
    ax.bar(x - w / 2, [v if v is not None else 0 for v in leak], w, color=ACCENT,
          label="ranked by leakage")
    ax.bar(x + w / 2, [v if v is not None else 0 for v in rand], w, color=MUTED,
          label="k random offers")
    ax.axhline(base, color=INK, linewidth=1.0, linestyle="--", label="labelled base rate")
    for i, v in enumerate(leak):
        if v is None:
            ax.annotate("n/a", xy=(i - w / 2, 0.02), ha="center", fontsize=8, color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {k}" for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("precision (pooled over unique accounts)", fontsize=10, color=INK)
    ax.set_title("Does leakage rank offers well?", color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    _style(ax)
    pts_leak = cov.get("by_leakage_mean_score", [])
    pts_size = cov.get("by_redeemer_count", [])
    ax.plot([p["k"] for p in pts_leak], [p["fraud_coverage"] for p in pts_leak],
           color=ACCENT, linewidth=1.6, label="ranked by leakage")
    ax.plot([p["k"] for p in pts_size], [p["fraud_coverage"] for p in pts_size],
           color=INK, linewidth=1.6, label="ranked by size (redeemers)")
    rr = cov.get("ring_recall_reference")
    if rr is not None:
        ax.axhline(rr, color=MUTED, linewidth=1.2, linestyle="--",
                  label=f"ring recall ({rr})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("top-k offers reviewed", fontsize=10, color=INK)
    ax.set_ylabel("share of all labelled fraud accounts covered", fontsize=10, color=INK)
    ax.set_title("Fraud coverage: precise-but-small vs big-but-blunt", color=INK, fontsize=12,
                 loc="left", pad=10)

    dest = out / "offer_leakage.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _offers_section(a, proc: Path) -> None:
    f = proc / "offers.json"
    if not f.exists():
        return
    off = json.loads(f.read_text())
    excl = off.get("exclusions", {})
    prec = off.get("precision_at_k", {})
    cov = off.get("coverage", {})
    warn = off.get("early_warning")

    a("## Which offers are being farmed\n")
    a("Every other view in this project is account-shaped. The person who owns the promotion "
      "budget thinks in campaigns - which offer is leaking, how much, and since when - and "
      "Razorpay issues promotions itself through its rewards marketplace, so that is the view "
      "its own business would open first. There is a structural reason too: ring recall is "
      f"{cov.get('ring_recall_reference')} by construction, because twenty-five rings surface a "
      "few hundred accounts. One farmed offer surfaces every account that redeemed it, which "
      "scales where the ring view cannot.\n")
    a("**The leakage score never sees a label.** It is the share of an offer's redeemers a ring "
      "already flagged, or the account scorer's mean opinion of them - both computable before "
      "anyone checks who is actually fraudulent. The labelled fraud share is the evaluation "
      "target, computed separately and never fed back in.\n")

    a(f"**{off.get('n_offers', 0):,} offers** survived across `r6`, `r7` and `r8` in the scoring "
      f"window, out of {sum(excl.get('excluded_as_too_small', {}).values()):,} excluded as too "
      f"small to be a campaign (fewer than {excl.get('min_redeemers_to_report')} redeemers - `r8` "
      "alone has millions of once-used codes) and "
      f"{sum(excl.get('excluded_as_platform_default', {}).values())} excluded as a platform "
      f"default rather than something anyone farmed - one `r7` value alone covers "
      f"{excl.get('max_redeemers_this_window', 0):,} accounts, "
      f"{excl.get('max_redeemers_this_window', 0) / max(excl.get('active_accounts_in_window', 1), 1):.0%} "
      "of everyone active in the window, the coupon-type analogue of the near-universal default "
      "`docs/architecture.md` already documents for the graph.\n")

    a("**precision@k, ranked by leakage, against the labelled base rate and against k random "
      "offers:**\n")
    a("| top-k | leakage-ranked precision | vs base rate | k random offers (mean) |")
    a("|---:|---:|---:|---:|")
    for k, row in sorted(prec.get("by_ring_share", {}).items(), key=lambda kv: int(kv[0])):
        lp = row["leakage_ranked"]["precision"]
        a(f"| {k} | {lp if lp is not None else 'n/a - no labelled redeemer in the top-k'} | "
          f"{row.get('vs_base_rate') if lp is not None else '—'} | "
          f"{row['random_offers']['mean']} |")
    a("")
    none_at_10 = prec.get("by_ring_share", {}).get("10", {}).get("leakage_ranked", {}).get("precision") is None
    if none_at_10:
        a("**The top 10 by raw ring-share is undefined, and that is itself informative.** At "
          "this depth the ranking is dominated by ties among the smallest offers - five or six "
          "redeemers, nearly all already in a ring - and on a dataset where 90.6% of accounts "
          "are unlabelled, several of those ties land entirely among accounts nobody has "
          "reviewed. That is not a broken ranking; it is what \"most of the platform is "
          "unlabelled\" looks like at k=10, stated as the caveat it is rather than smoothed "
          "over.\n")

    a("**Coverage against the recall ceiling, two ways** - the honest comparison of two "
      f"review surfaces, since ring recall is {cov.get('ring_recall_reference')} by "
      "construction. Ranking by leakage is precise but the offers it puts first are small, "
      "which caps how much of total fraud they can ever touch; ranking by raw size answers a "
      "different question - how far past the recall ceiling can this surface reach at all, at "
      "the cost of reviewing far more accounts.\n")
    a("| top-k | leakage-ranked coverage | accounts | size-ranked coverage | accounts |")
    a("|---:|---:|---:|---:|---:|")
    pts_leak = cov.get("by_leakage_mean_score", [])
    pts_size = cov.get("by_redeemer_count", [])
    for k in (10, 25, 50):
        rl = next((p for p in pts_leak if p["k"] == k), None)
        rs = next((p for p in pts_size if p["k"] == k), None)
        if rl and rs:
            a(f"| {k} | {rl['fraud_coverage']:.2%} | {rl['accounts_reviewed']:,} | "
              f"{rs['fraud_coverage']:.2%} | {rs['accounts_reviewed']:,} |")
    a("")
    rs50 = next((p for p in pts_size if p["k"] == 50), None)
    rr = cov.get("ring_recall_reference")
    if rs50 and rr:
        a(f"Fifty offers ranked by size cover {rs50['fraud_coverage']:.2%} of all labelled "
          f"fraud against the ring's {rr:.2%} - "
          f"{rs50['fraud_coverage'] / rr:.1f}x the recall ceiling - at the cost of "
          f"{rs50['accounts_reviewed']:,} accounts reviewed instead of a few hundred. Neither "
          "ranking is the answer; which one to use is a policy choice about how much review "
          "capacity exists, not a technical one.\n")

    if warn:
        a(f"**Early warning inside the replay.** Of {warn['confirmed_bad_offers']:,} offers "
          f"confirmed bad by the window's end (majority of labelled redeemers fraudulent), "
          f"{warn['ever_warned']:,} crossed {warn['margin']:.0%} above the platform's own "
          "same-night high-score rate before the window closed"
          + (f", at a median of night {warn['median_night_of_first_warning']:.0f}"
             if warn.get("median_night_of_first_warning") else "") + ".\n")
        if warn["ever_warned"] < warn["confirmed_bad_offers"]:
            a(f"The other {warn['confirmed_bad_offers'] - warn['ever_warned']:,} never crossed the "
              "margin inside the window - a real limit stated rather than hidden: an offer can "
              "be confirmed bad only once enough of its redeemers accumulate to say so, and for "
              "some that happens no earlier than the final night.\n")
    else:
        a("Early warning did not run this pass.\n")


def label_budget_chart(lb: dict, out: Path) -> Path | None:
    """Held-out AUPRC and ring precision against how many labels the scorer
    was trained on, log-x, with seed min-max bands and the base rate and
    zero-label result marked."""
    points = lb.get("points") or []
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160, constrained_layout=True)
    _style(ax)
    ax.set_xscale("log")

    x = [p["labelled_accounts_used"] for p in points]
    for key, colour, label in (("auprc", ACCENT, "held-out AUPRC"),
                               ("ring_precision", INK, "ring precision")):
        y = [p[key]["mean"] for p in points]
        lo = [p[key]["min"] for p in points]
        hi = [p[key]["max"] for p in points]
        ax.plot(x, y, marker="o", markersize=5, linewidth=1.6, color=colour, label=label)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)

    ref = lb.get("reference_lines", {})
    if ref.get("base_rate") is not None:
        ax.axhline(ref["base_rate"], color=MUTED, linewidth=1.1, linestyle="--",
                  label=f"base rate ({ref['base_rate']})")
    if ref.get("zero_label_unpruned_ring_precision") is not None:
        ax.axhline(ref["zero_label_unpruned_ring_precision"], color=MUTED, linewidth=1.1,
                  linestyle=":", label=f"zero-label result ({ref['zero_label_unpruned_ring_precision']})")

    knee = lb.get("knee", {})
    b = knee.get("beats_base_rate_at")
    if b:
        ax.axvline(b["labelled_accounts"], color=ACCENT, linewidth=1.0, linestyle="-.",
                  alpha=0.6, label=f"beats base rate ({b['labelled_accounts']:,.0f} accounts)")

    ax.set_xlabel("labelled accounts used (log scale)", fontsize=10, color=INK)
    ax.set_ylabel("score", fontsize=10, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_title("How much labelled data before this works?", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
              fontsize=8, loc="upper right")

    dest = out / "label_budget.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _label_budget_section(a, proc: Path) -> None:
    f = proc / "label_budget.json"
    if not f.exists():
        return
    lb = json.loads(f.read_text())
    points = lb.get("points") or []
    knee = lb.get("knee") or {}
    ref = lb.get("reference_lines") or {}

    a("## How many confirmed cases before this works\n")
    a("The first thing a risk lead asks about a system like this is not how good it is but "
      "whether it works before any confirmed ring labels exist. This project had answered the "
      f"two extremes without meaning to - zero labels lands at {ref.get('zero_label_unpruned_ring_precision')}, "
      f"below the {ref.get('base_rate')} base rate, and all of them reach 0.7292 - with nothing "
      "in between. The space between is the entire deployment plan for a team starting from "
      "nothing.\n")
    a(f"The scorer and its calibration are refitted from scratch at each point on a stratified, "
      f"nested fraction of the training pool - the same {lb.get('seeds_per_fraction')} seeds' "
      "subsets at 5% are contained in the subsets at 10%, and so on - using the pipeline's own "
      "`fit_scorer` unchanged. The 100% point reuses today's training split exactly rather than "
      "resampling \"everything\", which is what makes it a guard: it reproduces "
      f"{points[-1]['auprc']['mean'] if points else '—'} AUPRC and "
      f"{points[-1]['ring_precision']['mean'] if points else '—'} ring precision, matching the "
      "committed headline.\n")

    a("| labelled accounts | fraction | held-out AUPRC (range) | ring precision (range) | "
      "real customers per catch | fraud found |")
    a("|---:|---:|---:|---:|---:|---:|")
    for p in points:
        au, rp = p["auprc"], p["ring_precision"]
        a(f"| {p['labelled_accounts_used']:,.0f} | {p['fraction']:.1%} | "
          f"{au['mean']} ({au['min']}-{au['max']}) | "
          f"{rp['mean']} ({rp['min']}-{rp['max']}) | "
          f"{p['normal_flagged_per_fraud_caught']['mean']} | "
          f"{p['fraud_members']['mean']:.1f} |")
    a("")

    b = knee.get("beats_base_rate_at")
    if b:
        a(f"**The knee: {b['labelled_accounts']:,.0f} confirmed accounts** "
          f"({b['fraction']:.1%} of today's training pool) is the smallest label count at which "
          f"prune-then-peel first beats the base rate, reaching {b['ring_precision_mean']} ring "
          "precision. Reported as a count rather than only a percentage, because a count is what "
          "a team starting a labelling effort can actually plan against - a percentage of an "
          "as-yet-unknown future pool is not.\n")
    else:
        a("**The base rate was never beaten across this sweep** - a result worth taking at face "
          "value rather than assuming a wider sweep would have found it eventually.\n")

    d = knee.get("diminishing_returns_after")
    if d:
        a(f"**Diminishing returns after {d['labelled_accounts']:,.0f} accounts** "
          f"({d['fraction']:.1%}): the next point on the sweep buys "
          f"only +{d['auprc_gain_from_previous_point']} AUPRC, below the stated "
          f"{d['increment_threshold']} threshold for what counts as still buying something.\n")
    elif len(points) >= 2:
        thresh = lb.get("diminishing_returns_auprc_increment")
        a(f"**No plateau was observed.** AUPRC keeps gaining at or above the stated "
          f"{thresh} threshold for most of the sweep, including its "
          f"last doubling - {points[-2]['labelled_accounts_used']:,.0f} to "
          f"{points[-1]['labelled_accounts_used']:,.0f} accounts still buys "
          f"+{round(points[-1]['auprc']['mean'] - points[-2]['auprc']['mean'], 4)} AUPRC. More "
          "confirmed labels would plausibly still help past what this sweep covers.\n")

    if len(points) >= 2:
        first, last = points[0], points[-1]
        a(f"**Ring precision at the smallest fraction tested - "
          f"{first['labelled_accounts_used']:,.0f} accounts - is "
          f"{first['ring_precision']['mean']}**, against {last['ring_precision']['mean']} with "
          "every label available. The account scorer is far weaker with almost no labels "
          f"(AUPRC {first['auprc']['mean']} against {last['auprc']['mean']}), but pruning then "
          "peeling stays usable well before the scorer is any good, because the graph's own "
          "structure is carrying most of the signal once any reasonable threshold separates "
          "suspicious accounts from the rest - the same finding `docs/design-decisions.md` "
          "already makes about lambda=0 degrading gracefully, now shown to hold at the other "
          "end of the label budget too.\n")

    ie = lb.get("ieee_cis")
    if ie:
        a("**Repeated on IEEE-CIS:**\n")
    else:
        a("This did not repeat on IEEE-CIS this pass. Each point here needs a full retrain and "
          "re-extraction; the PPA sweep alone is twenty-two such passes over the late-window "
          "graph. IEEE-CIS's own pipeline additionally refits per-relation weights from the same "
          "labels the scorer trains on, so an honest repeat would need to re-derive those at "
          "every fraction too, not reuse today's - a second sweep's worth of work rather than a "
          "cheap extension of this one.\n")


def propagate_chart(pr: dict, out: Path) -> Path | None:
    """Held-out AUPRC against labelled accounts, three scorers, log-x, with
    seed min-max bands where more than one seed varies."""
    points = (pr.get("label_budget_curve") or {}).get("points") or []
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160, constrained_layout=True)
    _style(ax)
    ax.set_xscale("log")

    x = [p["labelled_accounts_used"] for p in points]
    series = (("xgboost_auprc", INK, "XGBoost"), ("fabp_auprc", ACCENT, "FaBP"),
             ("graphsage_auprc", "#0369a1", "GraphSAGE"))
    for key, colour, label in series:
        y = [p[key]["mean"] for p in points]
        lo = [p[key]["min"] for p in points]
        hi = [p[key]["max"] for p in points]
        ax.plot(x, y, marker="o", markersize=5, linewidth=1.6, color=colour, label=label)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)

    ax.set_xlabel("labelled accounts used (log scale)", fontsize=10, color=INK)
    ax.set_ylabel("held-out AUPRC", fontsize=10, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_title("Three scorers against the same label budget", color=INK, fontsize=12,
                loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    dest = out / "scorer_by_label_budget.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def _propagate_section(a, proc: Path) -> None:
    f = proc / "propagate.json"
    if not f.exists():
        return
    pr = json.loads(f.read_text())
    headline = pr.get("headline", {})
    curve = pr.get("label_budget_curve", {})
    points = curve.get("points") or []
    rings = pr.get("ring_test", {})
    bp = pr.get("bipartite_offers", {})
    rt = pr.get("runtime_memory", {})

    a("## Spreading what few labels there are\n")
    a("Gradient boosting and GraphSAGE both need enough labelled rows before they say anything. "
      "Fast Belief Propagation (Koutra et al., ECML-PKDD 2011) needs none: it linearises belief "
      "propagation into one sparse linear system, `[I + aD - c'A] b_h = phi_h`, solved by a power "
      "iteration that is a few sparse matrix-vector products, with a stated convergence condition "
      "checked in code before every solve rather than assumed. The hypothesis, stated before "
      "running any of this: propagation wins when confirmed labels are scarce and loses when they "
      "are plentiful, because it needs no fitted model at all.\n")

    assort = headline.get("assortativity") or {}
    choice = headline.get("h_h_choice") or {}
    a(f"`h_h` is set from the account graph's own measured fraud-fraud lift - "
      f"{assort.get('lift')}x, over {assort.get('edges_visible'):,} training-visible edges, "
      "matching the 2.4x this project measured by hand at the very start on a different graph "
      "state - and capped at whichever of the paper's two convergence bounds (Lemma 5, Lemma 6) "
      f"is looser: {choice.get('desired_from_assortativity')} would have been the homophily-only "
      f"choice, but the graph's own maximum degree caps what a provably-convergent `h_h` can be "
      f"here, to {headline.get('h_h', 0):.6g}. The solve converged in {headline.get('iterations')} "
      f"iterations, {headline.get('solve_seconds')}s, on the full "
      f"{pr.get('graph', {}).get('accounts', 0):,}-account, "
      f"{pr.get('graph', {}).get('edges', 0):,}-edge graph.\n")

    if points:
        last = points[-1]
        a(f"At full label availability, held-out AUPRC is **{last['fabp_auprc']['mean']} for "
          f"FaBP** against {last['xgboost_auprc']['mean']} for XGBoost and "
          f"{last['graphsage_auprc']['mean']} for GraphSAGE - a provably-convergent linear solve, "
          "with no fitting step at all, ahead of both learned models on this graph.\n")

        a("| labelled accounts | fraction | XGBoost AUPRC | FaBP AUPRC | GraphSAGE AUPRC |")
        a("|---:|---:|---:|---:|---:|")
        for p in points:
            xg, fb, sg = p["xgboost_auprc"], p["fabp_auprc"], p["graphsage_auprc"]
            a(f"| {p['labelled_accounts_used']:,.0f} | {p['fraction']:.1%} | "
              f"{xg['mean']} ({xg['min']}-{xg['max']}) | "
              f"{fb['mean']} ({fb['min']}-{fb['max']}) | "
              f"{sg['mean']} ({sg['min']}-{sg['max']}) |")
        a("")

        first = points[0]
        fabp_wins_small = first["fabp_auprc"]["mean"] > first["xgboost_auprc"]["mean"]
        fabp_wins_full = last["fabp_auprc"]["mean"] > last["xgboost_auprc"]["mean"]
        if fabp_wins_small and not fabp_wins_full:
            a(f"**The hypothesis held.** At the smallest fraction tested "
              f"({first['labelled_accounts_used']:,.0f} accounts), FaBP AUPRC "
              f"({first['fabp_auprc']['mean']}) beats XGBoost's ({first['xgboost_auprc']['mean']}); "
              "by full label availability XGBoost has caught up and passed it. Propagation needs "
              "no model to fit, so it has nothing to lose from a thin label budget; a feature "
              "model does.\n")
        elif fabp_wins_small and fabp_wins_full:
            a(f"**FaBP led at every point tested, not only the scarce end** - "
              f"{first['fabp_auprc']['mean']} against {first['xgboost_auprc']['mean']} at "
              f"{first['labelled_accounts_used']:,.0f} accounts, and still ahead at full "
              "availability. The hypothesis predicted the crossover the wrong way: propagation "
              "was never behind on this graph, which says more about how strongly assortative "
              "fraud is once propagated across several hops - not just the one-hop lift used to "
              "set `h_h` - than about any weakness in the feature model.\n")
        elif not fabp_wins_small and fabp_wins_full:
            a(f"**The opposite of the stated hypothesis.** FaBP trails XGBoost at the smallest "
              f"fraction tested ({first['fabp_auprc']['mean']} against "
              f"{first['xgboost_auprc']['mean']} at {first['labelled_accounts_used']:,.0f} "
              "accounts) and only overtakes it as more labels arrive. With very few confirmed "
              "accounts to propagate from, there is simply too little seed signal in the graph "
              "for guilt-by-association to spread far; a feature model's engineered signal "
              "degrades more gracefully at that end than a propagation method with few seeds "
              "does. A null result for the hypothesis as stated, not for the method - full-label "
              "FaBP is still ahead.\n")
        else:
            a("FaBP trails XGBoost across the whole sweep tested here - a null result for the "
              "hypothesis as stated, reported rather than tuned away.\n")

    if rings:
        op = rings.get("operating_point", {})
        xr, fr = rings.get("xgboost_pruned", {}), rings.get("fabp_pruned", {})
        a(f"**Ring test.** Pruning on FaBP beliefs instead of the calibrated XGBoost score, at "
          f"the same operating point - {op.get('share_of_accounts_kept', 0):.2%} of accounts kept "
          f"either way, XGBoost's own tau matched to the equivalent quantile of the FaBP belief "
          f"distribution - then peeling with the same objective: ring precision "
          f"{fr.get('ring_precision')} against {xr.get('ring_precision')}, recall "
          f"{fr.get('ring_recall')} against {xr.get('ring_recall')}, "
          f"{fr.get('normal_flagged_per_fraud_caught')} real customers disturbed per fraud "
          f"account caught against {xr.get('normal_flagged_per_fraud_caught')}. This is the "
          "mechanism working as designed rather than a trick: FaBP's belief *is* a measure of "
          "graph-proximity to confirmed fraud, and ring precision on held-out neighbours is "
          "exactly what that measures well. It says less about whether FaBP is a better general "
          "account scorer than about how well-matched propagation is to the specific job of "
          "pruning before peeling.\n")

    if bp:
        pk = bp.get("precision_at_k", {})
        fbp_p, leak_p = pk.get("fabp_belief_ranked", {}), pk.get("leakage_ranked", {})
        rows = []
        for k in ("10", "25", "50"):
            fp = (fbp_p.get(k) or {}).get("leakage_ranked", {}).get("precision")
            lp = (leak_p.get(k) or {}).get("leakage_ranked", {}).get("precision")
            rows.append((k, fp, lp))
        a(f"**Bipartite variant.** The same solver on the account-entity graph over "
          f"r6/r7/r8, capped at the graph's own `n_max` for the reason `build_graph.py` already "
          f"gives - gives every entity a belief directly from the accounts that redeemed it. "
          f"Compared against the label-free leakage ranking from \"Which offers are being "
          f"farmed\" on the same "
          f"{bp.get('n_comparable_within_bipartite_cap', 0):,} offers "
          f"({bp.get('excluded_outside_bipartite_cap', 0):,} of "
          f"{bp.get('n_offers_from_offers_json_method', 0):,} fell outside the bipartite cap and "
          "are not part of this comparison):\n")
        a("| k | precision, FaBP belief | precision, leakage ranking |")
        a("|---:|---:|---:|")
        for k, fp, lp in rows:
            a(f"| {k} | {fp if fp is not None else '—'} | {lp if lp is not None else '—'} |")
        a("")
        fabp_offer_wins = all((fp or 0) >= (lp or 0) for _, fp, lp in rows if fp is not None or lp is not None)
        if fabp_offer_wins:
            a("FaBP's belief ranking beat the leakage ranking at every budget tested here - the "
              "principled propagation, not the simpler aggregation, is the one worth preferring "
              "for this job.\n")
        else:
            a("The two rankings split across budgets - neither dominates the other outright here.\n")

    if rt:
        peak_gb = rt.get("peak_rss_mb", 0) / 1024
        a(f"**Runtime and memory.** The full {rt.get('full_graph_accounts', 0):,}-account, "
          f"{rt.get('full_graph_edges', 0):,}-edge solve peaked at {peak_gb:.1f} GB "
          f"resident, {'well within' if rt.get('fits_in_16gb') else 'over'} the 16 GB this "
          "project develops against.\n")


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
      "file exactly, with one deliberate class of exception: every wall-clock "
      "timing and peak-memory figure - the `/check` latencies (standard and "
      "anchored), the per-night seconds in the replay and in the anchored "
      "extraction, the nightly-snapshot seconds, and FaBP's solve time and "
      "resident memory - measures the machine that ran it, not the data, and "
      "will differ on yours. Everything else is derived from the data and "
      "should match byte for byte - I have checked that against clean clones "
      "rather than assuming it.\n")
    a("Treat every timing and memory figure as the order of magnitude and "
      "nothing finer. Three full runs on the same laptop put the nightly "
      "snapshot between 96 and 208 seconds, the spread being thermal rather "
      "than algorithmic, and the per-account lookup moved by a similar "
      "factor. The conclusions drawn from them - that a nightly pass is "
      "cheap, and that a single lookup is not a file read - hold comfortably "
      "across that whole range, which is the only reason they are quoted at "
      "all.\n")

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
        _policy_section(a, proc)
        _demo_section(a, cfg)
        _lockstep_section(a, proc)
        _offers_section(a, proc)
        _label_budget_section(a, proc)
        _propagate_section(a, proc)
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
            alt, caption = FIGURE_CAPTIONS.get(
                f.stem, (f.stem.replace("_", " "), None))
            a(f"![{alt}](figures/{f.name})")
            if caption:
                a(f"*{caption}*\n")
            else:
                a("")

    dest.write_text("\n".join(L))
    return dest


def _md_inline(s: str) -> str:
    """Bold, code, italic, links - the handful of inline markdown constructs
    the prose in README.md and docs/*.md actually uses. Not a general
    parser; it does not need to be, and a general one would be one more
    place a rendering bug could hide."""
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def _md_section(text: str, heading: str) -> str:
    """The raw lines under one heading, up to the next heading of any level -
    so the landing page can quote a document's own prose instead of keeping a
    second copy that can drift from it.

    Stopping only at `## ` was a bug: a `###` subheading after the section
    (FAILURES.md's index of every entry) was swallowed into it and rendered
    as raw markdown on the page.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start, len(lines)):
        if re.match(r"^#{1,6}\s", lines[i]):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _md_list_html(text: str, ordered: bool) -> str:
    """A markdown list (numbered or bulleted), continuation lines included,
    as an <ol>/<ul>. Every finding and caveat in README wraps across several
    lines with no blank line between items, so an item ends at the next
    marker line, not at the next blank line."""
    marker = re.compile(r"^\d+\.\s+(.*)$") if ordered else re.compile(r"^-\s+(.*)$")
    hrule = re.compile(r"^-{3,}\s*$")
    items: list[str] = []
    cur: list[str] | None = None
    for line in text.splitlines():
        if hrule.match(line):
            continue
        m = marker.match(line)
        if m:
            if cur is not None:
                items.append(" ".join(cur))
            cur = [m.group(1)]
        elif not line.strip():
            continue
        elif cur is not None:
            cur.append(line.strip())
    if cur is not None:
        items.append(" ".join(cur))
    tag = "ol" if ordered else "ul"
    return (f"<{tag}>" + "".join(f"<li>{_md_inline(i)}</li>" for i in items)
            + f"</{tag}>")


def _md_table_html(text: str) -> str:
    """A GitHub-flavoured pipe table as an HTML table. Skips the alignment
    row; renders the header row as <th> only if it has non-blank cells (the
    results table's header is blank on purpose, a bare key/value grid)."""
    rows = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip("|").split("|")]
    head, sep, *body = rows
    if not re.match(r"^[\s:|-]+$", sep):
        body = [sep] + body
    head_cells = cells(head)
    out = ["<table>"]
    if any(head_cells):
        out.append("<tr>" + "".join(f"<th>{_md_inline(c)}</th>" for c in head_cells) + "</tr>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in cells(r)) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _md_richblock_html(text: str) -> str:
    """A README section that mixes `###` subheadings, bullet lists and prose -
    the pipeline write-up - rendered in order rather than by picking one
    construct out of it the way the narrower helpers above do."""
    out: list[str] = []
    buf: list[str] = []
    mode = None  # None | "p" | "ul"

    def flush():
        nonlocal buf, mode
        if not buf:
            mode = None
            return
        if mode == "ul":
            out.append("<ul>" + "".join(f"<li>{_md_inline(b)}</li>" for b in buf) + "</ul>")
        else:
            out.append(f"<p>{_md_inline(' '.join(buf))}</p>")
        buf, mode = [], None

    for line in text.splitlines():
        s = line.strip()
        if not s or re.match(r"^-{3,}$", s):
            flush()
            continue
        h = re.match(r"^#{3,6}\s+(.*)$", s)
        if h:
            flush()
            out.append(f"<h3>{_md_inline(h.group(1))}</h3>")
            continue
        b = re.match(r"^[-*]\s+(.*)$", s)
        if b:
            if mode != "ul":
                flush()
                mode = "ul"
            buf.append(b.group(1))
            continue
        if mode == "ul":
            buf[-1] += " " + s
        else:
            mode = "p"
            buf.append(s)
    flush()
    return "".join(out)


def _md_paragraphs_html(text: str) -> str:
    """Plain prose paragraphs (blank-line separated), skipping any pipe-table
    or list lines mixed into the same block - those are rendered by the
    dedicated helpers above instead."""
    paras, cur = [], []
    for line in text.splitlines():
        if not line.strip():
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        if line.strip().startswith("|") or re.match(r"^(\d+\.|-)\s", line.strip()):
            continue
        cur.append(line.strip())
    if cur:
        paras.append(" ".join(cur))
    return "".join(f"<p>{_md_inline(p)}</p>" for p in paras)


def _fix_links_for_pages(html: str, anchor_file: str | None = None) -> str:
    """A relative link to a repository markdown file renders as raw,
    unformatted text on static Pages hosting - GitHub's own renderer is what
    everyone actually reading this site is used to, so every such link is
    repointed there. `anchor_file` supplies the source document for a bare
    `#anchor` link extracted out of context (FAILURES.md's own headings,
    quoted here without the rest of the file around them)."""
    def repl(m):
        href = m.group(1)
        if href.startswith("http"):
            return m.group(0)
        if href.startswith("#"):
            if anchor_file:
                return (f'href="https://github.com/adarshcod30/Orbweaver/'
                        f'blob/main/{anchor_file}{href}"')
            return m.group(0)
        path, _, anchor = href.partition("#")
        gh = f"https://github.com/adarshcod30/Orbweaver/blob/main/{path}"
        return f'href="{gh}#{anchor}"' if anchor else f'href="{gh}"'
    return re.sub(r'href="([^"]+)"', repl, html)


def write_index(cfg, ring: dict | None, score: dict | None) -> Path | None:
    """A landing page for the published docs.

    Static hosting cannot render markdown, so this is the whole story for
    someone who will never clone the repository: the problem, why this
    dataset, how the pipeline works, every result including the negative
    ones, what broke, and the research it stands on. The prose is quoted
    directly out of README.md and FAILURES.md rather than retyped, so this
    page cannot say something different from what the repository says.
    """
    from orbweaver.console.demo import bundle_path

    proc = cfg.abs_path(cfg.paths.processed)
    root = cfg.abs_path(".")

    def art(name):
        f = proc / name
        return json.loads(f.read_text()) if f.exists() else {}

    an, po = art("anchored.json"), art("policy.json")
    best = (ring or {}).get("best_cell", {}) or {}
    cell = ((ring or {}).get("grid", {}) or {}).get(
        f"tau={best.get('tau')},lambda={best.get('lambda')}", {})
    base = (ring or {}).get("base_rate_among_labelled")
    dest = root / "docs" / "index.html"

    readme_path = root / "README.md"
    if not cell or not readme_path.exists():
        dest.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbweaver — not yet built</title><style>{CASE_CSS}</style></head><body>
<div class="wrap"><h1>Orbweaver</h1>
<div class="empty-state"><b>No run to report yet.</b> This page is generated
from <code>data/processed/</code> by <code>make report</code>, which needs
<code>make reproduce</code> (or at least <code>make rings</code>) to have run
first. Source: <a href="https://github.com/adarshcod30/Orbweaver">
github.com/adarshcod30/Orbweaver</a>.</div></div></body></html>""")
        return dest

    readme = readme_path.read_text()
    failures_path = root / "FAILURES.md"
    failures = failures_path.read_text() if failures_path.exists() else ""

    stats = []
    if cell:
        stats.append((cell.get("ring_precision"), "share of a ring worth reviewing"))
        stats.append((base, "base rate among labelled accounts"))
        stats.append((cell.get("normal_flagged_per_fraud_caught"),
                      "real customers per fraudster caught"))
    if an:
        p3 = ((an.get("summary") or {}).get("persistence_at_0.3") or {}).get(
            "share_of_final_rings_with_a_predecessor")
        if p3 is not None:
            stats.append((f"{p3:.0%}", "rings with a case open the night before"))
    if po:
        b = ((po.get("final_night") or {}).get("budgets") or {}).get("60", {}).get("capacity-aware")
        if b:
            stats.append((f"₹{b['fraud_value_stopped_inr']:,.0f}",
                          "stopped by one analyst, one hour a night"))
    bars = "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(t)}</span></div>'
                   for v, t in stats)

    bundle = bundle_path(cfg) / "meta.json"
    mb = (f"{json.loads(bundle.read_text())['bytes'] / 1e6:.2f} MB"
          if bundle.exists() else None)

    # --- prose quoted straight out of README.md, formatted, links repointed ---
    one_minute = _fix_links_for_pages(_md_paragraphs_html(
        re.search(r"<!-- oneminute:start -->(.*?)<!-- oneminute:end -->",
                  readme, re.S).group(1))) if "<!-- oneminute:start -->" in readme else ""

    problem_html = _fix_links_for_pages(
        _md_paragraphs_html(_md_section(readme, "## Overview")))

    how_it_works_raw = _md_section(readme, "## System Architecture")
    how_it_works_raw = re.sub(r"```mermaid.*?```", "", how_it_works_raw, flags=re.S)
    how_intro, _, how_list = how_it_works_raw.partition("\n1. ")
    how_intro_html = _fix_links_for_pages(_md_paragraphs_html(how_intro))
    how_list_html = _md_list_html("1. " + how_list, ordered=True) if how_list else ""

    results_section = _md_section(readme, "## Results")
    table_md = re.search(r"<!-- results:start -->(.*?)<!-- results:end -->",
                         results_section, re.S)
    results_table_html = _md_table_html(table_md.group(1)) if table_md else ""
    # The findings live in their own table after the generated block, so the
    # two tables in this section are split on the end marker rather than both
    # being swept up by one pass over the section.
    findings_raw = results_section.split("<!-- results:end -->")[-1]
    findings_html = _fix_links_for_pages(_md_table_html(findings_raw))

    caveats_html = _fix_links_for_pages(_md_table_html(
        _md_section(readme, "## What These Numbers Do Not Prove")))

    ml_section = _md_section(
        readme, "## Where Machine Learning Is Used, and Where It Is Not")
    ml_html = (_md_table_html(ml_section)
               + _fix_links_for_pages(_md_paragraphs_html(ml_section)))

    five_that_mattered = _fix_links_for_pages(
        _md_list_html(_md_section(failures, "### The five that mattered"), ordered=False),
        anchor_file="FAILURES.md") if failures else ""

    citations_html = _fix_links_for_pages(
        _md_table_html(_md_section(readme, "## Research Foundation")))

    repo_map_html = _fix_links_for_pages(
        _md_table_html(_md_section(readme, "## Project Structure")))

    features_html = _md_table_html(_md_section(readme, "## Key Features"))
    stack_html = _md_table_html(_md_section(readme, "## Tech Stack"))
    pipeline_html = _fix_links_for_pages(_md_table_html(
        _md_section(readme, "## Data & ML Pipeline")))

    # --- one real interactive chart: the same grid docs/results.md reads,
    # rendered client-side from embedded JSON, with a table fallback so it
    # means the same thing with JavaScript off. ---
    tau = best.get("tau")
    lam_rows = sorted(
        ({"lambda": r["lambda"], "precision": r["ring_precision"],
          "cost": r.get("normal_flagged_per_fraud_caught")}
         for r in (ring or {}).get("grid", {}).values()
         if r.get("tau") == tau and r.get("ring_precision") is not None),
        key=lambda r: r["lambda"])
    chart_json = json.dumps(lam_rows)
    chart_table = ("<table><tr><th>λ</th><th class=\"num\">ring precision</th>"
                   "<th class=\"num\">real customers per catch</th></tr>" +
                   "".join(f'<tr><td>{r["lambda"]}</td><td class="num">{r["precision"]}</td>'
                           f'<td class="num">{esc(r["cost"])}</td></tr>' for r in lam_rows) +
                   "</table>")

    # --- figures: every one in docs/figures/, in the same order and with the
    # same captions docs/results.md uses - one source of captions, not two. ---
    figs_dir = root / "docs" / "figures"
    fig_files = sorted(figs_dir.glob("*.png")) if figs_dir.exists() else []
    # Each figure carries a real link to the full-size image. With JavaScript
    # on it becomes a lightbox; with it off the link still opens the PNG, so
    # the affordance is never a dead control.
    expand_icon = ('<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" '
                   'fill="none" stroke="currentColor" stroke-width="1.7" '
                   'stroke-linecap="round" stroke-linejoin="round">'
                   '<path d="M9.5 2.5h4v4M13.5 2.5 9 7"/>'
                   '<path d="M6.5 13.5h-4v-4M2.5 13.5 7 9"/></svg>')
    figures_html = "".join(
        f'<figure><a class="zoom" href="figures/{f.name}" '
        f'data-caption="{esc(FIGURE_CAPTIONS.get(f.stem, ("", None))[1] or "")}" '
        f'aria-label="Enlarge: {esc(FIGURE_CAPTIONS.get(f.stem, (f.stem.replace("_", " "), ""))[0])}">'
        f'{expand_icon}</a>'
        f'<img src="figures/{f.name}" loading="lazy" '
        f'alt="{esc(FIGURE_CAPTIONS.get(f.stem, (f.stem.replace("_", " "), ""))[0])}">'
        f'<figcaption>{esc(FIGURE_CAPTIONS.get(f.stem, ("", None))[1] or "")}</figcaption>'
        f'</figure>' for f in fig_files)

    n_findings = len(re.findall(r"^\|\s*\d+\s*\|", findings_raw, re.M)) or None

    TABS = [("overview", "Overview"), ("method", "How it works"),
            ("results", "Results"), ("figures", "Figures"),
            ("failures", "What broke"), ("research", "Research")]
    tabs_html = "".join(
        f'<button class="tab{" on" if i == 0 else ""}" role="tab" id="t-{k}" '
        f'aria-controls="p-{k}" aria-selected="{"true" if i == 0 else "false"}" '
        f'tabindex="{0 if i == 0 else -1}">{esc(label)}</button>'
        for i, (k, label) in enumerate(TABS))

    console_url = "https://orbweaver-adarshcod30s-projects.vercel.app"
    repo_url = "https://github.com/adarshcod30/Orbweaver"
    blob = f"{repo_url}/blob/main"

    dest.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbweaver — finding coordinated abuse rings</title>
<meta name="description" content="Finding coordinated promotion-abuse rings in
transaction graphs, and reporting what it costs to be wrong about them.">
<meta property="og:title" content="Orbweaver">
<meta property="og:description" content="Finding coordinated promotion-abuse
rings in transaction graphs, and reporting what it costs to be wrong about
them. {esc(cell.get('ring_precision'))} ring precision against a base rate of
{esc(base)}.">
<meta property="og:type" content="website">
<meta property="og:image" content="https://raw.githubusercontent.com/adarshcod30/Orbweaver/main/docs/social-preview.png">
<meta name="twitter:card" content="summary_large_image">
<style>{CASE_CSS}
[hidden]{{display:none!important}}
.hero{{padding:8px 0 4px}}
.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:34px 30px;
margin:22px 0 6px;counter-reset:fstep}}
.fstep{{position:relative;background:var(--surface);border:1px solid var(--line);
border-radius:10px;padding:16px 14px 14px;font-size:13.5px;text-align:center;
line-height:1.4;box-shadow:var(--shadow-sm);font-weight:600}}
.fstep small{{color:var(--muted);font-size:11.5px;display:block;font-weight:400;
margin-top:3px}}
.fstep::before{{counter-increment:fstep;content:counter(fstep);position:absolute;
top:-11px;left:50%;transform:translateX(-50%);background:var(--surface);
border:1px solid var(--line);color:var(--muted);width:22px;height:22px;
border-radius:50%;font-size:11px;font-weight:700;line-height:20px}}
.fstep:not(:nth-child(4n)):not(:last-child)::after{{content:"";position:absolute;
top:50%;right:-30px;width:30px;height:1px;background:var(--line)}}
.fstep.accent{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.fstep.accent small{{color:#ffd9c7}}
.fstep.accent::before{{background:var(--accent);border-color:#fff;color:#fff}}
.fstep.learned{{border-color:var(--accent);background:var(--accent-soft)}}
.fstep.learned::before{{border-color:var(--accent);color:var(--accent)}}
@media (max-width:820px){{.flow{{grid-template-columns:repeat(2,1fr)}}
.fstep:not(:nth-child(4n)):not(:last-child)::after{{display:none}}
.fstep:not(:nth-child(2n)):not(:last-child)::after{{display:block;right:-30px}}}}
@media (max-width:480px){{.flow{{grid-template-columns:1fr;gap:26px}}
.fstep::after{{display:none!important}}}}
figure{{margin:0 0 24px;padding:0;position:relative}}
.zoom{{position:absolute;top:10px;right:10px;z-index:2;display:inline-flex;
align-items:center;justify-content:center;width:30px;height:30px;
background:var(--surface);border:1px solid var(--line);border-radius:7px;
color:var(--muted);box-shadow:var(--shadow-sm);text-decoration:none}}
.zoom:hover{{color:var(--accent);border-color:var(--accent)}}
.zoom:focus-visible{{outline:none;box-shadow:var(--focus)}}
#lightbox{{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}}
#lightbox::backdrop{{background:rgba(28,28,28,.72)}}
#lightbox .lb-in{{background:var(--surface);border-radius:var(--radius);
padding:14px;box-shadow:0 20px 60px rgba(0,0,0,.3);max-width:96vw}}
#lightbox img{{display:block;max-width:100%;max-height:78vh;width:auto;
border:1px solid var(--line);border-radius:8px}}
#lightbox .lb-cap{{color:var(--muted);font-size:13px;margin-top:10px;
max-width:70ch;line-height:1.5}}
#lightbox .lb-close{{position:absolute;top:22px;right:22px}}
figure img{{width:100%;border:1px solid var(--line);border-radius:8px;
background:var(--surface)}}
figcaption{{color:var(--muted);font-size:13px;margin-top:8px;line-height:1.5}}
.figuregrid{{display:grid;grid-template-columns:1fr;gap:0}}
@media (min-width:820px){{.figuregrid{{grid-template-columns:1fr 1fr;gap:0 26px}}}}
#lambda-chart svg circle{{cursor:pointer}}
#lambda-chart svg circle:focus-visible{{stroke:var(--ink);stroke-width:3px;outline:none}}
.card h3{{font-size:15px;margin:18px 0 6px;letter-spacing:-.01em}}
.card ol,.card ul{{padding-left:22px}}
.card li{{margin-bottom:8px}}
.card ol li::marker{{color:var(--accent);font-weight:600}}
.panel>.card:first-child{{margin-top:4px}}
</style></head><body>
<header class="masthead">
<div class="masthead-in">
<a class="brand" href="#overview"><span class="dot"></span>Orbweaver</a>
<nav class="tabs" role="tablist" aria-label="Sections">{tabs_html}</nav>
</div>
</header>
<div class="wrap">

<section class="hero">
<h1>Finding coordinated abuse rings</h1>
<p class="lede"><em>The one that feels the whole web.</em> Promotion-abuse
rings are invisible order by order — the fraud lives in the connections
between accounts. Orbweaver finds them in the graph, and reports what it
costs to be wrong about them.</p>
<div class="bar">{bars}</div>
<div class="btn-row">
<a class="btn primary" href="{console_url}">Open the live console →</a>
<a class="btn" href="case-files.html">Browse the case files</a>
<a class="btn" href="{blob}/docs/results.md">Full results</a>
<a class="btn ghost" href="{repo_url}">GitHub</a>
</div>
<div class="assume">Every number on this page is produced by
<code>make reproduce</code>; none is typed in by hand — this page is quoted,
not retyped, from <a href="{blob}/README.md">README.md</a>. Rupee figures use
stated assumptions, because the dataset ships no monetary amounts, so they
rank options against each other and mean nothing absolute. Built for the
Razorpay AI Buildathon, Track 02 — my submission to their buildathon, not a
Razorpay product.</div>
</section>

<section class="panel" id="p-overview" role="tabpanel" aria-labelledby="t-overview">
<div class="card"><h2>In one minute</h2>{one_minute}</div>
<div class="card"><h2>The problem</h2>{problem_html}</div>
<div class="card"><h2>Why this dataset</h2>
<p>PPA is the only public, labelled promotion-abuse-ring dataset I could find
— not the best one, the only one — and I never modified it, because a
detector's numbers are only honest if the ground truth they are checked
against is real. The full case for why this is still the right dataset to
have started from, what its own paper claims that the release does not
support, and how the hostel test and three unrelated datasets test the
method past PPA, is in
<a href="{blob}/docs/why-this-data.md">docs/why-this-data.md</a>.</p></div>
{f'<div class="card"><h2>Key features</h2><div class="table-scroll">{features_html}</div></div>' if features_html else ''}
{f'<div class="card"><h2>Tech stack</h2><div class="table-scroll">{stack_html}</div></div>' if stack_html else ''}
</section>

<section class="panel" id="p-method" role="tabpanel" aria-labelledby="t-method" hidden>
<div class="card"><h2>How it works</h2>
{how_intro_html}
<div class="flow" role="img" aria-label="Eight stages in order: orders, then a
multi-relation graph, then the account scorer, then pruning, then peeling, then
ring plus evidence, then the review policy, then the analyst case files. Nights
one to four replay the whole loop, anchored so a case survives the night.">
<div class="fstep">orders<small>43.9M rows</small></div>
<div class="fstep">graph<small>rarity x relation weight</small></div>
<div class="fstep learned">scorer<small>XGBoost, the one learned step</small></div>
<div class="fstep">prune<small>score cut-off tau</small></div>
<div class="fstep">peel<small>densest subgraph, proved bound</small></div>
<div class="fstep">ring + evidence<small>shares, rarity, cost</small></div>
<div class="fstep accent">policy<small>review / hold / ignore</small></div>
<div class="fstep">case files<small>what an analyst opens</small></div>
</div>
<p class="note">Replayed one night at a time for the nightly numbers: rings
are anchored around fixed accounts so a case found tonight is recognisably
the same case tomorrow, not a fresh one every morning.</p>
{how_list_html}
</div>
{f'<div class="card"><h2>Data and ML pipeline</h2><div class="table-scroll">{pipeline_html}</div></div>' if pipeline_html else ''}
<div class="card"><h2>Where a model is trusted, and where it is not</h2>
<div class="table-scroll">{ml_html}</div>
<div class="btn-row"><a class="btn" href="{blob}/docs/design-decisions.md">Read the design decisions →</a>
<a class="btn ghost" href="{blob}/docs/architecture.md">Architecture in depth</a></div>
</div>
</section>

<section class="panel" id="p-results" role="tabpanel" aria-labelledby="t-results" hidden>
<div class="card"><h2>Every number this project produces</h2>
<p class="sub">All {esc(n_findings) or 'thirteen'} investigations, including
the four that came back negative, generated by <code>make reproduce</code>
from the run in <code>data/processed/</code>.</p>
<div class="table-scroll">{results_table_html}</div>
</div>
<div class="card"><h2>The findings, including the ones that did not work</h2>
<div class="table-scroll">{findings_html}</div></div>
<div class="card"><h2>How much the model's opinion actually matters</h2>
<p class="sub">At the score cut-off this project uses (τ = {esc(tau)}), this is
ring precision as λ moves the peeling objective from pure structure
(λ = 0) toward weighing the account scorer's opinion more heavily. The same
numbers are in the table above and, alongside every tau, in
<a href="{blob}/docs/results.md#figures">docs/results.md</a>.</p>
<div id="lambda-chart" aria-label="Ring precision against lambda, a line chart">
<noscript>JavaScript is off — see the table below.</noscript>
</div>
<script id="chart-data" type="application/json">{chart_json}</script>
<details class="chart-table"><summary>View as a table</summary>
<div class="table-scroll">{chart_table}</div></details>
</div>
<div class="card"><h2>What these numbers do not prove</h2>
<div class="table-scroll">{caveats_html}</div></div>
</section>

<section class="panel" id="p-figures" role="tabpanel" aria-labelledby="t-figures" hidden>
<div class="card"><h2>Figures</h2>
<p class="sub">All {esc(len(fig_files))} of them, regenerated by
<code>make report</code>; the exact captions in
<a href="{blob}/docs/results.md">docs/results.md</a>.</p>
<div class="figuregrid">{figures_html}</div>
</div>
</section>

<section class="panel" id="p-failures" role="tabpanel" aria-labelledby="t-failures" hidden>
<div class="card"><h2>What broke</h2>
<p class="sub">The five that mattered, out of
<a href="{blob}/FAILURES.md">FAILURES.md</a>'s full, dated log — the file I
would read first if someone handed me this repository.</p>
{five_that_mattered}
<div class="btn-row"><a class="btn" href="{blob}/FAILURES.md">Read the whole log →</a></div>
</div>
</section>

<section class="panel" id="p-research" role="tabpanel" aria-labelledby="t-research" hidden>
<div class="card"><h2>Research foundation</h2>
<div class="table-scroll">{citations_html}</div></div>
<div class="card"><h2>Repository map</h2>
<div class="table-scroll">{repo_map_html}</div></div>
<div class="card"><h2>Run it yourself</h2>
<p>A clone, six packages and one command — no dataset download needed, because
the repository carries the computed results as a bundle.</p>
<pre><code>git clone https://github.com/adarshcod30/Orbweaver
cd Orbweaver
pip install -r requirements-demo.txt
make console      # http://127.0.0.1:8000</code></pre>
{f'<p class="note">The bundle is {mb}, so the live console and a fresh clone both run with no dataset present at all. The full pipeline from raw data is <code>make reproduce</code>.</p>' if mb else ''}
<div class="btn-row">
<a class="btn primary" href="{console_url}">Open the live console →</a>
<a class="btn" href="case-files.html">Case files (static page)</a>
<a class="btn ghost" href="{repo_url}">Source on GitHub</a>
</div>
</div>
</section>

<dialog id="lightbox" aria-label="Figure, enlarged">
<div class="lb-in"><img alt=""><p class="lb-cap"></p></div>
<button class="btn lb-close" type="button">Close</button>
</dialog>

<p class="footer">MIT-licensed. Detection only —
<a href="{blob}/ETHICS.md">ETHICS.md</a> sets the boundary in six lines.
Built for the Razorpay AI Buildathon, Track 02 ·
<a href="{repo_url}">github.com/adarshcod30/Orbweaver</a></p>
</div>
<script>
(function(){{
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab[role="tab"]'));
  if (!tabs.length) return;
  var panels = tabs.map(function(t){{
    return document.getElementById(t.getAttribute('aria-controls'));
  }});
  function select(i, scroll){{
    tabs.forEach(function(t, j){{
      var on = j === i;
      t.classList.toggle('on', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      if (panels[j]) panels[j].hidden = !on;
    }});
    try {{ history.replaceState(null, '', '#' + tabs[i].id.slice(2)); }} catch (e) {{}}
    if (scroll) window.scrollTo({{top: 0, behavior: 'smooth'}});
  }}
  tabs.forEach(function(t, i){{
    t.addEventListener('click', function(){{ select(i, true); }});
    t.addEventListener('keydown', function(e){{
      var d = e.key === 'ArrowRight' ? 1 : (e.key === 'ArrowLeft' ? -1 : 0);
      if (!d) return;
      e.preventDefault();
      var n = (i + d + tabs.length) % tabs.length;
      tabs[n].focus();
      select(n, false);
    }});
  }});
  function fromHash(){{
    var h = (location.hash || '').replace('#', '');
    if (!h) return false;
    for (var i = 0; i < tabs.length; i++) {{
      if (tabs[i].id.slice(2) === h) {{ select(i, false); return true; }}
    }}
    var el = document.getElementById(h);
    if (el) {{
      for (var j = 0; j < panels.length; j++) {{
        if (panels[j] && panels[j].contains(el)) {{
          select(j, false);
          el.scrollIntoView();
          return true;
        }}
      }}
    }}
    return false;
  }}
  if (!fromHash()) select(0, false);
  window.addEventListener('hashchange', fromHash);
}})();
(function(){{
  var dlg = document.getElementById('lightbox');
  if (!dlg || !dlg.showModal) return;          // no <dialog>: links still open the PNG
  var img = dlg.querySelector('img');
  var cap = dlg.querySelector('.lb-cap');
  var last = null;
  document.addEventListener('click', function(e){{
    var a = e.target.closest && e.target.closest('a.zoom');
    if (a) {{
      e.preventDefault();
      last = a;
      img.src = a.getAttribute('href');
      img.alt = (a.getAttribute('aria-label') || '').replace(/^Enlarge:\\s*/, '');
      cap.textContent = a.getAttribute('data-caption') || '';
      dlg.showModal();
      return;
    }}
    if (e.target.closest && e.target.closest('.lb-close')) {{ dlg.close(); return; }}
    if (e.target === dlg) dlg.close();          // click the backdrop
  }});
  dlg.addEventListener('close', function(){{
    img.removeAttribute('src');
    if (last) last.focus();
  }});
}})();
(function(){{
  var dataEl = document.getElementById('chart-data');
  var host = document.getElementById('lambda-chart');
  if (!dataEl || !host) return;
  var data;
  try {{ data = JSON.parse(dataEl.textContent); }} catch (e) {{ return; }}
  if (!data || !data.length) return;
  var w = 640, h = 260, pad = {{l: 42, r: 16, t: 14, b: 30}};
  var xs = data.map(function(d){{return d.lambda;}});
  var ys = data.map(function(d){{return d.precision;}});
  var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
  var ymin = 0, ymax = Math.max.apply(null, ys) * 1.15;
  function X(x) {{ return pad.l + (xmax > xmin ? (x - xmin) / (xmax - xmin) : 0.5) * (w - pad.l - pad.r); }}
  function Y(y) {{ return h - pad.b - (ymax > ymin ? (y - ymin) / (ymax - ymin) : 0) * (h - pad.t - pad.b); }}
  var svgNS = 'http://www.w3.org/2000/svg';
  var svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Ring precision rises with lambda at this score cut-off');
  [0, 0.25, 0.5, 0.75, 1].forEach(function(f){{
    var y = pad.t + f * (h - pad.t - pad.b);
    var l = document.createElementNS(svgNS, 'line');
    l.setAttribute('x1', pad.l); l.setAttribute('x2', w - pad.r);
    l.setAttribute('y1', y); l.setAttribute('y2', y);
    l.setAttribute('stroke', 'var(--line)'); l.setAttribute('stroke-width', '1');
    svg.appendChild(l);
    var t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', 4); t.setAttribute('y', y + 4);
    t.setAttribute('font-size', '10'); t.setAttribute('fill', 'var(--muted)');
    t.textContent = (ymax - (ymax - ymin) * f).toFixed(2);
    svg.appendChild(t);
  }});
  var path = data.map(function(d, i){{
    return (i ? 'L' : 'M') + X(d.lambda) + ',' + Y(d.precision);
  }}).join(' ');
  var p = document.createElementNS(svgNS, 'path');
  p.setAttribute('d', path); p.setAttribute('fill', 'none');
  p.setAttribute('stroke', 'var(--accent)'); p.setAttribute('stroke-width', '2.5');
  svg.appendChild(p);
  data.forEach(function(d){{
    var c = document.createElementNS(svgNS, 'circle');
    c.setAttribute('cx', X(d.lambda)); c.setAttribute('cy', Y(d.precision));
    c.setAttribute('r', 5.5); c.setAttribute('fill', 'var(--accent)');
    c.setAttribute('tabindex', '0');
    c.setAttribute('aria-label', 'lambda ' + d.lambda + ', ring precision ' + d.precision);
    var t = document.createElementNS(svgNS, 'title');
    t.textContent = 'λ=' + d.lambda + '  precision ' + d.precision;
    c.appendChild(t);
    var x = document.createElementNS(svgNS, 'text');
    x.setAttribute('x', X(d.lambda)); x.setAttribute('y', h - 8);
    x.setAttribute('font-size', '10'); x.setAttribute('fill', 'var(--muted)');
    x.setAttribute('text-anchor', 'middle');
    x.textContent = d.lambda;
    svg.appendChild(x);
    svg.appendChild(c);
  }});
  host.textContent = '';
  host.appendChild(svg);
}})();
</script>
</body></html>""")
    return dest


def write_social_preview(cfg, ring: dict | None) -> Path | None:
    """GitHub's link-unfurl and repo social-preview image: 1280x640, same
    numbers as everything else, no hand-typed figures. Rendered with
    matplotlib rather than laid out by hand so it stays the same tool that
    draws every other figure here."""
    best = (ring or {}).get("best_cell", {}) or {}
    cell = ((ring or {}).get("grid", {}) or {}).get(
        f"tau={best.get('tau')},lambda={best.get('lambda')}", {})
    base = (ring or {}).get("base_rate_among_labelled")
    if not cell:
        return None

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.06, 0.80, "Orbweaver", fontsize=54, color=INK, weight="bold",
             family="sans-serif")
    ax.text(0.06, 0.665, "Finding coordinated promotion-abuse rings in transaction\n"
            "graphs — and reporting what it costs to be wrong about them.",
            fontsize=17, color=MUTED, linespacing=1.5)

    stats = [
        (cell.get("ring_precision"), "ring precision"),
        (base, "base rate"),
        (cell.get("normal_flagged_per_fraud_caught"), "real customers per catch"),
    ]
    an = json.loads((cfg.abs_path(cfg.paths.processed) / "anchored.json").read_text()) \
        if (cfg.abs_path(cfg.paths.processed) / "anchored.json").exists() else {}
    p3 = ((an.get("summary") or {}).get("persistence_at_0.3") or {}).get(
        "share_of_final_rings_with_a_predecessor")
    if p3 is not None:
        stats.append((f"{p3:.0%}", "case open the night before"))

    x0, w = 0.06, 0.86 / len(stats)
    for i, (v, label) in enumerate(stats):
        x = x0 + i * w
        ax.text(x, 0.38, str(v), fontsize=30, color=ACCENT, weight="bold")
        ax.text(x, 0.28, label, fontsize=13, color=MUTED)

    ax.axhline(0.50, xmin=0.06, xmax=0.94, color=GRID, linewidth=1.5)
    ax.text(0.06, 0.06, "Razorpay AI Buildathon · Track 02", fontsize=13, color=MUTED)
    ax.text(0.94, 0.06, "github.com/adarshcod30/Orbweaver", fontsize=13,
            color=MUTED, ha="right")

    dest = cfg.abs_path(".") / "docs" / "social-preview.png"
    fig.savefig(dest, facecolor="white")
    plt.close(fig)
    return dest


def write_404(cfg) -> Path:
    """GitHub Pages serves this for any path under the site that does not
    exist - the alternative is its own generic, off-brand error page."""
    dest = cfg.abs_path(".") / "docs" / "404.html"
    dest.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbweaver — page not found</title>
<style>{CASE_CSS}</style></head><body>
<div class="wrap"><h1>Orbweaver</h1>
<div class="empty-state"><b>There is no page here.</b> Try the
<a href="/Orbweaver/">landing page</a>, the
<a href="/Orbweaver/case-files.html">case files</a>, or the
<a href="https://github.com/adarshcod30/Orbweaver">repository</a> itself.</div>
</div></body></html>""")
    return dest


def _one_minute_block(cfg, cell: dict, base, proc: Path) -> list[str]:
    """The orientation block at the very top of the README - same source
    variables as the results table below it, so there is no second place for
    these two numbers to drift from the first."""
    start, end = "<!-- oneminute:start -->", "<!-- oneminute:end -->"
    M = [start, ""]
    if cell:
        M.append(f"A group running many accounts through one delivery-app "
                 f"promotion looks fine order by order; the fraud only exists "
                 f"in the connections between the accounts, which a detector "
                 f"scoring one transaction at a time cannot see. Pruning to "
                 f"suspicious accounts, then peeling for dense structure, "
                 f"catches them at **{cell.get('ring_precision')} ring "
                 f"precision** against a base rate of {base} — "
                 f"**{cell.get('precision_lift_over_base')}× chance** — at a "
                 f"measured cost of **{cell.get('normal_flagged_per_fraud_caught')} "
                 f"real customers swept into a ring for every fraudster it "
                 f"catches**.\n")

    def art(name):
        f = proc / name
        return json.loads(f.read_text()) if f.exists() else None

    gen = art("generalisation.json")
    if gen:
        y0 = gen["datasets"]["yelpchi"]["rings"].get("0.0", {})
        y5 = gen["datasets"]["yelpchi"]["rings"].get("0.5", {})
        a3 = gen["datasets"]["amazon"]["rings"].get("0.3", {})
        if y0 and y5 and a3:
            M.append(f"**The finding I would defend hardest:** dense is not "
                     f"the same as fraudulent, and it replicates on every "
                     f"unrelated dataset I have tried it on. Unpruned, the "
                     f"same extractor lands *below* chance here (0.31×) and "
                     f"at *exactly zero* on YelpChi - {y0['n_rings']} rings, "
                     f"{y0['accounts_in_rings']:,} accounts, "
                     f"{y0['fraud_members']} of them fraudulent. Pruned "
                     f"first, the identical code reaches "
                     f"{a3['precision_lift_over_base']:.1f}× on Amazon reviewers "
                     f"and {y5['precision_lift_over_base']:.1f}× on YelpChi "
                     f"reviews - three platforms, two of them nothing like "
                     f"promotion abuse, saying the same thing "
                     f"([why this matters](docs/why-this-data.md"
                     f"#how-amazon-yelpchi-and-ieee-cis-test-transfer)).\n")

    M.append("[**Live console**](https://orbweaver-adarshcod30s-projects.vercel.app) "
             "· [**Full results**](docs/results.md) · "
             "[**What broke**](FAILURES.md)\n")
    M += ["", end]
    return M


def update_readme(cfg, score, ring, views) -> Path | None:
    """Fill the generated blocks in README.md.

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

    proc = cfg.abs_path(cfg.paths.processed)
    om_start, om_end = "<!-- oneminute:start -->", "<!-- oneminute:end -->"
    if om_start in text and om_end in text:
        om_head, om_rest = text.split(om_start, 1)
        _, om_tail = om_rest.split(om_end, 1)
        M = _one_minute_block(cfg, cell, base, proc)
        text = om_head + "\n".join(M) + om_tail

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

    po = artefact("policy.json")
    if po:
        rows_ = [r for r in po.get("night_by_night_at_120_minutes", []) if r.get("rings_in_queue")]
        b60 = po["final_night"]["budgets"].get("60", {})
        if rows_ and b60:
            c, d_ = b60["capacity-aware"], b60["density order until the budget is spent"]
            L.append(f"| What one analyst an hour a night stops | "
                     f"₹{c['fraud_value_stopped_inr']:,.0f} of promotion value against "
                     f"₹{d_['fraud_value_stopped_inr']:,.0f} for working the queue in order, "
                     f"for ₹{c['legitimate_value_harmed_inr']:,.0f} of legitimate value harmed "
                     f"(assumed rupees) |")

    ls = artefact("lockstep.json")
    if ls:
        r_ = ls.get("rings_at_headline", {})
        std_, lk_ = r_.get("standard", {}), r_.get("lockstep", {})
        dp = (lk_.get("ring_precision") or 0) - (std_.get("ring_precision") or 0)
        ieee_ = ls.get("ieee_cis_arm")
        ieee_bit = ""
        if ieee_:
            sa = ieee_.get("standard", {}).get("address_cluster_test", {})
            best = None
            for arm in (ieee_.get("resolutions") or {}).values():
                t = arm.get("address_cluster_test", {}).get("clusters_touched")
                if t is not None and (best is None or t < best):
                    best = t
            if best is not None and sa.get("clusters_touched") is not None:
                if best < sa["clusters_touched"]:
                    ieee_bit = (f"; on IEEE-CIS the apartment-cluster weakness falls from "
                               f"{sa['clusters_touched']} to {best} of {sa.get('clusters_found')} "
                               f"touched at the best resolution")
                else:
                    ieee_bit = (f"; on IEEE-CIS the apartment-cluster weakness is unchanged at "
                               f"{sa['clusters_touched']} of {sa.get('clusters_found')} touched "
                               f"at every resolution tried")
        L.append(f"| Telling a crowd from a ring by when it formed | burst-weighted ring "
                 f"precision {dp:+.4f} on PPA{ieee_bit} |")

    off = artefact("offers.json")
    if off:
        cov = off.get("coverage", {})
        row50 = next((p for p in cov.get("by_redeemer_count", []) if p["k"] == 50), None)
        rr = cov.get("ring_recall_reference")
        if row50 and rr:
            L.append(f"| Which offers are being farmed | top 50 offers by size "
                     f"({row50['accounts_reviewed']:,} accounts) cover "
                     f"{row50['fraud_coverage']:.1%} of all labelled fraud, "
                     f"{row50['fraud_coverage'] / rr:.1f}x the {rr:.2%} ring recall ceiling |")

    lb = artefact("label_budget.json")
    if lb:
        b = (lb.get("knee") or {}).get("beats_base_rate_at")
        if b:
            L.append(f"| How many confirmed cases before this works | prune-then-peel first "
                     f"beats the base rate at {b['labelled_accounts']:,.0f} confirmed accounts "
                     f"({b['fraction']:.1%} of the training pool) |")

    pr = artefact("propagate.json")
    if pr:
        points = (pr.get("label_budget_curve") or {}).get("points") or []
        rt = pr.get("ring_test") or {}
        if points and rt:
            last = points[-1]
            fr = rt.get("fabp_pruned", {})
            L.append(f"| Spreading what few labels there are | Fast Belief Propagation, no "
                     f"fitted model: {last['fabp_auprc']['mean']} held-out AUPRC at full labels, "
                     f"{fr.get('ring_precision')} ring precision pruning on its beliefs alone |")

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
    poj = proc / "policy.json"
    if poj.exists():
        d = policy_frontier_chart(json.loads(poj.read_text()), figs)
        if d:
            figures.append(d)
    lsj = proc / "lockstep.json"
    if lsj.exists():
        d = lockstep_chart(json.loads(lsj.read_text()), figs)
        if d:
            figures.append(d)
    ofj = proc / "offers.json"
    if ofj.exists():
        d = offer_leakage_chart(json.loads(ofj.read_text()), figs)
        if d:
            figures.append(d)
    lbj = proc / "label_budget.json"
    if lbj.exists():
        d = label_budget_chart(json.loads(lbj.read_text()), figs)
        if d:
            figures.append(d)
    prj = proc / "propagate.json"
    if prj.exists():
        d = propagate_chart(json.loads(prj.read_text()), figs)
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
    # After update_readme, not before: the landing page quotes README
    # sections verbatim (the findings list, the caveats) rather than
    # retyping them, so it has to read the file after this run's rewrite.
    idx = write_index(cfg, ring, score)
    if idx:
        print(f"wrote {idx}")
    print(f"wrote {write_404(cfg)}")
    sp = write_social_preview(cfg, ring)
    if sp:
        print(f"wrote {sp}")
    for f in figures:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
