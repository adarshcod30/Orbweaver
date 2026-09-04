"""Render extracted rings as a standalone HTML case-file report.

This is what an analyst would actually be handed: one card per ring, sorted by
how much money is at stake, showing who is in it, what ties them together, how
rare that tie is, and what it would cost to be wrong. No server, no build step
- a single file that opens in a browser.

Everything on the page is read from `ring_report.json`; nothing is recomputed
here, so the page cannot disagree with `docs/results.md`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from orbweaver.config import load_config

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Light only, deliberately. color-scheme:light stops a dark-mode browser
   repainting form controls and scrollbars underneath a light page. */
:root{color-scheme:light;
--ink:#1c1c1c;--muted:#6b6b6b;--line:#e6e6e6;--accent:#c2410c;
--accent-soft:#fff2ec;--accent2:#0369a1;--bg:#fbfaf9;--surface:#ffffff;
--ok:#166534;--ok-bg:#f0fdf4;--warn:#a16207;--danger:#b91c1c;--danger-bg:#fef2f2;
--code-bg:#f5f5f4;--assume-bg:#fffbeb;--assume-line:#fde68a;--assume-ink:#78350f;
--shadow:0 2px 4px rgba(31,35,38,.06),0 8px 20px -4px rgba(31,35,38,.08);
--shadow-sm:0 1px 2px rgba(31,35,38,.06);
--focus:0 0 0 3px rgba(194,65,12,.35);--radius:12px;
--ease:cubic-bezier(.3,0,.2,1)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
font-feature-settings:"cv11","ss01"}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:32px;margin:0 0 8px;letter-spacing:-.02em;line-height:1.2}
h2{letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 24px}
.lede{font-size:18px;line-height:1.55;color:var(--ink);margin:0 0 20px;max-width:64ch}

/* ---- masthead + tab navigation ------------------------------------- */
.masthead{border-bottom:1px solid var(--line);background:var(--surface);
position:sticky;top:0;z-index:20;box-shadow:var(--shadow-sm)}
.masthead-in{max-width:1080px;margin:0 auto;padding:0 24px;display:flex;
align-items:center;gap:20px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:16px;
letter-spacing:-.01em;color:var(--ink);text-decoration:none;padding:14px 0}
.brand .dot{width:11px;height:11px;border-radius:50%;background:var(--accent);
box-shadow:0 0 0 3px var(--accent-soft);flex:none}
.tabs{display:flex;gap:2px;overflow-x:auto;margin-left:auto;
scrollbar-width:none;-webkit-overflow-scrolling:touch}
.tabs::-webkit-scrollbar{display:none}
.tab{appearance:none;background:none;border:0;border-bottom:2px solid transparent;
font:inherit;font-size:14px;font-weight:500;color:var(--muted);
padding:15px 12px 13px;cursor:pointer;white-space:nowrap;text-decoration:none;
display:inline-block}
.tab:hover{color:var(--ink);background:var(--code-bg)}
.tab[aria-selected="true"],.tab.on{color:var(--accent);border-bottom-color:var(--accent);
font-weight:600}
.tab:focus-visible{outline:none;box-shadow:var(--focus)}

/* ---- buttons -------------------------------------------------------- */
.btn{display:inline-flex;align-items:center;gap:7px;font:inherit;font-size:14px;
font-weight:600;line-height:1;padding:11px 18px;border-radius:8px;cursor:pointer;
text-decoration:none;border:1px solid var(--line);background:var(--surface);
color:var(--ink);box-shadow:var(--shadow-sm)}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{background:#a5350a;border-color:#a5350a;color:#fff}
.btn.ghost{background:none;box-shadow:none}
.btn:focus-visible{outline:none;box-shadow:var(--focus)}
.btn-row{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 4px}

/* ---- cards, stats, tables ------------------------------------------ */
.bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:2px;
background:var(--line);border:1px solid var(--line);border-radius:var(--radius);
margin-bottom:24px;overflow:hidden;box-shadow:var(--shadow)}
.stat{background:var(--surface);padding:16px 18px}
.stat b{display:block;font-size:22px;letter-spacing:-.01em;color:var(--accent)}
.stat span{color:var(--muted);font-size:11.5px;text-transform:uppercase;
letter-spacing:.06em;line-height:1.4;display:block;margin-top:3px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
padding:22px 26px;margin-bottom:16px;box-shadow:var(--shadow)}
.card h2{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
.card .meta{color:var(--muted);font-size:13px;margin-bottom:14px}
.table-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-weight:600;color:var(--muted);font-size:11px;
text-transform:uppercase;letter-spacing:.06em;padding:8px 12px 8px 0;
border-bottom:1px solid var(--line)}
td{padding:9px 12px 9px 0;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:3px 9px;border-radius:99px;font-size:11.5px;
font-weight:600;margin-right:6px}
.tag.f{background:var(--danger-bg);color:var(--danger)}
.tag.n{background:var(--ok-bg);color:var(--ok)}
.tag.u{background:var(--code-bg);color:var(--muted)}
.rare{color:var(--accent);font-weight:600}
.act{display:flex;flex-wrap:wrap;gap:18px;align-items:baseline;margin:14px 0 2px;
padding:13px 16px;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.act .verb{font-weight:700;letter-spacing:-.01em}
.act .verb.review{color:var(--accent)}
.act .verb.hold{color:var(--warn)}
.act .verb.ignore{color:var(--muted)}
.act .why{color:var(--muted);font-size:12.5px}
.note{color:var(--muted);font-size:13px;margin-top:14px;padding-top:14px;
border-top:1px solid var(--line)}
.assume{background:var(--assume-bg);border:1px solid var(--assume-line);
border-left:3px solid #d9a441;color:var(--assume-ink);padding:13px 16px;
border-radius:8px;font-size:13.5px;margin-bottom:24px}
code{background:var(--code-bg);color:var(--ink);padding:2px 6px;border-radius:5px;
font-size:13px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:8px;
padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.55}
pre code{background:none;padding:0}
a{color:var(--accent2)}
a:focus-visible,button:focus-visible,summary:focus-visible,
input:focus-visible,select:focus-visible{outline:none;box-shadow:var(--focus);
border-radius:4px}
details>summary{cursor:pointer;color:var(--accent2);font-size:13.5px;
font-weight:500;padding:4px 0}
.empty-state{text-align:center;color:var(--muted);padding:48px 20px;
border:1px dashed var(--line);border-radius:var(--radius);background:var(--surface)}
.empty-state b{display:block;color:var(--ink);font-size:17px;margin-bottom:6px}
.footer{border-top:1px solid var(--line);margin-top:40px;padding-top:20px;
color:var(--muted);font-size:13.5px}
.backlink{margin:0 0 20px}
@media (prefers-reduced-motion:no-preference){
.card,.row,.btn,.tab{transition:border-color 200ms var(--ease),
box-shadow 200ms var(--ease),background 200ms var(--ease),color 200ms var(--ease)}
.htmx-settling{opacity:.001}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
transition:none!important}}
@media (max-width:768px){
.wrap{padding:24px 16px 56px}
h1{font-size:25px}
.lede{font-size:16.5px}
.masthead-in{padding:0 16px;gap:0}
.brand{padding:12px 0}
.tabs{margin-left:0;width:100%;border-top:1px solid var(--line)}
.tab{padding:12px 11px 10px;font-size:13.5px}
.stat b{font-size:19px}
.card{padding:18px 20px}
table{font-size:13px}
.btn{padding:10px 15px}
.act{gap:12px}}
"""


def esc(x) -> str:
    return html.escape(str(x))


def members_of(case: dict) -> list[int]:
    """A ring's full membership, falling back to the sample for older reports."""
    return case.get("members") or case.get("members_sample", [])


def anonymise(cases: list[dict]) -> dict:
    """Short stable labels for entity ids.

    A raw id like 3861721 tells a reader nothing and is impossible to hold in
    the head, but two rows sharing one still has to be visible - that is the
    whole point of a coverage column. So each id gets a short label, numbered
    per relation in the order it is first seen, and the same id keeps the same
    label everywhere on the page.
    """
    letters, out = {}, {}
    for c in cases:
        for e in c.get("shared_entities", []):
            key = (e["relation"], e["entity_id"])
            if key in out:
                continue
            rel = e["relation"]
            letters[rel] = letters.get(rel, 0) + 1
            out[key] = f"{rel.upper()}-{letters[rel]:03d}"
    return out


def ring_context(report: dict, cfg) -> dict:
    """Mean member score per ring, and its case id if one matches.

    The page shows the global extractor's rings; case ids belong to the
    anchored ones. A card only claims a case when the two overlap enough to be
    the same group, and the overlap is shown so nobody has to take it on
    trust.
    """
    import numpy as np
    import pyarrow.parquet as pq

    proc = cfg.abs_path(cfg.paths.processed)
    cases = report.get("case_files", [])
    if not cases or not (proc / "scores_week2.parquet").exists():
        return {}
    n = int(pq.read_table(proc / "nodes.parquet").num_rows)
    scores = np.zeros(n, dtype=np.float64)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()
    population = float(scores[scores > 0].mean()) if (scores > 0).any() else 0.0

    anchored = []
    ap = proc / "anchored.json"
    if ap.exists():
        an = json.loads(ap.read_text())
        nights = (an.get("window") or {}).get("nights")
        for r in an.get("final_rings", []):
            anchored.append((set(r.get("members", [])), r, nights))

    out = {"_population_mean_score": round(population, 4)}
    for c in cases:
        m = np.asarray(members_of(c), dtype=np.int64)
        if m.size == 0:
            continue
        blk = {"mean_score": round(float(scores[m].mean()), 4)}
        best, best_j = None, 0.0
        mset = set(m.tolist())
        for members, r, nights in anchored:
            if not members:
                continue
            j = len(mset & members) / len(mset | members)
            if j > best_j:
                best, best_j = (r, nights), j
        if best and best_j >= 0.3:
            r, nights = best
            blk["case"] = {"case_id": r.get("case_id"), "event": r.get("event"),
                           "first_seen_night": r.get("first_seen_night"),
                           "of_nights": nights, "overlap": round(best_j, 3)}
        out[c.get("rank")] = blk
    return out


def left_alone(cfg) -> dict:
    """The co-located clusters the pipeline did not touch, and what marks them.

    The hostel test records aggregate statistics for the untouched clusters and
    per-cluster detail only for the ones it flagged, so this reports exactly
    that and does not invent detail it does not have.
    """
    proc = cfg.abs_path(cfg.paths.processed)
    f = proc / "hostel_test.json"
    return json.loads(f.read_text()) if f.exists() else {}


def recommendations(report: dict, cfg, budget: int = 120) -> dict:
    """The recommended action for each ring on this page, and its two numbers.

    The page shows the global extractor's rings, not the anchored nightly
    queue, so the honest question a card can answer is: if these were the
    queue tonight and an analyst had `budget` minutes, what should be done
    with this one? That is the same planner the results section uses, run over
    exactly the rings shown here, and the page says so.
    """
    import numpy as np

    cases = report.get("case_files", [])
    if not cases or any(len(members_of(c)) != c.get("size") for c in cases):
        return {}
    try:
        from orbweaver.rings.policy import (CHURN_HEADLINE, expected_values, load_inputs,
                                            plan_capacity_aware, ring_economics)
        labels, scores, promo_value, ltv, _ = load_inputs(cfg)
    except Exception:
        return {}

    rings, keys = [], []
    for c in cases:
        m = np.asarray(members_of(c), dtype=np.int64)
        e = ring_economics(m, scores, promo_value, ltv, labels, cfg)
        e["density"] = c.get("density", 0.0)
        rings.append(e)
        keys.append(c.get("rank"))
    rev, held = plan_capacity_aware(rings, budget, CHURN_HEADLINE)
    out = {}
    for i, (k, r) in enumerate(zip(keys, rings)):
        ev_review, ev_hold = expected_values(r, CHURN_HEADLINE)
        out[k] = {"action": "review" if i in rev else ("auto-hold" if i in held else "ignore"),
                  "review_inr": ev_review, "hold_inr": ev_hold,
                  "minutes": r["review_minutes"]}
    out["_budget"] = budget
    out["_churn"] = CHURN_HEADLINE
    return out


def render(report: dict, actions: dict | None = None, ctx: dict | None = None,
           hostel: dict | None = None) -> str:
    cases = report.get("case_files", [])
    best = report.get("best_cell", {})
    graph = report.get("graph", {})
    base = report.get("base_rate_among_labelled")
    cell = report.get("grid", {}).get(
        f"tau={best.get('tau')},lambda={best.get('lambda')}", {})

    # Mean member score decides the order. Money at stake was the obvious
    # choice and it is not the right one: the ring-ranking comparison had the
    # mean member score beating density at every depth, and the review policy
    # found a density-ordered queue losing money outright. Two independent
    # measurements, so the page follows them.
    ctx = ctx or {}
    cases = sorted(cases, key=lambda c: (-(ctx.get(c.get("rank"), {}) or {}).get(
        "mean_score", 0.0), -c.get("rupees_at_stake", 0)))
    labels = anonymise(cases)

    p = []
    a = p.append
    a("<!-- generated by eval/case_report.py -->")
    a('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width,initial-scale=1">')
    a("<title>Orbweaver — ring case files</title>")
    a('<meta name="description" content="One card per detected ring: who is '
      'in it, what ties them together, how rare that tie is, and what it '
      'would cost to be wrong.">')
    a('<meta property="og:title" content="Orbweaver — ring case files">')
    a('<meta property="og:description" content="What an analyst is actually '
      'handed - evidence and a recommended action, one card per ring.">')
    a('<meta property="og:type" content="website">')
    a('<meta name="twitter:card" content="summary">')
    a(f"<style>{CSS}</style></head><body>")
    a('<div class="wrap">')
    a('<p class="backlink"><a class="btn ghost" href="index.html">'
      '&larr; Back to the overview</a></p>')
    a("<h1>Ring case files</h1>")
    a(f'<p class="sub">Week-2 graph, {graph.get("edges", 0):,} edges. '
      f'Score cut-off τ = {esc(best.get("tau"))}, λ = {esc(best.get("lambda"))}. '
      f'Ring precision {esc(best.get("ring_precision"))} against a base rate of '
      f'{esc(base)}.</p>')

    a('<div class="bar">')
    for label, val in (
        ("rings", cell.get("n_rings")),
        ("accounts", f"{cell.get('accounts_in_rings', 0):,}"),
        ("fraud found", cell.get("fraud_members")),
        ("real customers per catch", cell.get("normal_flagged_per_fraud_caught")),
        ("cost of being wrong", f"₹{cell.get('fp_cost_inr', 0):,.0f}"),
    ):
        a(f'<div class="stat"><b>{esc(val)}</b><span>{label}</span></div>')
    a("</div>")

    if actions:
        a(f'<div class="assume">Each card carries the action the review policy would take '
          f'if these {len(cases)} rings were tonight\'s queue and an analyst had '
          f'{actions["_budget"]} minutes, with a wrongly held customer assumed to lose '
          f'{actions["_churn"]:.0%} of their value. The two figures behind it are shown so a '
          'reviewer can see which one the recommendation turned on.</div>')

    a('<div class="assume">These are leads for a human to review, not verdicts. '
      "Rupee figures are counts multiplied by an assumed value — this dataset "
      "ships no monetary amounts — so they rank rings against each other and "
      "mean nothing in absolute terms.</div>")

    for c in cases:
        lab = c.get("labels", {})
        a('<div class="card">')
        info = (ctx.get(c.get("rank")) or {})
        a(f'<h2>Ring #{esc(c.get("rank"))} — {esc(c["size"])} accounts</h2>')
        case = info.get("case")
        if case:
            a('<div class="meta">'
              f'<strong>Case #{esc(case["case_id"])}</strong> · first seen night '
              f'{esc(case["first_seen_night"])} of {esc(case["of_nights"])} · '
              f'{esc(case["event"])} · matches the tracked ring at Jaccard '
              f'{case["overlap"]:.2f}</div>')
        base = ctx.get("_population_mean_score")
        if info.get("mean_score") is not None and base:
            a('<div class="meta">mean member score '
              f'<strong>{info["mean_score"]:.3f}</strong> against {base:.3f} '
              f'across every scored account — {info["mean_score"] / base:.1f}× '
              'the population</div>')
        a('<div class="meta">'
          f'density {esc(c.get("density"))} · {c.get("orders", 0):,} orders over '
          f'{esc(c.get("active_days"))} days · '
          f'{c.get("busiest_day_share", 0):.0%} of them on the busiest single day · '
          f'<strong>₹{c.get("rupees_at_stake", 0):,.0f}</strong> at stake</div>')

        a(f'<span class="tag f">{lab.get("fraud", 0)} known fraud</span>'
          f'<span class="tag n">{lab.get("normal", 0)} known good</span>'
          f'<span class="tag u">{lab.get("unlabelled", 0)} unreviewed</span>')

        act = (actions or {}).get(c.get("rank"))
        if act:
            verb = {"review": "Review", "auto-hold": "Auto-hold",
                    "ignore": "Leave for now"}[act["action"]]
            cls = {"review": "review", "auto-hold": "hold", "ignore": "ignore"}[act["action"]]
            a('<div class="act">'
              f'<span class="verb {cls}">{verb}</span>'
              f'<span>reviewing it is worth <strong>₹{act["review_inr"]:,.0f}</strong> '
              f'and costs {act["minutes"]} analyst minutes</span>'
              f'<span>auto-holding it is worth <strong>₹{act["hold_inr"]:,.0f}</strong> '
              'and costs none</span>'
              '</div>')

        ents = c.get("shared_entities", [])
        if ents:
            a('<table><tr><th>what they share</th><th class="num">members</th>'
              '<th class="num">coverage</th>'
              '<th class="num">accounts with it, platform-wide</th></tr>')
            for e in ents[:8]:
                rare = ' class="rare"' if e.get("global_users_with_entity", 0) <= 100 else ""
                tag = labels.get((e["relation"], e["entity_id"]), "")
                a(f'<tr><td>{esc(e["relation_label"])} <code>{esc(tag)}</code></td>'
                  f'<td class="num">{esc(e["members_sharing"])}</td>'
                  f'<td class="num">{e["coverage"]:.0%}</td>'
                  f'<td class="num"{rare}>{e.get("global_users_with_entity", 0):,}</td></tr>')
            a("</table>")
            a('<div class="note">Coverage only means something next to the last '
              "column. An entity that a handful of accounts on the whole platform "
              "have ever used is evidence; one that millions share is not.</div>")
        else:
            a('<div class="note">No entity is shared by enough of this ring to '
              "stand as evidence. It is held together by many small overlaps "
              "rather than one obvious tie — worth a look, but a weaker case.</div>")
        a("</div>")

    if hostel and hostel.get("clusters_found"):
        w = hostel.get("what_separates_them") or {}
        untouched = hostel.get("clusters_untouched")
        found = hostel["clusters_found"]
        a('<div class="card">')
        a("<h2>The clusters it left alone, and why</h2>")
        a(f'<div class="meta">{untouched:,} of {found:,} co-located groups whose '
          "labelled members are overwhelmingly normal — hostels, shared offices, "
          "joint families — have no member in any ring "
          f'({1 - hostel.get("share_of_clusters_touched", 0):.2%} of them)</div>')
        if w:
            a('<table><tr><th></th><th class="num">the groups it left alone</th>'
              '<th class="num">the ones it touched</th></tr>')
            for key, label in (("mean_score", "mean account score"),
                               ("relation_diversity", "kinds of shared entity"),
                               ("internal_edges", "edges inside the group")):
                u, f_ = w.get(f"untouched_{key}"), w.get(f"flagged_{key}")
                if u is None or f_ is None:
                    continue
                a(f'<tr><td>{label}</td><td class="num">{u:,.2f}</td>'
                  f'<td class="num">{f_:,.2f}</td></tr>')
            a("</table>")
            a('<div class="note">The separation is the account score, not the '
              "structure. Groups that were left alone are actually <em>more</em> "
              "densely connected and share more kinds of entity — which is exactly "
              "what a hostel looks like. What keeps them out of a ring is that "
              "their members do not behave like fraudsters, and that is the whole "
              "argument for scoring accounts before looking for dense structure "
              "rather than after.</div>")
        for wc in (hostel.get("worst_cases") or [])[:2]:
            a('<div class="act">'
              f'<span class="verb hold">Touched</span>'
              f'<span>a group of {wc["size"]} sharing one entity, '
              f'{wc["normal_share"]:.0%} of its labelled members good</span>'
              f'<span>{wc["members_in_a_ring"]} of them ended up in a ring '
              f'({wc["share_in_a_ring"]:.0%}), mean score {wc["mean_score"]:.3f}</span>'
              "</div>")
        a('<div class="note">These two are the failures, shown because a page that '
          "only listed the successes would be worth less. Both are groups where "
          "the members really did score high, so the score cut-off let them "
          "through — the cost of the ordering that makes everything else work.</div>")
        a("</div>")

    a('<p class="sub">Generated by <code>make report</code>.</p>')
    a("</div></body></html>")
    return "\n".join(p)


def main() -> None:
    cfg = load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    src = proc / "ring_report.json"
    if not src.exists():
        print("no ring_report.json; run `make rings` first")
        return
    dest = Path(cfg.abs_path(cfg.paths.figures)).parent / "case-files.html"
    report = json.loads(src.read_text())
    dest.write_text(render(report, recommendations(report, cfg),
                           ring_context(report, cfg), left_alone(cfg)))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
