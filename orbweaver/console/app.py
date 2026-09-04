"""A small review console over the extracted rings.

The static page produced by `eval/case_report.py` is the deliverable an analyst
could be handed; this is the same data with the two things a queue actually
needs — filtering by what the ring shares and how much is at stake, and a
drill-down into one case.

FastAPI serving HTML fragments to HTMX, deliberately: no build step, no npm, no
JavaScript bundle. The whole thing is one file and it reads the same
`ring_report.json` everything else does, so it cannot show a number that
disagrees with `docs/results.md`.

    make console      # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import time

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from orbweaver.config import load_config
from orbweaver.console.check import CheckIndex, render_card
from eval.case_report import CSS, esc

app = FastAPI(title="Orbweaver")

# Built once at import, not per request. The whole point of /check is that a
# per-transaction system could call it inside a request, and that is only true
# if answering one account is array indexing rather than file reads.
_INDEX = None


def num(x) -> str:
    """3.0 nights reads as a typo; 3 nights reads as a count."""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def demo_mode() -> bool:
    """Serve the committed bundle when there is no full run to serve from."""
    from orbweaver.console import demo
    return demo.should_use_demo(load_config())


def check_index():
    """The route handler for "/" is also called index(), so this one is named
    apart from it - the collision silently shadowed this function."""
    global _INDEX
    if _INDEX is None:
        if demo_mode():
            from orbweaver.console.demo import DemoIndex
            _INDEX = DemoIndex(load_config())
        else:
            _INDEX = CheckIndex(load_config())
    return _INDEX


def load_report() -> dict:
    cfg = load_config()
    if demo_mode():
        from orbweaver.console.demo import bundle_path
        path = bundle_path(cfg) / "rings.json"
    else:
        path = cfg.abs_path(cfg.paths.processed) / "ring_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def artefact(name: str) -> dict:
    """A results artefact, from the bundle in demo mode and the run otherwise."""
    cfg = load_config()
    if demo_mode():
        from orbweaver.console.demo import bundle_path
        path = bundle_path(cfg) / name
    else:
        path = cfg.abs_path(cfg.paths.processed) / name
    return json.loads(path.read_text()) if path.exists() else {}


DEMO_BANNER = (
    '<div class="assume"><strong>This is the demo bundle.</strong> The numbers, '
    'rings and evidence are the real ones from a full run, but the graph itself '
    'is not here - it is 35.7 million edges - so <code>/check</code> serves stored '
    'neighbour counts rather than computing a ring around an account live. '
    'Everything is reproducible from the raw data with <code>make reproduce</code>.</div>')

NAV = ('<p class="sub"><a href="/">queue</a> · <a href="/offers">offers</a> · '
       '<a href="/replay">nightly replay</a> · <a href="/findings">findings</a> · '
       '<a href="/health">health</a></p>')


def page(title: str, description: str, body: str) -> str:
    """Every full page shares one head: charset, viewport, OG/Twitter tags,
    the shared theme-aware stylesheet. Fragments HTMX swaps into #detail are
    not full documents and do not go through this."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary">
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<style>{CSS}{EXTRA_CSS}</style></head><body>
<div class="wrap">{body}</div></body></html>"""


def empty_state(title: str, note: str) -> str:
    return (f'<div class="empty-state"><b>{esc(title)}</b>{esc(note)}</div>')


EXTRA_CSS = """
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:20px}
.controls label{display:block;font-size:11px;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);margin-bottom:4px}
.controls input,.controls select{font:14px inherit;padding:7px 10px;
border:1px solid var(--line);border-radius:7px;background:var(--surface);
color:var(--ink)}
.row{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:14px 18px;margin-bottom:10px;cursor:pointer;box-shadow:var(--shadow)}
.row:hover{border-color:var(--accent)}
.row:focus-visible{outline:none;box-shadow:var(--focus)}
.row .top{display:flex;justify-content:space-between;gap:16px;align-items:baseline}
.row .why{color:var(--muted);font-size:13px;margin-top:5px}
.empty{color:var(--muted);padding:30px 0}
"""


def ring_rows(report: dict, shares: str, min_fraud: int) -> str:
    cases = report.get("case_files", [])
    out = []
    for c in sorted(cases, key=lambda x: -x.get("rupees_at_stake", 0)):
        ents = c.get("shared_entities", [])
        if shares and shares != "any":
            if not any(e["relation"] == shares for e in ents):
                continue
        if c.get("labels", {}).get("fraud", 0) < min_fraud:
            continue
        top = ents[0] if ents else None
        why = (f'{top["coverage"]:.0%} share one {top["relation_label"]}, '
               f'held by {top["global_users_with_entity"]:,} accounts platform-wide'
               if top else "no single strong shared entity")
        lab = c.get("labels", {})
        out.append(
            f'<div class="row" tabindex="0" role="button" '
            f'aria-label="Open ring {c["rank"]}" '
            f'hx-get="/ring/{c["rank"]}" hx-target="#detail" '
            f'hx-trigger="click, keyup[key==\'Enter\']" hx-swap="innerHTML">'
            f'<div class="top"><strong>Ring #{c["rank"]} — {c["size"]} accounts</strong>'
            f'<span>₹{c.get("rupees_at_stake", 0):,.0f}</span></div>'
            f'<div class="why">{esc(why)}</div>'
            f'<div class="why">{lab.get("fraud", 0)} known fraud · '
            f'{lab.get("normal", 0)} known good · '
            f'{lab.get("unlabelled", 0)} unreviewed</div></div>')
    return "".join(out) or '<div class="empty">Nothing matches that filter.</div>'


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    report = load_report()
    if not report:
        return page("Orbweaver — ring review queue",
                     "The review queue an analyst would actually be handed.",
                     empty_state("No ring report found.",
                                 " Run make rings (or make reproduce) to "
                                 "produce ring_report.json, then reload."))
    best = report.get("best_cell", {})
    opts = "".join(
        f'<option value="{k}">{v}</option>' for k, v in
        [("any", "anything"), ("r1", "order location"), ("r3", "delivery record"),
         ("r6", "promotion"), ("r7", "coupon type"), ("r8", "sales stimulation")])
    body = f"""<h1>Ring review queue</h1>
<p class="sub">Sorted by money at stake. Score cut-off τ = {esc(best.get('tau'))},
λ = {esc(best.get('lambda'))}; ring precision {esc(best.get('ring_precision'))}
against a base rate of {esc(report.get('base_rate_among_labelled'))}. A ring is
a candidate — several accounts a shared entity ties together, dense enough to
look coordinated — not a verdict.</p>
{DEMO_BANNER if demo_mode() else ""}
<div class="assume">Leads for a human to review, not verdicts. Rupee figures use
an assumed value — this dataset ships no monetary amounts.</div>
{NAV}
<form class="controls" hx-get="/rings" hx-target="#rings" hx-trigger="change, load">
  <div><label>shares a</label><select name="shares">{opts}</select></div>
  <div><label>at least this many known fraud</label>
    <input type="number" name="min_fraud" value="0" min="0" style="width:90px"></div>
</form>
<div id="rings"></div>
<div id="detail"></div>"""
    return page("Orbweaver — ring review queue",
                "The review queue an analyst would actually be handed: "
                "evidence and a recommended action, one card per ring.", body)


@app.get("/rings", response_class=HTMLResponse)
def rings(shares: str = Query("any"), min_fraud: int = Query(0)) -> str:
    return ring_rows(load_report(), shares, min_fraud)


def load_offers() -> dict:
    return artefact("offers.json")


def offer_rows(out: dict, by: str, min_redeemers: int) -> str:
    offers = [o for o in out.get("offers", []) if o["redeemers"] >= min_redeemers]
    key = (lambda o: -o["share_in_a_ring"]) if by == "ring_share" else (lambda o: -o["mean_score"])
    offers = sorted(offers, key=key)[:200]
    rows = []
    for o in offers:
        rows.append(
            f'<div class="row" tabindex="0" role="button" '
            f'aria-label="Open {esc(o["relation_label"])} offer {o["entity"]}" '
            f'hx-get="/offers/{o["relation"]}/{o["entity"]}" '
            f'hx-target="#detail" hx-trigger="click, keyup[key==\'Enter\']" '
            f'hx-swap="innerHTML">'
            f'<div class="top"><strong>{esc(o["relation_label"])} — '
            f'{o["redeemers"]:,} redeemers</strong>'
            f'<span>₹{o["rupees_at_stake"]:,.0f}</span></div>'
            f'<div class="why">{o["share_in_a_ring"]:.0%} already in a ring · '
            f'mean score {o["mean_score"]:.3f} · on {o["distinct_rings"]} distinct rings</div></div>')
    return "".join(rows) or '<div class="empty">Nothing matches that filter.</div>'


@app.get("/offers", response_class=HTMLResponse)
def offers_page() -> str:
    out = load_offers()
    if not out:
        return page("Orbweaver — offers",
                     "Which promotions are being farmed, ranked with no label.",
                     empty_state("No offer report found.",
                                 " Run make offers to produce offers.json, "
                                 "then reload.") + NAV)
    body = f"""<h1>Which offers are being farmed</h1>
<p class="sub">{out['n_offers']:,} offers across the promotion, coupon and sales-
stimulation relations, in the scoring window. Ranked by leakage - the share of
redeemers a ring already flagged, or the account scorer's mean opinion of
them - never by a label.</p>
{DEMO_BANNER if demo_mode() else ""}
<div class="assume">The ranking uses no label. The labelled fraud share on
each card is what it is checked against, shown for review, never fed back
into the ranking.</div>
{NAV}
<form class="controls" hx-get="/offers/rows" hx-target="#offers" hx-trigger="change, load">
  <div><label>rank by</label><select name="by">
    <option value="ring_share">share already in a ring</option>
    <option value="mean_score">mean member score</option>
  </select></div>
  <div><label>at least this many redeemers</label>
    <input type="number" name="min_redeemers" value="5" min="1" style="width:90px"></div>
</form>
<div id="offers"></div>
<div id="detail"></div>"""
    return page("Orbweaver — offers",
                f"{out['n_offers']:,} promotions ranked by leakage, with no "
                "label used in the ranking.", body)


@app.get("/offers/rows", response_class=HTMLResponse)
def offers_rows(by: str = Query("ring_share"), min_redeemers: int = Query(5)) -> str:
    return offer_rows(load_offers(), by, min_redeemers)


@app.get("/offers/{relation}/{entity}", response_class=HTMLResponse)
def offer_detail(relation: str, entity: int) -> str:
    out = load_offers()
    o = next((x for x in out.get("offers", [])
             if x["relation"] == relation and x["entity"] == entity), None)
    if not o:
        return '<div class="empty">No such offer.</div>'
    ring_links = "".join(
        f'<span class="tag n" style="cursor:pointer" tabindex="0" role="button" '
        f'aria-label="Open ring {r}" hx-get="/ring/{r}" hx-target="#detail" '
        f'hx-trigger="click, keyup[key==\'Enter\']" hx-swap="innerHTML">ring #{r}</span> '
        for r in o.get("ring_ranks", [])) or '<span class="note">none yet</span>'
    fraud_share = o.get("fraud_share_among_labelled")
    return f"""<div class="card">
<h2>{esc(o['relation_label'])} #{o['entity']} — {o['redeemers']:,} redeemers</h2>
<div class="meta">{o['share_in_a_ring']:.0%} already in a ring ·
mean score {o['mean_score']:.3f} · p90 score {o['p90_score']:.3f} ·
<strong>₹{o['rupees_at_stake']:,.0f}</strong> at stake</div>
<div class="meta">{esc(f"burst z {o['burst_z']}") if o.get('burst_z') is not None else ""}</div>
<span class="tag f">{o['fraud_redeemers']} known fraud</span>
<span class="tag n">{o['labelled_redeemers'] - o['fraud_redeemers']} known good</span>
<span class="tag u">{o['redeemers'] - o['labelled_redeemers']} unreviewed</span>
<div class="note">labelled fraud share: {esc(fraud_share) if fraud_share is not None else 'too few labelled to say'}
 — the evaluation target, not an input to the ranking above</div>
<div class="note">rings on this offer: {ring_links}</div>
<div class="note">{esc(o.get('rupees_at_stake_basis', ''))}</div>
</div>"""


@app.get("/ring/{rank}", response_class=HTMLResponse)
def ring_detail(rank: int) -> str:
    report = load_report()
    case = next((c for c in report.get("case_files", []) if c["rank"] == rank), None)
    if not case:
        return '<div class="empty">No such ring.</div>'
    lab = case.get("labels", {})
    rows = "".join(
        f'<tr><td>{esc(e["relation_label"])}</td>'
        f'<td class="num">{e["members_sharing"]}</td>'
        f'<td class="num">{e["coverage"]:.0%}</td>'
        f'<td class="num"{" class=rare" if e["global_users_with_entity"] <= 100 else ""}>'
        f'{e["global_users_with_entity"]:,}</td></tr>'
        for e in case.get("shared_entities", [])[:10])
    table = (f'<table><tr><th>what they share</th><th class="num">members</th>'
             f'<th class="num">coverage</th>'
             f'<th class="num">platform-wide</th></tr>{rows}</table>'
             if rows else '<p class="note">No single strong shared entity.</p>')
    members = ", ".join(str(m) for m in case.get("members_sample", [])[:20])
    offer_links = "".join(
        f'<span class="tag n" style="cursor:pointer" tabindex="0" role="button" '
        f'aria-label="Open {esc(e["relation_label"])} offer {e["entity_id"]}" '
        f'hx-get="/offers/{e["relation"]}/{e["entity_id"]}" hx-target="#detail" '
        f'hx-trigger="click, keyup[key==\'Enter\']" hx-swap="innerHTML">'
        f'{esc(e["relation_label"])} #{e["entity_id"]}</span> '
        for e in case.get("shared_entities", []) if e["relation"] in ("r6", "r7", "r8"))
    offers_note = (f'<div class="note">offers this ring redeemed: {offer_links}</div>'
                  if offer_links else "")
    return f"""<div class="card">
<h2>Ring #{case['rank']} — {case['size']} accounts</h2>
<div class="meta">density {esc(case.get('density'))} ·
{case.get('orders', 0):,} orders over {esc(case.get('active_days'))} days ·
{case.get('busiest_day_share', 0):.0%} on the busiest single day ·
<strong>₹{case.get('rupees_at_stake', 0):,.0f}</strong> at stake</div>
<span class="tag f">{lab.get('fraud', 0)} known fraud</span>
<span class="tag n">{lab.get('normal', 0)} known good</span>
<span class="tag u">{lab.get('unlabelled', 0)} unreviewed</span>
{table}
<div class="note">First accounts: {esc(members)}</div>
<div class="note">{esc(case.get('rupees_at_stake_basis', ''))}</div>
{offers_note}
</div>"""


@app.get("/check/{account}")
def check(account: int):
    """Everything known about one account, as JSON."""
    t0 = time.perf_counter()
    out = check_index().check(account)
    out["took_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out


@app.get("/check/{account}/card", response_class=HTMLResponse)
def check_card(account: int) -> str:
    """The same answer as a card a person can read. Reachable two ways: as an
    HTMX fragment, and as a URL someone opens directly (the video script
    does this), so it carries its own charset and wrap rather than assuming
    a parent page already did."""
    t0 = time.perf_counter()
    result = check_index().check(account)
    took = (time.perf_counter() - t0) * 1000.0
    return page(f"Orbweaver — account {account}",
                "What this pipeline knows about one account, answered live.",
                f"{render_card(result)}"
                f'<p class="sub">answered in {took:.1f} ms</p>{NAV}')


@app.get("/health")
def health():
    """Enough to tell a deployment apart from a broken one."""
    from orbweaver.console.demo import available, bundle_path
    cfg = load_config()
    report = load_report()
    ok = bool(report.get("case_files"))
    out = {"ok": ok, "mode": "demo" if demo_mode() else "full run",
           "rings": len(report.get("case_files", [])),
           "bundle_present": available(cfg)}
    if demo_mode():
        meta = bundle_path(cfg) / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            out["accounts_served"] = m.get("accounts_in_bundle")
            out["bundle_mb"] = round(m.get("bytes", 0) / 1e6, 2)
    return out


FIGURES = [
    ("headline_precision_vs_cost.png",
     "Ring precision against the cost of being wrong, across operating points."),
    ("ring_persistence.png",
     "Whether a ring found tonight existed last night, and when each was first seen."),
    ("policy_frontier.png",
     "What each review policy stops against what it harms, as the budget grows."),
    ("time_to_detection.png", "Precision by how many nights of data have accumulated."),
    ("merchant_vs_platform.png",
     "What one business sees against what the platform sees, at equal review capacity."),
    ("adversarial.png", "Precision as an attacker fragments the ring into smaller cells."),
    ("relation_lift.png", "How much each kind of shared entity predicts fraud."),
    ("ieee_relation_lift.png", "The same method on a payment processor's transactions."),
    ("queue_by_ranking.png", "Three ways of ordering the review queue."),
    ("ring_context.png", "Ring history as an account feature, and why it did nothing."),
]


@app.get("/findings", response_class=HTMLResponse)
def findings() -> str:
    """A read-only page of the figures, for someone who will not run anything."""
    an, po = artefact("anchored.json"), artefact("policy.json")
    lines = []
    s_ = an.get("summary") or {}
    if s_:
        p3 = (s_.get("persistence_at_0.3") or {}).get("share_of_final_rings_with_a_predecessor")
        d = s_.get("days_to_detection") or {}
        if p3 is not None:
            lines.append(f"{p3:.0%} of the rings found on the last night had a case open the "
                         f"night before, and the median case was first seen on night "
                         f"{num(d.get('median'))} of {(an.get('window') or {}).get('nights')}.")
    if po:
        b = (po.get("final_night", {}).get("budgets") or {}).get("60", {})
        if b:
            c = b.get("capacity-aware", {})
            lines.append(f"With one analyst for an hour a night, the review policy stops "
                         f"₹{c.get('fraud_value_stopped_inr', 0):,.0f} of promotion value and "
                         f"harms ₹{c.get('legitimate_value_harmed_inr', 0):,.0f} of legitimate "
                         "value, on stated assumptions.")
    figs = "".join(
        f'<div class="card"><h2>{esc(name.replace("_", " ").replace(".png", ""))}</h2>'
        f'<p class="note">{esc(caption)}</p>'
        f'<img src="https://raw.githubusercontent.com/adarshcod30/Orbweaver/main/docs/figures/{name}" '
        f'alt="{esc(caption)}" style="max-width:100%;border:1px solid var(--line);'
        f'border-radius:8px;margin-top:10px"></div>'
        for name, caption in FIGURES)
    intro = "".join(f"<p>{esc(t)}</p>" for t in lines)
    body = f"""<h1>What this found, and what it did not</h1>
<p class="sub">Every figure is regenerated by <code>make reproduce</code>; none of
the numbers are typed in by hand. Four of the thirteen investigations returned
negative results and they are reported alongside the rest.</p>
{NAV}
{intro}
{DEMO_BANNER if demo_mode() else ""}
{figs}"""
    return page("Orbweaver — findings",
                "What this project found across thirteen investigations, "
                "four of which came back negative.", body)


@app.get("/replay", response_class=HTMLResponse)
def replay_page() -> str:
    """The queue is not a snapshot - it is rebuilt every night, and a ring
    found tonight has to be recognisable as the same case tomorrow. This is
    that replay, one night at a time, for someone deciding whether a nightly
    process is worth running at all."""
    an, po = artefact("anchored.json"), artefact("policy.json")
    if not an or not an.get("nights"):
        return page("Orbweaver — nightly replay",
                     "Replaying the scoring window one night at a time.",
                     empty_state("No replay found.",
                                 " Run make anchored to produce anchored.json, "
                                 "then reload.") + NAV)
    nights_budget = {b["night"]: b for b in po.get("night_by_night_at_120_minutes", [])}
    s_ = an.get("summary") or {}
    persisted = (s_.get("persistence_at_0.3") or {}).get(
        "share_of_final_rings_with_a_predecessor")
    persisted_pct = f"{persisted:.0%}" if persisted is not None else "an unmeasured share"
    d = s_.get("days_to_detection") or {}
    window_nights = (an.get("window") or {}).get("nights")

    rows = []
    for n in an["nights"]:
        night = n["night"]
        fn = n.get("final_night") or {}
        budget = nights_budget.get(night, {})
        rows.append(
            f'<tr><td>night {night}</td>'
            f'<td class="num">{n.get("rings_after_dedupe", 0):,}</td>'
            f'<td class="num">{esc(fn.get("ring_precision"))}</td>'
            f'<td class="num">{fn.get("fraud_members", 0)}</td>'
            f'<td class="num">₹{budget.get("fraud_value_stopped_inr", 0):,.0f}</td>'
            f'<td class="num">₹{budget.get("cumulative_fraud_value_stopped_inr", 0):,.0f}</td></tr>')
    table = ('<table><tr><th>replaying up to</th><th class="num">rings found</th>'
             '<th class="num">precision if this were the last night</th>'
             '<th class="num">fraud accounts in those rings</th>'
             '<th class="num">stopped that night</th>'
             '<th class="num">running total</th></tr>'
             + "".join(rows) + "</table>")

    events = (s_.get("events_on_the_final_night_at_0.3")
              or (an.get("nights", [{}])[-1] or {}).get("events") or {})
    events_line = (
        f'On the final night: {events.get("born", 0)} rings appeared for the '
        f'first time, {events.get("continued", 0)} continued from the night '
        f'before, {events.get("died", 0)} disappeared, {events.get("merged", 0)} '
        f'merged into another case and {events.get("split", 0)} split in two.'
        if events else "")

    body = f"""<h1>Replaying the window, one night at a time</h1>
<p class="sub">One night of data puts the queue at chance. This replays each
night of the scoring window as it actually accumulated, re-extracting rings
anchored around fixed accounts so a case can be tracked across nights instead
of recomputed from scratch — {esc(persisted_pct)} of final-night rings had a
case open the night before, and the median case was first seen on night
{esc(num(d.get('median')))} of {esc(window_nights)}.</p>
{DEMO_BANNER if demo_mode() else ""}
{NAV}
{table}
<div class="note">"Precision if this were the last night" scores that night's
rings against the same labels used everywhere else in this project — it is
not a forecast, it is what the queue would have looked like had the replay
stopped there. "Stopped" and "running total" assume one analyst working two
hours a night at the capacity-aware policy, on the same stated ₹ assumptions
as the rest of this console.</div>
{f'<p class="sub">{esc(events_line)}</p>' if events_line else ''}
<p class="sub">Figures: <a href="https://raw.githubusercontent.com/adarshcod30/Orbweaver/main/docs/figures/time_to_detection.png">time_to_detection.png</a>
· <a href="https://raw.githubusercontent.com/adarshcod30/Orbweaver/main/docs/figures/ring_persistence.png">ring_persistence.png</a>
· full numbers in <a href="https://github.com/adarshcod30/Orbweaver/blob/main/docs/results.md#a-ring-you-can-find-again-tomorrow">docs/results.md</a>.</p>"""
    return page("Orbweaver — nightly replay",
                "Replaying the scoring window one night at a time: does a "
                "ring found tonight survive to be the same case tomorrow.",
                body)


@app.exception_handler(404)
async def not_found(request, exc):
    from fastapi.responses import HTMLResponse as _HTMLResponse
    body = page("Orbweaver — page not found",
                "That page does not exist.",
                empty_state("Nothing here.",
                            f" There is no page at {esc(request.url.path)}. "
                            "Try the review queue, the offers page, or "
                            "nightly replay.") + NAV)
    return _HTMLResponse(body, status_code=404)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
