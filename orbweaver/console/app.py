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


EXTRA_CSS = """
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-bottom:20px}
.controls label{display:block;font-size:11px;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);margin-bottom:4px}
.controls input,.controls select{font:14px inherit;padding:7px 10px;
border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink)}
.row{background:#fff;border:1px solid var(--line);border-radius:10px;
padding:14px 18px;margin-bottom:10px;cursor:pointer}
.row:hover{border-color:var(--accent)}
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
            f'<div class="row" hx-get="/ring/{c["rank"]}" hx-target="#detail" '
            f'hx-swap="innerHTML">'
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
        return "<p>No ring report found. Run <code>make rings</code> first.</p>"
    best = report.get("best_cell", {})
    opts = "".join(
        f'<option value="{k}">{v}</option>' for k, v in
        [("any", "anything"), ("r1", "order location"), ("r3", "delivery record"),
         ("r6", "promotion"), ("r7", "coupon type"), ("r8", "sales stimulation")])
    return f"""<!doctype html><meta charset="utf-8"><title>Orbweaver</title>
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<style>{CSS}{EXTRA_CSS}</style>
<div class="wrap">
<h1>Ring review queue</h1>
<p class="sub">Sorted by money at stake. Score cut-off τ = {esc(best.get('tau'))},
λ = {esc(best.get('lambda'))}; ring precision {esc(best.get('ring_precision'))}
against a base rate of {esc(report.get('base_rate_among_labelled'))}.</p>
{DEMO_BANNER if demo_mode() else ""}
<div class="assume">Leads for a human to review, not verdicts. Rupee figures use
an assumed value — this dataset ships no monetary amounts.</div>
<p class="sub"><a href="/offers">Which offers are being farmed</a> ·
<a href="/findings">What this found, and what it did not</a> ·
<a href="/health">health</a></p>
<form class="controls" hx-get="/rings" hx-target="#rings" hx-trigger="change, load">
  <div><label>shares a</label><select name="shares">{opts}</select></div>
  <div><label>at least this many known fraud</label>
    <input type="number" name="min_fraud" value="0" min="0" style="width:90px"></div>
</form>
<div id="rings"></div>
<div id="detail"></div>
</div>"""


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
            f'<div class="row" hx-get="/offers/{o["relation"]}/{o["entity"]}" '
            f'hx-target="#detail" hx-swap="innerHTML">'
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
        return "<p>No offer report found. Run <code>make offers</code> first.</p>"
    return f"""<!doctype html><meta charset="utf-8"><title>Orbweaver — offers</title>
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<style>{CSS}{EXTRA_CSS}</style>
<div class="wrap">
<h1>Which offers are being farmed</h1>
<p class="sub">{out['n_offers']:,} offers across the promotion, coupon and sales-
stimulation relations, in the scoring window. Ranked by leakage - the share of
redeemers a ring already flagged, or the account scorer's mean opinion of
them - never by a label.</p>
{DEMO_BANNER if demo_mode() else ""}
<div class="assume">The ranking uses no label. The labelled fraud share on
each card is what it is checked against, shown for review, never fed back
into the ranking.</div>
<form class="controls" hx-get="/offers/rows" hx-target="#offers" hx-trigger="change, load">
  <div><label>rank by</label><select name="by">
    <option value="ring_share">share already in a ring</option>
    <option value="mean_score">mean member score</option>
  </select></div>
  <div><label>at least this many redeemers</label>
    <input type="number" name="min_redeemers" value="5" min="1" style="width:90px"></div>
</form>
<div id="offers"></div>
<div id="detail"></div>
<p class="sub"><a href="/">Back to the review queue</a></p>
</div>"""


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
        f'<span class="tag n" style="cursor:pointer" '
        f'hx-get="/ring/{r}" hx-target="#detail" hx-swap="innerHTML">ring #{r}</span> '
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
        f'<span class="tag n" style="cursor:pointer" '
        f'hx-get="/offers/{e["relation"]}/{e["entity_id"]}" hx-target="#detail" hx-swap="innerHTML">'
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
    """The same answer as a card a person can read."""
    t0 = time.perf_counter()
    result = check_index().check(account)
    took = (time.perf_counter() - t0) * 1000.0
    return (f"<style>{CSS}</style><div class=\"wrap\">"
            f"{render_card(result)}"
            f"<p class=\"sub\">answered in {took:.1f} ms</p></div>")


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
                         f"{d.get('median')} of {(an.get('window') or {}).get('nights')}.")
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
    return f"""<!doctype html><meta charset="utf-8"><title>Orbweaver — findings</title>
<style>{CSS}</style><div class="wrap">
<h1>What this found, and what it did not</h1>
<p class="sub">Every figure is regenerated by <code>make reproduce</code>; none of
the numbers are typed in by hand. Three of the nine investigations returned
negative results and they are reported alongside the rest.</p>
{intro}
{DEMO_BANNER if demo_mode() else ""}
{figs}
<p class="sub"><a href="/">Back to the review queue</a></p>
</div>"""


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
