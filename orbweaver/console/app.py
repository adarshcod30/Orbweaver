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
_INDEX: CheckIndex | None = None


def check_index() -> CheckIndex:
    """The route handler for "/" is also called index(), so this one is named
    apart from it - the collision silently shadowed this function."""
    global _INDEX
    if _INDEX is None:
        _INDEX = CheckIndex(load_config())
    return _INDEX


def load_report() -> dict:
    cfg = load_config()
    path = cfg.abs_path(cfg.paths.processed) / "ring_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
<div class="assume">Leads for a human to review, not verdicts. Rupee figures use
an assumed value — this dataset ships no monetary amounts.</div>
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


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
