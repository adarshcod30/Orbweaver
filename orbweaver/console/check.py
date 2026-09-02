"""Everything known about one account, fast enough to sit in a request.

The case-file page answers "what is in this ring". This answers the question a
per-transaction system would actually ask: *this account is in front of me now
— is it in a ring, and what would you tell me about it?*

That makes latency a result rather than an implementation detail. Every index
is built once when the process starts; a lookup is then array indexing and a
couple of small slices, with no file read and no model call on the hot path.
The score has already been computed by the batch pipeline, because scoring an
account here would mean rebuilding its graph features, which is not a
request-time operation.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config


class CheckIndex:
    """Read-only indexes for per-account lookup, loaded once."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg = cfg or load_config()
        proc = cfg.abs_path(cfg.paths.processed)

        labels = pq.read_table(proc / "nodes.parquet")["label"].to_numpy()
        self.n = int(labels.size)
        self.labels = labels

        self.scores = np.zeros(self.n, dtype=np.float32)
        t = pq.read_table(proc / "scores_week2.parquet")
        self.scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

        # Ring membership, from whichever ring report exists.
        self.ring_of = np.full(self.n, -1, dtype=np.int32)
        self.rings: list[dict] = []
        for name in ("ring_report_deep.json", "ring_report.json"):
            p = proc / name
            if not p.exists():
                continue
            payload = json.loads(p.read_text())
            for case in payload.get("case_files", []):
                members = np.asarray(case.get("members_sample", []), dtype=np.int64)
                if members.size == 0:
                    continue
                i = len(self.rings)
                self.ring_of[members] = i
                self.rings.append({
                    "rank": case.get("rank"),
                    "size": case.get("size"),
                    "density": case.get("density"),
                    "rupees_at_stake": case.get("rupees_at_stake"),
                    "busiest_day_share": case.get("busiest_day_share"),
                    "shared_entities": case.get("shared_entities", [])[:5],
                    "labels": case.get("labels", {}),
                })
            break

        # Adjacency in the scoring window, as CSR so a neighbour lookup is a
        # slice rather than a scan.
        e = pq.read_table(proc / "edges_week2_late.parquet",
                          columns=["src", "dst", "weight", "min_entity_size"])
        src = e["src"].to_numpy().astype(np.int64)
        dst = e["dst"].to_numpy().astype(np.int64)
        w = e["weight"].to_numpy().astype(np.float32)
        esz = e["min_entity_size"].to_numpy().astype(np.int32)
        node = np.concatenate([src, dst])
        other = np.concatenate([dst, src])
        order = np.argsort(node, kind="stable")
        self.nbr = other[order]
        self.nbr_w = np.concatenate([w, w])[order]
        self.nbr_entity = np.concatenate([esz, esz])[order]
        counts = np.bincount(node, minlength=self.n)
        self.indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(counts, out=self.indptr[1:])

        self.assumption = (
            f"Rupee figures use an assumed Rs."
            f"{cfg.cost.assumed_avg_promo_value_inr:.0f} per promotion; this "
            "dataset ships no monetary amounts.")

    def check(self, account: int) -> dict:
        """One account's answer. Array indexing only — no file reads here."""
        if not (0 <= account < self.n):
            return {"account": account, "known": False,
                    "reason": "outside the account range of this window"}

        lo, hi = self.indptr[account], self.indptr[account + 1]
        nbr = self.nbr[lo:hi]
        in_ring = self.ring_of[nbr]
        rare = self.nbr_entity[lo:hi] <= 10

        label = int(self.labels[account]) if account < self.labels.size else -1
        ring_i = int(self.ring_of[account])
        out: dict = {
            "account": int(account),
            "known": True,
            "score": round(float(self.scores[account]), 4),
            "label": {1: "known fraud", 0: "known good", -1: "unreviewed"}[label],
            "neighbours": int(nbr.size),
            "neighbours_in_a_ring": int((in_ring >= 0).sum()),
            "rare_links": int(rare.sum()),
            "in_a_ring": ring_i >= 0,
            "assumption": self.assumption,
        }
        if ring_i >= 0:
            r = self.rings[ring_i]
            out["ring"] = {
                "rank": r["rank"], "size": r["size"], "density": r["density"],
                "rupees_at_stake": r["rupees_at_stake"],
                "same_day_share": r["busiest_day_share"],
                "labels": r["labels"],
                "shared": [
                    {"what": e.get("relation_label"),
                     "coverage": e.get("coverage"),
                     "accounts_platform_wide": e.get("global_users_with_entity")}
                    for e in r["shared_entities"]
                ],
            }
        return out


def render_card(result: dict) -> str:
    """A small HTML card, for a human rather than a caller."""
    import html

    def esc(x):
        return html.escape(str(x))

    if not result.get("known"):
        return f'<div class="card"><h2>Account {esc(result["account"])}</h2>' \
               f'<p class="note">{esc(result.get("reason", "unknown"))}</p></div>'

    rows = ""
    if result.get("in_a_ring"):
        r = result["ring"]
        shared = "".join(
            f'<tr><td>{esc(s["what"])}</td><td class="num">{s["coverage"]:.0%}</td>'
            f'<td class="num">{s["accounts_platform_wide"]:,}</td></tr>'
            for s in r["shared"] if s.get("coverage") is not None)
        rows = (f'<p><strong>In ring #{esc(r["rank"])}</strong> — '
                f'{esc(r["size"])} accounts, density {esc(r["density"])}, '
                f'₹{r["rupees_at_stake"]:,.0f} at stake, '
                f'{r["same_day_share"]:.0%} of orders on the busiest day.</p>'
                + (f'<table><tr><th>shares</th><th class="num">coverage</th>'
                   f'<th class="num">platform-wide</th></tr>{shared}</table>'
                   if shared else ""))
    else:
        rows = '<p class="note">Not in any surfaced ring.</p>'

    return (f'<div class="card"><h2>Account {esc(result["account"])}</h2>'
            f'<div class="meta">score {esc(result["score"])} · '
            f'{esc(result["label"])} · {esc(result["neighbours"])} neighbours, '
            f'{esc(result["neighbours_in_a_ring"])} of them in a ring · '
            f'{esc(result["rare_links"])} rare links</div>'
            f'{rows}<div class="note">{esc(result["assumption"])}</div></div>')
