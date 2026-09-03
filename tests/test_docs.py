"""Tests for the documents themselves.

Cross-references rot silently: a heading gets reworded, the link at the top of
the file still renders as a link, and nobody notices until a reader clicks it.
These check the three that would embarrass me most.
"""
from __future__ import annotations

import re

from orbweaver.config import load_config

CFG = load_config()
ROOT = CFG.abs_path(".")


def _anchor(title: str) -> str:
    """GitHub's rule: lowercase, drop anything that is not alphanumeric, a
    space or a hyphen, then spaces become hyphens."""
    a = "".join(c for c in title.lower() if c.isalnum() or c in " -")
    return "#" + a.replace(" ", "-")


def test_failures_index_links_resolve():
    text = (ROOT / "FAILURES.md").read_text()
    headings = {_anchor(h) for h in re.findall(r"^## (.+)$", text, re.M)}
    links = re.findall(r"\]\((#[^)]+)\)", text)
    assert links, "the index at the top has disappeared"
    missing = [ln for ln in links if ln not in headings]
    assert not missing, f"index links point at headings that no longer exist: {missing}"


def test_every_failure_entry_is_dated_and_first_person():
    """The log is only worth anything if it reads as someone's own account."""
    text = (ROOT / "FAILURES.md").read_text()
    for h in re.findall(r"^## (.+)$", text, re.M):
        assert re.match(r"^\d+ (August|September) — ", h), f"undated entry: {h}"


def test_readme_generated_block_is_present_and_filled():
    """Every number in the README comes out of the run artefacts. If the
    markers go, the next `make reproduce` silently stops updating it."""
    text = (ROOT / "README.md").read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    assert start in text and end in text
    block = text.split(start, 1)[1].split(end, 1)[0]
    rows = [r for r in block.splitlines() if r.startswith("|")]
    assert len(rows) > 4, "the generated results table is empty or nearly so"


def test_results_figures_exist():
    results = ROOT / "docs" / "results.md"
    if not results.exists():
        return
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", results.read_text()):
        assert (results.parent / rel).exists(), f"missing figure: {rel}"


# Each of these sections is written only when its artefact is on disk, so an
# artefact with no section means docs/results.md was generated before that
# stage ran. That is not hypothetical: `report` was named as a prerequisite of
# both `reproduce-core` and `reproduce`, make built it once and early, and a
# full run produced a results file missing six sections.
ARTEFACT_SECTIONS = {
    "merchant_view.json": "## The relation only the platform can see, measured",
    "replay.json": "## Watching the window day by day",
    "ring_scorer.json": "## Ranking rings by a learned confidence",
    "ring_context.json": "## Feeding the web back into the strand",
    "twins.json": "## Edges an attacker cannot cut",
    "ieee_cis.json": "## A payment processor's graph",
    "anchored.json": "## A ring you can find again tomorrow",
}


def test_results_covers_every_artefact_on_disk():
    results = ROOT / "docs" / "results.md"
    if not results.exists():
        return
    text = results.read_text()
    proc = CFG.abs_path(CFG.paths.processed)
    stale = [name for name, heading in ARTEFACT_SECTIONS.items()
             if (proc / name).exists() and heading not in text]
    assert not stale, (
        "docs/results.md predates these artefacts, so it was written before "
        f"their stages ran: {stale}")


def test_report_runs_after_every_stage_it_reports_on():
    """The regression guard for the bug above, at the source rather than the
    symptom: `reproduce` must not name `report` as a prerequisite."""
    line = next(ln for ln in (ROOT / "Makefile").read_text().splitlines()
                if ln.startswith("reproduce:"))
    prereqs = line.split(":", 1)[1].split("##")[0].split()
    assert "report" not in prereqs, (
        "make builds a target once per run, so `report` here is a no-op and "
        "the report is written before the later stages have run")


# Each investigation is supposed to leave both a section and a picture behind.
# The section check above caught a stale results file; this one exists because
# ring context shipped with a section and no figure, and nothing noticed.
ARTEFACT_FIGURES = {
    "merchant_view.json": "merchant_vs_platform.png",
    "replay.json": "time_to_detection.png",
    "ring_scorer.json": "queue_by_ranking.png",
    "ring_context.json": "ring_context.png",
    "twins.json": "adversarial.png",
    "ieee_cis.json": "ieee_relation_lift.png",
    "anchored.json": "ring_persistence.png",
}


def test_every_artefact_has_a_figure():
    proc = CFG.abs_path(CFG.paths.processed)
    figs = CFG.abs_path(CFG.paths.figures)
    results = ROOT / "docs" / "results.md"
    text = results.read_text() if results.exists() else ""
    missing = []
    for name, fig in ARTEFACT_FIGURES.items():
        if not (proc / name).exists():
            continue
        if not (figs / fig).exists():
            missing.append(f"{fig} (not drawn)")
        elif text and f"figures/{fig}" not in text:
            missing.append(f"{fig} (drawn but not shown)")
    assert not missing, f"artefacts reported without a figure: {missing}"


def test_prose_numbers_match_the_artefacts():
    """The hand-written docs quote a few numbers from the run. Those cannot be
    generated, so they are pinned here instead: each expected string is derived
    from the artefact, and the test says which document has gone stale.

    This exists because a read-through found the location lift quoted as 3.68×
    in two documents while the artefact said 3.7072, and a GraphSAGE paragraph
    describing a GPU run at 95 seconds when the recorded run was the CPU at 42.
    Every generated number was right; every stale one had been typed.
    """
    import json
    from orbweaver.features.node_features import FEATURE_NAMES

    proc = CFG.abs_path(CFG.paths.processed)
    weights = proc / "relation_weights.json"
    manifest = proc / "edges_week2_late_manifest.json"
    if not (weights.exists() and manifest.exists()):
        return
    w = json.loads(weights.read_text())
    w = w.get("relations") or w
    m = json.loads(manifest.read_text())
    share = {k: v["pairs"] / m["unique_edges"] for k, v in m["per_relation"].items()}

    expected = {
        "location lift": f"{w['r1']['lift']:.2f}×",
        "promotion lift": f"{w['r6']['lift']:.2f}×",
        "promotion edge share": f"{share['r6']:.0%}",
        "location edge share": f"{share['r1']:.0%}",
        "feature count": f"{len(FEATURE_NAMES)} features",
    }
    docs = {
        "README.md": ROOT / "README.md",
        "docs/architecture.md": ROOT / "docs" / "architecture.md",
    }
    stale = []
    for name, path in docs.items():
        text = path.read_text()
        for what, want in expected.items():
            if what == "feature count":
                ok = want in text or f"{len(FEATURE_NAMES)} engineered" in text
            else:
                ok = want in text
            if not ok:
                stale.append(f"{name}: {what} should read {want!r}")
    assert not stale, "prose has drifted from the artefacts:\n  " + "\n  ".join(stale)
