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
