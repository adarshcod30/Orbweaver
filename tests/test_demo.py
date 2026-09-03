"""Tests for the demo bundle and the console that serves it.

The point of the bundle is that someone can run the console without the 4 GB
dataset, so the test that matters is the one that proves exactly that: copy
the repository into a temp directory, delete `data/processed` entirely, and
check the console still answers.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap

import pytest

from orbweaver.config import load_config
from orbweaver.console import demo as D

CFG = load_config()
ROOT = CFG.abs_path(".")
BUNDLE = ROOT / D.BUNDLE_DIR

needs_bundle = pytest.mark.skipif(not (BUNDLE / "accounts.parquet").exists(),
                                  reason="run `make demo-bundle` first")


@needs_bundle
def test_bundle_is_under_the_size_cap():
    meta = json.loads((BUNDLE / "meta.json").read_text())
    assert meta["bytes"] <= D.MAX_BYTES, \
        f"bundle is {meta['bytes'] / 1e6:.1f} MB, cap is {D.MAX_BYTES / 1e6:.0f} MB"
    actual = sum(f.stat().st_size for f in BUNDLE.iterdir()
                 if f.is_file() and f.name != "meta.json")
    assert actual == meta["bytes"], "meta.json disagrees with what is on disk"


@needs_bundle
def test_the_bundle_carries_no_raw_graph():
    """35.7 million edges must not have crept in. If a parquet here is large,
    something is shipping the graph."""
    for f in BUNDLE.iterdir():
        assert f.stat().st_size < 20 * 1024 * 1024, f"{f.name} is too big for a bundle"
    assert not (BUNDLE / "edges_week2_late.parquet").exists()


@needs_bundle
def test_check_answers_for_a_ring_member_and_a_sampled_outsider():
    ix = D.DemoIndex(CFG)
    rings = json.loads((BUNDLE / "rings.json").read_text()).get("case_files", [])
    assert rings, "the bundle has no rings to serve"
    member = rings[0]["members_sample"][0]
    r = ix.check(int(member))
    assert r["known"] and r["in_a_ring"]
    assert r["ring"]["rank"] == rings[0]["rank"]
    assert r["demo"] is True

    outsiders = [int(a) for a in ix.ids[ix.ring < 0][:5]]
    assert outsiders, "no non-member accounts in the sample"
    r = ix.check(outsiders[0])
    assert r["known"] and not r["in_a_ring"]
    assert "score" in r and "neighbours" in r


@needs_bundle
def test_check_says_plainly_what_it_cannot_do_without_the_graph():
    """The demo must not imply it computed a ring live when it read one."""
    ix = D.DemoIndex(CFG)
    an = json.loads((BUNDLE / "anchored.json").read_text())
    anchor = an["final_rings"][0]["anchor"]
    r = ix.check(int(anchor))
    ar = r["anchored_ring"]
    assert ar["stored"] is True, "a stored ring must not claim to be computed"
    assert ar["found"] and ar["case"]["case_id"] == an["final_rings"][0]["case_id"]

    outsiders = [int(a) for a in ix.ids[ix.ring < 0][:200]]
    missing = next((a for a in outsiders if a not in ix.case_of), None)
    if missing is not None:
        ar = ix.check(missing)["anchored_ring"]
        assert not ar["found"] and "no graph" in ar["reason"]


@needs_bundle
def test_an_account_outside_the_sample_is_refused_honestly():
    ix = D.DemoIndex(CFG)
    absent = int(ix.ids.max()) + 1
    r = ix.check(absent)
    assert not r["known"] and "demo sample" in r["reason"]


@needs_bundle
def test_the_console_starts_from_the_bundle_alone():
    """The real test: a copy of the repository with no data/processed at all
    must still serve the queue, a ring, /check and /health."""
    fastapi = pytest.importorskip("fastapi")
    tmp = CFG.abs_path(".") / ".demo_check"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        for name in ("orbweaver", "eval", "config", D.BUNDLE_DIR):
            src = ROOT / name
            if src.exists():
                shutil.copytree(src, tmp / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (tmp / "data").mkdir(exist_ok=True)
        (tmp / "data" / "processed").mkdir(exist_ok=True)   # present but empty
        script = textwrap.dedent("""
            import json, sys
            from fastapi.testclient import TestClient
            from orbweaver.console.app import app, demo_mode
            assert demo_mode(), "did not fall back to the bundle"
            c = TestClient(app)
            h = c.get("/health").json()
            assert h["ok"] and h["mode"] == "demo", h
            assert c.get("/").status_code == 200
            assert c.get("/rings").status_code == 200
            rings = json.load(open("demo/rings.json"))["case_files"]
            r = c.get(f"/ring/{rings[0]['rank']}")
            assert r.status_code == 200 and "accounts" in r.text
            m = rings[0]["members_sample"][0]
            j = c.get(f"/check/{m}").json()
            assert j["known"] and j["in_a_ring"], j
            assert c.get(f"/check/{m}/card").status_code == 200
            assert c.get("/findings").status_code == 200
            print("OK", h["accounts_served"], h["bundle_mb"])
        """)
        (tmp / "run_check.py").write_text(script)
        out = subprocess.run([sys.executable, "run_check.py"], cwd=tmp,
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, f"console failed from the bundle:\n{out.stdout}\n{out.stderr}"
        assert out.stdout.startswith("OK"), out.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@needs_bundle
def test_every_bundle_file_is_actually_committed():
    """The bundle only works from a clone if git is carrying it. The blanket
    `*.parquet` rule in .gitignore silently dropped the account table on the
    first attempt, and every local test still passed because the file was
    sitting there untracked."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files", D.BUNDLE_DIR],
                             cwd=ROOT, capture_output=True, text=True, check=True)
    have = {line.split("/")[-1] for line in tracked.stdout.split()}
    on_disk = {f.name for f in BUNDLE.iterdir() if f.is_file()}
    missing = on_disk - have
    assert not missing, f"the bundle needs these but git is not carrying them: {sorted(missing)}"
