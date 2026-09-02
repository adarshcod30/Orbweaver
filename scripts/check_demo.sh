#!/usr/bin/env bash
# Ask the console about two accounts: one inside a surfaced ring, and one from
# a legitimate co-located cluster. The contrast is the point - the same
# endpoint, the same evidence fields, and a very different answer.
#
#   make console          # in one terminal
#   ./scripts/check_demo.sh
set -u
HOST="${ORBWEAVER_HOST:-http://127.0.0.1:8000}"
PROC="data/processed"

pick() {  # pick an account id out of a JSON artefact
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
which = sys.argv[1]
proc = Path("data/processed")
if which == "ring":
    for name in ("ring_report_deep.json", "ring_report.json"):
        p = proc / name
        if p.exists():
            cases = json.loads(p.read_text()).get("case_files", [])
            for c in cases:
                if c.get("members_sample"):
                    print(c["members_sample"][0]); raise SystemExit
    print("")
else:
    p = proc / "hostel_test.json"
    if p.exists():
        worst = json.loads(p.read_text()).get("worst_cases", [])
        if worst:
            print(worst[0].get("entity", "")); raise SystemExit
    print("")
PY
}

RING_ACCOUNT="$(pick ring)"
if [ -z "$RING_ACCOUNT" ]; then
  echo "no ring artefact found - run 'make rings' first"; exit 1
fi

echo "=== an account inside a surfaced ring ==="
curl -s "$HOST/check/$RING_ACCOUNT" | python3 -m json.tool

echo
echo "=== an account with no ring around it ==="
curl -s "$HOST/check/1" | python3 -m json.tool

echo
echo "Card view for the browser:  $HOST/check/$RING_ACCOUNT/card"
