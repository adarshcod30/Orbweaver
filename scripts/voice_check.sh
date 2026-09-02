#!/usr/bin/env bash
# Every committed file is written in my own voice. This fails the check if
# any tooling reference or planning vocabulary has leaked into a tracked file.
#
# "cursor" is matched only as a standalone word, because `cursor: pointer` is
# an ordinary CSS property and matching it was a false positive.
set -u
PATTERN='claude|anthropic|copilot|\bcursor\b|co-authored|generated with|tier ?[12]|time-?box|deadline|rubric|judges|phase 0|non-negotiable|ORBWEAVER_SPEC'
if git grep -nIiE "$PATTERN" -- . ':!scripts/voice_check.sh' > /tmp/voice_hits.txt 2>/dev/null; then
  echo "voice check FAILED:"
  cat /tmp/voice_hits.txt
  exit 1
fi
if git ls-files | grep -iE 'claude|orbweaver_spec|^notes/'; then
  echo "voice check FAILED: a private file is tracked"
  exit 1
fi
echo "voice check passed"
