#!/usr/bin/env bash
# Every committed file is written in my own voice. This fails the check if
# any tooling reference or planning vocabulary has leaked into a tracked file.
#
# "cursor" matches bare, then the CSS declaration `cursor:pointer` (or
# `cursor: pointer`) is filtered back out as a known-benign hit, rather than
# trying to exclude it with a \b word-boundary in the pattern itself: \b is a
# GNU regex extension, not POSIX ERE, and macOS's git grep silently does not
# support it at all - so a \bcursor\b pattern quietly matched nothing there,
# made the check pass by accident on this machine, and still matched
# "cursor:pointer" as a whole word on Linux CI, where \b does work as
# intended. Filtering after a plain match is portable either way.
set -u
PATTERN='claude|anthropic|copilot|cursor|co-authored|generated with|tier ?[12]|time-?box|deadline|rubric|judges|phase 0|non-negotiable|ORBWEAVER_SPEC'
git grep -nIiE "$PATTERN" -- . ':!scripts/voice_check.sh' 2>/dev/null \
  | grep -viE 'cursor: *pointer' > /tmp/voice_hits.txt
if [ -s /tmp/voice_hits.txt ]; then
  echo "voice check FAILED:"
  cat /tmp/voice_hits.txt
  exit 1
fi
if git ls-files | grep -iE 'claude|orbweaver_spec|^notes/'; then
  echo "voice check FAILED: a private file is tracked"
  exit 1
fi
echo "voice check passed"
