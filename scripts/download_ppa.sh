#!/usr/bin/env bash
# Download the PPA dataset from OSF (PromoGuardian, IEEE S&P 2026).
# Resumable: safe to re-run; curl -C - continues partial files.
set -u
VO="671050154acf4c0fa6b86a9337e74c2c"
BASE="https://files.us.osf.io/v1/resources/rasje/providers/osfstorage"
OUT="data/raw/ppa"
mkdir -p "$OUT"

# name<TAB>osf_id<TAB>expected_md5<TAB>expected_bytes
FILES=$(cat <<'TSV'
readme.md	678dbaba79e752e765282c71	1077d4254e8bb3ec6ca3a51cc2c8c429	990
node.csv	678a3b186f4ce207afdd63ac	c1728a46b76da1d17401767066dab5e6	142373615
edge.csv	678a3f06dd20491e69dd66a0	279514689a55d5fad4625044f2a6322f	675362394
order_train.csv	68a981a8735b388b45affe0b	-	1625195771
order_test.csv	68a988213a63b29047bcbe0e	-	1559383792
TSV
)

echo "PPA download started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
while IFS=$'\t' read -r name id md5 bytes; do
  [ -z "$name" ] && continue
  dest="$OUT/$name"
  if [ -f "$dest" ] && [ "$(wc -c < "$dest" | tr -d ' ')" = "$bytes" ]; then
    echo "SKIP $name (already complete, $bytes bytes)"
    continue
  fi
  echo "GET  $name ($bytes bytes) -> $dest"
  curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
       -o "$dest" "$BASE/$id?view_only=$VO"
  rc=$?
  got=$(wc -c < "$dest" 2>/dev/null | tr -d ' ')
  echo "DONE $name rc=$rc bytes=$got expected=$bytes"
done <<< "$FILES"

echo "--- verification ---"
while IFS=$'\t' read -r name id md5 bytes; do
  [ -z "$name" ] && continue
  dest="$OUT/$name"
  got=$(wc -c < "$dest" 2>/dev/null | tr -d ' ')
  if [ "$got" != "$bytes" ]; then echo "SIZE-MISMATCH $name got=$got want=$bytes"; continue; fi
  if [ "$md5" != "-" ]; then
    actual=$(md5 -q "$dest")
    [ "$actual" = "$md5" ] && echo "MD5-OK   $name" || echo "MD5-FAIL $name got=$actual want=$md5"
  else
    echo "SIZE-OK  $name (no published md5) md5=$(md5 -q "$dest")"
  fi
done <<< "$FILES"
echo "PPA download finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
