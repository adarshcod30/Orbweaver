#!/usr/bin/env bash
# Amazon and YelpChi, for the generalisation check.
#
# These come from the original CARE-GNN release (Dou et al., CIKM 2020) rather
# than the GADBench bundle. GADBench ships its copies in DGL's serialisation
# format and DGL publishes no wheel for Python 3.13 on arm64 macOS; the
# underlying data is the same and scipy reads these .mat files directly.
set -u
OUT="data/raw/gadbench"
mkdir -p "$OUT"
BASE="https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data"

for f in Amazon YelpChi; do
  if [ -f "$OUT/$f.mat" ]; then echo "SKIP $f.mat (present)"; continue; fi
  echo "GET  $f.zip"
  curl -L -C - --retry 8 --retry-all-errors --connect-timeout 30 \
       -o "$OUT/$f.zip" "$BASE/$f.zip"
  unzip -o -q "$OUT/$f.zip" -d "$OUT"
  echo "DONE $f.mat ($(wc -c < "$OUT/$f.mat") bytes)"
done
