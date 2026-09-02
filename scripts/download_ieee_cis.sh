#!/usr/bin/env bash
# IEEE-CIS fraud detection (Vesta), for the payment-processor graph.
#
# Needs Kaggle credentials and the competition rules accepted in a browser.
# A 403 means the rules have not been accepted - that cannot be fixed from
# here, so the script says so and stops rather than retrying.
#
# Credentials live at ~/.config/kaggle/kaggle.json on newer clients and
# ~/.kaggle/kaggle.json on older ones; both are accepted.
set -u
OUT="data/raw/ieee_cis"
COMP="ieee-fraud-detection"
NEEDED="train_transaction.csv train_identity.csv"

if [ -f "$HOME/.config/kaggle/kaggle.json" ]; then
  export KAGGLE_CONFIG_DIR="$HOME/.config/kaggle"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
  export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
else
  echo "no kaggle.json found in ~/.config/kaggle or ~/.kaggle"; exit 1
fi

have_all=1
for f in $NEEDED; do
  [ -f "$OUT/$f" ] || have_all=0
done
if [ "$have_all" = "1" ]; then
  echo "SKIP: the labelled files are already present"
  exit 0
fi

# Free space guard. The archive is about 1.2 GB and expands to roughly 2 GB.
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ "$FREE_GB" -lt 6 ]; then
  echo "only ${FREE_GB} GB free; need 6 GB. Stopping rather than filling the disk."
  exit 1
fi

mkdir -p "$OUT"
echo "downloading $COMP (about 1.2 GB) ..."
if ! python3 -m kaggle competitions download -c "$COMP" -p "$OUT" 2>"$OUT/.err"; then
  if grep -qiE "403|forbidden|rules" "$OUT/.err"; then
    echo "Kaggle returned 403. The competition rules for $COMP have to be"
    echo "accepted in a browser first; that cannot be done from here."
    sed -n '1,5p' "$OUT/.err"
    exit 2
  fi
  echo "download failed:"; sed -n '1,10p' "$OUT/.err"; exit 1
fi

echo "extracting only the labelled files ..."
for f in $NEEDED; do
  unzip -o -q "$OUT/$COMP.zip" "$f" -d "$OUT" || true
done
rm -f "$OUT/$COMP.zip" "$OUT/.err"
ls -la "$OUT"
