#!/bin/sh
# Waits for the twelve baseline runs to finish, then trains the offset arm on
# all three rungs. Queued rather than run alongside, because the machine is
# already fully subscribed and adding work would slow everything without
# finishing anything sooner.
cd "$(dirname "$0")" || exit 1

while [ "$(ls data/runs/*.json 2>/dev/null | wc -l)" -lt 12 ]; do
  sleep 60
done

echo "baseline complete, starting the offset arm"
for op in add addmod mulmod; do
  echo "$op"
done | xargs -P 3 -n 1 sh -c '
  python src/experiments/length_generalization.py \
    --steps 6000 --threads 2 --ops "$0" --arms abacus-offset \
    --out "data/runs/$0-abacus-offset.json" > "data/runs/$0-abacus-offset.log" 2>&1
  echo "done $0 abacus-offset"
'
echo "offset arm complete"
