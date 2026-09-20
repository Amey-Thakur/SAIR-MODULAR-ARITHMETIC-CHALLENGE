#!/bin/sh
# One process per (task, encoding) so the eight cores are all used. Torch
# scales poorly past a couple of threads on a model this small, so two threads
# each and six at a time beats one big multi-threaded run.
mkdir -p data/runs
for op in add addmod mulmod; do
  for arm in absolute rope abacus-input abacus-inject; do
    echo "$op $arm"
  done
done | xargs -P 6 -n 2 sh -c '
  python src/experiments/length_generalization.py \
    --steps 6000 --threads 2 --ops "$0" --arms "$1" \
    --out "data/runs/$0-$1.json" > "data/runs/$0-$1.log" 2>&1
  echo "done $0 $1"
'
