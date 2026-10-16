# ==============================================================================
# File: derive_stats.py
# Description: The derived figures the paper quotes, with intervals, so that
#   no number in the manuscript is typed from a glance at a table.
#
#   Every accuracy here is a proportion of 1,000 exact-match trials, so it
#   carries sampling error, and a difference between two encodings is only
#   worth reporting if it survives that error. Wilson score intervals are used
#   rather than the normal approximation because several cells sit at or near
#   zero, where the normal interval is meaningless.
#
# Usage: py scripts/derive_stats.py [--data data/length_generalization.json]
# Tech Stack: Python 3.10+
# Author: Amey Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Z = 1.959964
TASKS = [("add", "a + b"), ("addmod", "(a + b) mod p"), ("mulmod", "(a x b) mod p")]
ARMS = ["absolute", "rope", "abacus-input", "abacus-inject", "abacus-offset"]


def wilson(x: int, n: int):
    """Score interval for x successes in n trials; behaves at 0 and at n."""
    if n == 0:
        return 0.0, 0.0
    p = x / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "length_generalization.json"))
    args = ap.parse_args()

    d = json.loads(Path(args.data).read_text(encoding="utf-8"))
    widths = [int(w) for w in d["test_widths"]]
    train_max = max(d["train_widths"])
    by = {(a["op"], a["encoding"]): a for a in d["arms"]}

    print(f"seed {d['seed']}, {d['steps']} steps, trained on widths "
          f"{d['train_widths']}, scored on {widths}")
    print(f"complete set: {d.get('complete')}")

    for op, label in TASKS:
        print(f"\n=== {label} ===")
        print(f"  {'encoding':16s} " + " ".join(f"{'w=' + str(w):>14s}" for w in widths))
        for arm in ARMS:
            rec = by.get((op, arm))
            if not rec:
                print(f"  {arm:16s} (missing)")
                continue
            cells = []
            for w in widths:
                acc = rec["accuracy"][str(w)]
                n = rec["n_test"][str(w)]
                lo, hi = wilson(round(acc * n), n)
                cells.append(f"{acc:5.1%} [{lo:.2f},{hi:.2f}]".rjust(14))
            print(f"  {arm:16s} " + " ".join(cells))

        # the comparison the paper turns on: inside versus beyond training
        print(f"  {'':16s} " + "  in-distribution      out-of-distribution")
        for arm in ARMS:
            rec = by.get((op, arm))
            if not rec:
                continue
            ind_n = sum(rec["n_test"][str(w)] for w in widths if w <= train_max)
            ind_x = sum(round(rec["accuracy"][str(w)] * rec["n_test"][str(w)])
                        for w in widths if w <= train_max)
            ood_n = sum(rec["n_test"][str(w)] for w in widths if w > train_max)
            ood_x = sum(round(rec["accuracy"][str(w)] * rec["n_test"][str(w)])
                        for w in widths if w > train_max)
            il, ih = wilson(ind_x, ind_n)
            ol, oh = wilson(ood_x, ood_n)
            print(f"  {arm:16s} {ind_x:5d}/{ind_n:<5d} = {ind_x / ind_n:6.1%} "
                  f"[{il:.3f},{ih:.3f}]   {ood_x:5d}/{ood_n:<5d} = "
                  f"{ood_x / ood_n:6.1%} [{ol:.3f},{oh:.3f}]")

    # H2: does re-injecting the significance signal beat injecting it once
    print("\n=== re-injection, out of distribution ===")
    for op, label in TASKS:
        a, b = by.get((op, "abacus-input")), by.get((op, "abacus-inject"))
        if not (a and b):
            continue
        def ood(rec):
            n = sum(rec["n_test"][str(w)] for w in widths if w > train_max)
            x = sum(round(rec["accuracy"][str(w)] * rec["n_test"][str(w)])
                    for w in widths if w > train_max)
            return x, n
        ax, an = ood(a)
        bx, bn = ood(b)
        al, ah = wilson(ax, an)
        bl, bh = wilson(bx, bn)
        overlap = not (ah < bl or bh < al)
        print(f"  {label:16s} input {ax / an:6.1%} [{al:.3f},{ah:.3f}]  "
              f"injected {bx / bn:6.1%} [{bl:.3f},{bh:.3f}]  "
              f"{'intervals overlap' if overlap else 'intervals disjoint'}")

    print("\n=== timing ===")
    for op, _ in TASKS:
        for arm in ARMS:
            rec = by.get((op, arm))
            if rec:
                print(f"  {op:8s} {arm:16s} {rec['params']:,} params, "
                      f"{rec['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
