# ==============================================================================
# File: merge_runs.py
# Description: Combines the twelve per-configuration result files written by
#   run_all.sh into the single JSON the paper is built from, and checks that
#   the set is complete and internally consistent before doing so.
#
#   Each configuration runs in its own process so the cores are all used, which
#   means each writes its own file. A paper built from a partial set would
#   quietly report a comparison with arms missing, so the merge refuses unless
#   every task and encoding is present and every run agrees on the seed, the
#   step count and the widths.
#
# Usage: py scripts/merge_runs.py [--runs data/runs] [--out data/length_generalization.json]
# Tech Stack: Python 3.10+
# Author: Amey Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# ==============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["add", "addmod", "mulmod"]
ARMS = ["absolute", "rope", "abacus-input", "abacus-inject"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "data" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "data" / "length_generalization.json"))
    ap.add_argument("--partial", action="store_true",
                    help="merge whatever is present, for inspecting a run in flight")
    args = ap.parse_args()

    runs = Path(args.runs)
    arms, missing, meta = [], [], {}

    for op in TASKS:
        for arm in ARMS:
            path = runs / f"{op}-{arm}.json"
            if not path.is_file():
                missing.append(f"{op}-{arm}")
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            if not d.get("arms"):
                missing.append(f"{op}-{arm} (no arms recorded)")
                continue
            rec = d["arms"][0]
            arms.append(rec)
            # every run must agree on the things that make them comparable
            key = (d["seed"], d["steps"], tuple(d["train_widths"]),
                   tuple(d["test_widths"]))
            meta.setdefault(key, []).append(f"{op}-{arm}")

    print(f"  found {len(arms)}/{len(TASKS) * len(ARMS)} configurations")
    if missing:
        print(f"  missing: {', '.join(missing)}")

    if len(meta) > 1:
        print("  FAIL  runs disagree on seed, steps or widths:")
        for key, who in meta.items():
            print(f"    {key} <- {', '.join(who)}")
        return 1

    if missing and not args.partial:
        print("  refusing to write a partial comparison; pass --partial to override")
        return 1

    if not arms:
        print("  nothing to merge")
        return 1

    seed, steps, train_widths, test_widths = next(iter(meta))
    out = {
        "seed": seed,
        "steps": steps,
        "train_widths": list(train_widths),
        "test_widths": list(test_widths),
        "complete": not missing,
        "arms": arms,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {args.out}")

    print(f"\n  {'task':8s} {'encoding':16s} " +
          " ".join(f"w={w}" for w in test_widths))
    for rec in arms:
        cells = " ".join(f"{rec['accuracy'][str(w)]:5.1%}" for w in test_widths)
        print(f"  {rec['op']:8s} {rec['encoding']:16s} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
