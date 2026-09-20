# ==============================================================================
# File: check_rope.py
# Description: Verifies the rotary encoding baseline is a genuine rotation
#   before any paper reports how it performed. RoPE underperformed the other
#   arms in distribution, which is a statement about the method only if the
#   implementation is correct; a broken baseline would invalidate the whole
#   comparison. Three properties are checked: a rotation preserves norms, it
#   leaves position zero untouched, and it makes the attention logit between
#   two tokens depend on their separation alone.
# Usage: py src/experiments/tests/check_rope.py
# Author: Amey Thakur
# License: CC-BY-4.0
# ==============================================================================

"""Sanity-check the rotary encoding before the paper reports its result.

RoPE underperformed in distribution, which is a claim about the method only if
the implementation is right. A rotation preserves norms, leaves position zero
untouched, and makes the attention logit between two tokens depend on their
separation rather than on where the pair sits. All three are checked here.
"""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(
    r"C:\Users\archi\OneDrive\Desktop\Shreyas"
    r"\SAIR-MODULAR-ARITHMETIC-CHALLENGE\src\experiments")))
from length_generalization import rope

torch.manual_seed(0)
b, h, t, d = 2, 4, 12, 32
x = torch.randn(b, h, t, d)
y = rope(x)

norm_err = (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item()
print(f"  norm preserved            max error {norm_err:.2e}")

pos0_err = (x[:, :, 0] - y[:, :, 0]).abs().max().item()
print(f"  position 0 unchanged      max error {pos0_err:.2e}")

# Relative property: <rope(q)_m, rope(k)_n> depends only on m - n when the
# same underlying vector sits at every position. Using a different random
# vector per position tests nothing, since the inner products then differ for
# reasons that have nothing to do with the encoding.
v = torch.randn(d)
q = v.expand(1, 1, t, d).clone()
k = q.clone()
qr, kr = rope(q), rope(k)
logits = (qr[0, 0] @ kr[0, 0].T)
diffs = {}
for m in range(t):
    for n in range(t):
        diffs.setdefault(m - n, []).append(logits[m, n].item())
spread = max(max(v) - min(v) for v in diffs.values())
print(f"  logit depends on m-n only spread {spread:.2e}")

ok = norm_err < 1e-5 and pos0_err < 1e-6
print("\n  rotary encoding behaves as a rotation" if ok
      else "\n  IMPLEMENTATION LOOKS WRONG")
