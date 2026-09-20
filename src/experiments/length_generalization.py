# ==============================================================================
# File: length_generalization.py
# Description: The controlled experiment behind the paper's central claim, that
#   describing a digit by what it is WORTH generalises to operand widths that
#   describing it by WHERE IT SITS does not.
#
#   Four arms differ only in how a token's place in the problem is encoded.
#   Everything else, parameter count, data, optimiser, schedule and seed, is
#   held fixed, so a difference in out-of-distribution accuracy is attributable
#   to the encoding and to nothing else.
#
#     absolute       learned embedding of the index in the sequence
#     rope           rotary encoding of that same index
#     abacus-input   learned embedding of the digit's SIGNIFICANCE, its place
#                    value inside its own number, added once at the input
#     abacus-inject  the same significance signal re-added at every layer,
#                    which is the architecture this repository submitted
#
#   Models train on operands of at most TRAIN_MAX_W digits and are scored on
#   widths beyond that. Accuracy is exact match on the whole answer, because a
#   remainder that is off by one is simply wrong.
#
# Usage: py src/experiments/length_generalization.py [--steps N] [--quick]
# Tech Stack: Python 3.10+, PyTorch 2.0+
# Author: Amey Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# problem setup
# ---------------------------------------------------------------------------

TRAIN_WIDTHS = [2, 3, 4]           # operand widths the model is trained on
TEST_WIDTHS = [2, 3, 4, 5, 6, 7, 8]  # 5 and beyond are out of distribution
# each rung: how the answer is formed, whether a modulus appears, and how
# many digits the answer occupies given operands of width w
TASKS = {
    "add":    {"modulus": False, "answer_width": lambda w: w + 1},
    "addmod": {"modulus": True,  "answer_width": lambda w: w},
    "mulmod": {"modulus": True,  "answer_width": lambda w: w},
}
SEED = 20260920

# character-level vocabulary; no sub-word merges, which fragment numbers
CHARS = ["<pad>", "<bos>", "<eos>"] + [str(d) for d in range(10)] + ["*", "+", "%", "="]
STOI = {c: i for i, c in enumerate(CHARS)}
PAD, BOS, EOS = STOI["<pad>"], STOI["<bos>"], STOI["<eos>"]
VOCAB = len(CHARS)

# significance is the place value of a digit inside its own number, so the
# units digit is 0 whatever the width of the number it belongs to. Non-digits
# get their own slot rather than a number's slot.
SIG_NONE = 0
SIG_BASE = 1                       # significance s occupies slot SIG_BASE + s
MAX_SIG = 24
# how far the significance indices may be shifted during training; large
# enough that the rows needed at the widest test width are well exercised
MAX_OFFSET = 16


def primes_with(digits: int, rng: random.Random, count: int) -> list[int]:
    """Primes with exactly this many decimal digits, by trial division, which
    is ample at these widths and keeps the generator dependency-free."""
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    out = []
    while len(out) < count:
        n = rng.randrange(lo | 1, hi, 2)
        if n > 2 and all(n % f for f in range(3, int(math.isqrt(n)) + 1, 2)) and n % 2:
            out.append(n)
    return out


def render(a: int, b: int, p: int, r: int, op: str, width: int):
    """The problem as tokens, with the significance of every token beside it.

    Layout is `<bos> a OP b [% p] = r <eos>`. Every number is written at a fixed
    width, zero padded, so a digit of a given significance sits at a fixed
    sequence index for a fixed operand width. That is the most generous
    possible setting for the positional arms: within a width they could in
    principle memorise the layout, and only a change of width disturbs them.

    The answer is written least significant digit first, so that generation
    order matches the direction carries travel. Significance is unaffected,
    being a property of the digit rather than of where it is written.
    """
    spec = TASKS[op]
    aw = spec["answer_width"](width)
    ids, sig = [BOS], [SIG_NONE]

    def number(value: str):
        n = len(value)
        for i, ch in enumerate(value):
            ids.append(STOI[ch])
            sig.append(SIG_BASE + min(n - 1 - i, MAX_SIG - 1))

    number(str(a).zfill(width))
    ids.append(STOI["*" if op == "mulmod" else "+"]); sig.append(SIG_NONE)
    number(str(b).zfill(width))
    if spec["modulus"]:
        ids.append(STOI["%"]); sig.append(SIG_NONE)
        number(str(p).zfill(width))
    ids.append(STOI["="]); sig.append(SIG_NONE)
    answer_at = len(ids)
    for i, ch in enumerate(str(r).zfill(aw)[::-1]):
        ids.append(STOI[ch])
        sig.append(SIG_BASE + min(i, MAX_SIG - 1))
    ids.append(EOS); sig.append(SIG_NONE)
    return ids, sig, answer_at


def make_examples(widths, n, rng, primes, op):
    """n problems, widths drawn uniformly from those given.

    For the modular rungs the operands are uniform below the modulus, so the
    answer is a residue of the same width. For plain addition they are uniform
    over the full width and the modulus is unused.
    """
    spec = TASKS[op]
    out = []
    for _ in range(n):
        w = rng.choice(widths)
        if spec["modulus"]:
            p = rng.choice(primes[w])
            a, b = rng.randrange(p), rng.randrange(p)
            r = (a + b) % p if op == "addmod" else (a * b) % p
        else:
            p = 0
            lo = 10 ** (w - 1)
            a, b = rng.randrange(lo, 10 ** w), rng.randrange(lo, 10 ** w)
            r = a + b
        out.append(render(a, b, p, r, op, w) + (w,))
    return out


def collate(batch, device):
    """Pad to the longest item and mark where each answer begins."""
    width = max(len(ids) for ids, _, _, _ in batch)
    x = torch.full((len(batch), width), PAD, dtype=torch.long)
    s = torch.zeros((len(batch), width), dtype=torch.long)
    starts = []
    for i, (ids, sig, at, _) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids)
        s[i, :len(sig)] = torch.tensor(sig)
        starts.append(at)
    return x.to(device), s.to(device), starts


# ---------------------------------------------------------------------------
# the model, with the encoding as the only moving part
# ---------------------------------------------------------------------------

def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position encoding applied to a (B, H, T, D) tensor."""
    b, h, t, d = x.shape
    half = d // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    ang = torch.arange(t, device=x.device).unsqueeze(1) * freq.unsqueeze(0)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Layer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, use_rope):
        super().__init__()
        self.h, self.dk, self.use_rope = n_heads, d_model // n_heads, use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, self.h, self.dk).transpose(1, 2) for z in (q, k, v))
        if self.use_rope:
            q, k = rope(q), rope(k)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.ffn(self.n2(x)))


class Model(nn.Module):
    """One architecture; `encoding` selects how place is described."""

    def __init__(self, encoding, d_model=128, n_heads=4, n_layers=3, d_ff=256,
                 dropout=0.0, max_len=64):
        super().__init__()
        self.encoding = encoding
        self.tok = nn.Embedding(VOCAB, d_model)
        # every arm gets exactly one extra embedding table, so parameter counts
        # match and the comparison is not confounded by capacity
        if encoding == "absolute":
            self.pos = nn.Embedding(max_len, d_model)
        elif encoding in ("abacus-input", "abacus-inject", "abacus-offset"):
            self.pos = nn.Embedding(SIG_BASE + MAX_SIG, d_model)
        else:                                   # rope carries no table
            self.pos = nn.Embedding(1, d_model)
        self.layers = nn.ModuleList([
            Layer(d_model, n_heads, d_ff, dropout, use_rope=(encoding == "rope"))
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, ids, sig):
        b, t = ids.shape
        x = self.tok(ids)
        if self.encoding == "absolute":
            place = self.pos(torch.arange(t, device=ids.device))[None]
        elif self.encoding in ("abacus-input", "abacus-inject", "abacus-offset"):
            place = self.pos(sig)
        else:
            place = None
        if place is not None:
            x = x + place
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=ids.device))
        for layer in self.layers:
            # re-adding the signal at every boundary is the architecture the
            # repository submitted; the input-only arm isolates its effect
            if self.encoding == "abacus-inject":
                x = x + place
            x = layer(x, mask)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# train and score
# ---------------------------------------------------------------------------

def answer_loss(logits, ids, at, width):
    """Cross-entropy on the answer digits only. Scoring the question as well
    would let a model look good by predicting operands it was just shown.

    Every example in a batch shares one width, so the answer starts at the same
    index throughout and this is one vectorised call rather than a Python loop.
    """
    pred = logits[:, at - 1:at - 1 + width]
    gold = ids[:, at:at + width]
    return F.cross_entropy(pred.reshape(-1, VOCAB), gold.reshape(-1))


@torch.no_grad()
def exact_match(model, examples, width, device, batch=256):
    """Fraction of problems whose every answer digit is right.

    Digits are decoded greedily and fed back, so an early mistake is not
    repaired later, which is the behaviour the competition scores.
    """
    model.eval()
    right = 0
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        ids, sig, starts = collate(chunk, device)
        at = starts[0]
        assert all(s == at for s in starts), "ragged answer starts in one batch"
        work, wsig = ids[:, :at].clone(), sig[:, :at].clone()
        for step in range(width):
            nxt = model(work, wsig)[:, -1].argmax(-1, keepdim=True)
            work = torch.cat([work, nxt], dim=1)
            wsig = torch.cat([wsig, sig[:, at + step:at + step + 1]], dim=1)
        right += (work[:, at:] == ids[:, at:at + width]).all(dim=1).sum().item()
    model.train()
    return right / max(1, len(examples))


def build_data(rng, op):
    """Training pools per width, and a disjoint held-out set per test width.

    Held-out problems that appear verbatim in training are dropped, so a model
    cannot score by recitation. Primes are drawn per width and shared across
    arms, so every arm sees the same problems in the same order.
    """
    primes = {w: primes_with(w, rng, 96) for w in sorted(set(TRAIN_WIDTHS) | set(TEST_WIDTHS))}
    train_by_w = {w: make_examples([w], 20_000, rng, primes, op) for w in TRAIN_WIDTHS}
    seen = {tuple(e[0]) for pool in train_by_w.values() for e in pool}
    tests = {}
    for w in TEST_WIDTHS:
        pool = make_examples([w], 1500, rng, primes, op)
        tests[w] = [e for e in pool if tuple(e[0]) not in seen][:1000]
    return train_by_w, tests


def run_arm(encoding, op, steps, device, log):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    train_by_w, tests = build_data(rng, op)

    model = Model(encoding).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                total_steps=steps, pct_start=0.1)
    log(f"[{op}/{encoding}] {params:,} parameters")

    t0, bs, curve = time.time(), 256, []
    for step in range(steps):
        # one width per batch, cycled so every training width gets an equal
        # share of gradient steps
        w = TRAIN_WIDTHS[step % len(TRAIN_WIDTHS)]
        pool = train_by_w[w]
        batch = [pool[rng.randrange(len(pool))] for _ in range(bs)]
        ids, sig, starts = collate(batch, device)
        if encoding == "abacus-offset":
            # shift every digit's significance by one random amount, so
            # rows far beyond the training widths receive gradients; the
            # non-digit slot is row 0 and must not move
            k = rng.randrange(0, MAX_OFFSET + 1)
            sig = torch.where(sig > 0,
                              torch.clamp(sig + k, max=SIG_BASE + MAX_SIG - 1),
                              sig)
        aw = TASKS[op]['answer_width'](w)
        loss = answer_loss(model(ids, sig), ids, starts[0], aw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(1, steps // 12) == 0:
            curve.append({"step": step + 1, "loss": round(loss.item(), 4)})
            log(f"[{op}/{encoding}] step {step + 1}/{steps} "
                f"loss {loss.item():.4f} ({time.time() - t0:.0f}s)")

    acc = {w: exact_match(model, tests[w], TASKS[op]['answer_width'](w), device)
           for w in TEST_WIDTHS}
    log(f"[{op}/{encoding}] " + "  ".join(f"w={w} {a:.1%}" for w, a in acc.items()))
    return {"op": op, "encoding": encoding, "params": params,
            "accuracy": {str(w): acc[w] for w in TEST_WIDTHS},
            "n_test": {str(w): len(tests[w]) for w in TEST_WIDTHS},
            "curve": curve, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--quick", action="store_true", help="tiny run, to check wiring")
    ap.add_argument("--ops", default="add,addmod,mulmod")
    ap.add_argument("--arms",
                    default="absolute,rope,abacus-input,abacus-inject,abacus-offset")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch threads; one per process when arms run in parallel")
    ap.add_argument("--out", default=str(ROOT / "data" / "length_generalization.json"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    steps = 40 if args.quick else args.steps
    device = torch.device("cpu")
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"device {device}, {torch.get_num_threads()} threads, {steps} steps per arm")
    log(f"train widths {TRAIN_WIDTHS}, test widths {TEST_WIDTHS}")

    results = []
    for op in args.ops.split(","):
        for arm in args.arms.split(","):
            results.append(run_arm(arm, op, steps, device, log))
            Path(args.out).write_text(json.dumps(
                {"seed": SEED, "steps": steps, "train_widths": TRAIN_WIDTHS,
                 "test_widths": TEST_WIDTHS, "arms": results, "log": lines},
                indent=2), encoding="utf-8")   # written after each arm, so a
                                               # long run is never lost entirely
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
