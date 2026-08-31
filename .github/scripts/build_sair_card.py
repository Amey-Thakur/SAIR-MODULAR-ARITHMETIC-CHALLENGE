# ==============================================================================
# File: build_sair_card.py
# Description: Builds the repository card in the visual language of the SAIR
#   Foundation's hero art for the Modular Arithmetic Challenge: a vertical
#   copper to aubergine gradient and white line work, read left to right.
#   Three operands go in, a network stands between, and the residue comes out.
#   The numbers on the card are a real instance: the build recomputes
#   (a * b) mod p and refuses to render if the printed residue is not the
#   answer. Every element declares a box, and no two boxes may intersect.
# Usage: python .github/scripts/build_sair_card.py .github/assets
# Tech Stack: Python 3.10+, Pillow
# ==============================================================================

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 675
SS = 3
FRAMES, DURATION = 30, 60

TOP = (0x8F, 0x57, 0x43)     # sampled from the top row of SAIR's hero
BOT = (0x34, 0x08, 0x25)     # and its bottom row, the SAIR card colour
INK = (255, 255, 255)
STROKE = 4

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")
UIB = "C:/Windows/Fonts/segoeuib.ttf"
# Cambria Math is the only face installed here carrying U+211A and U+22EF, so
# the mathematics is set in it rather than approximated in a sans face.
MATH, MATH_IDX = "C:/Windows/Fonts/cambria.ttc", 1

# -- the instance ----------------------------------------------------------
# A genuine triple, checked at build time. A card that shows arithmetic should
# show arithmetic that is true.

P = 1000000000000000000000000000000000012397
A = 781695396820365451993922799535871154328
B = 604977318445290163872615509437266081159
R = (A * B) % P

MUL = "\u2217"                            # U+2217 ASTERISK OPERATOR
FORMULA = "r = (a " + MUL + " b) mod p"

# -- layout -----------------------------------------------------------------
# Built on a 56px margin: three panels and two gaps span x 56 to x 1144, and
# the panels share one height, so the row is symmetric about the card's middle.

M = 56
LOGO = (M, 44, M + 240, 96)
PANEL_Y = (168, 604)
P1 = (M, 290)
P2 = (344, 856)
P3 = (910, 1144)
GAP1 = (P1[1], P2[0])
GAP2 = (P2[1], P3[0])
RADIUS = 22

BOXES = {
    "logo": LOGO,
    "operands": (P1[0], PANEL_Y[0], P1[1], PANEL_Y[1]),
    "network": (P2[0], PANEL_Y[0], P2[1], PANEL_Y[1]),
    "residue": (P3[0], PANEL_Y[0], P3[1], PANEL_Y[1]),
}


def assert_no_overlap():
    if (A * B) % P != R:
        raise SystemExit("the printed residue is not (a * b) mod p")
    names = list(BOXES)
    bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = BOXES[names[i]], BOXES[names[j]]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox > 0 and oy > 0:
                bad.append(f"{names[i]} and {names[j]} overlap by "
                           f"{ox:.0f}x{oy:.0f}")
    if bad:
        raise SystemExit("layout collision: " + "; ".join(bad))
    xs = [v for b in BOXES.values() for v in (b[0], b[2])]
    ys = [v for b in BOXES.values() for v in (b[1], b[3])]
    print(f"  extent x {min(xs):.0f}-{max(xs):.0f}  y {min(ys):.0f}-{max(ys):.0f}")
    print(f"  (a * b) mod p verified, {len(str(R))} digits")



_scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def f(path, size, index=0):
    return ImageFont.truetype(path, max(1, int(round(size * SS))), index=index)


def fit_width(path, text, target, index=0):
    lo, hi = 4.0, 400.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _scratch.textlength(text, font=f(path, mid, index)) / SS < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_height(path, text, target, index=0):
    lo, hi = 4.0, 800.0
    for _ in range(40):
        mid = (lo + hi) / 2
        bb = _scratch.textbbox((0, 0), text, font=f(path, mid, index))
        if (bb[3] - bb[1]) / SS < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bg_at(py):
    t = max(0.0, min(1.0, py / (H - 1.0)))
    return tuple(round(TOP[i] + (BOT[i] - TOP[i]) * t) for i in range(3))


def dim(py, amount):
    b = bg_at(py)
    return tuple(round(b[i] + (INK[i] - b[i]) * amount) for i in range(3))


def ease(t):
    return t * t * (3 - 2 * t)


OFFSET = [0.0, 0.0]          # set once by centre_offset(), then applied to all


class Pen:
    def __init__(self, draw):
        self.d = draw

    def _p(self, pts):
        ox, oy = OFFSET
        return [((p[0] + ox) * SS, (p[1] + oy) * SS) for p in pts]

    def line(self, pts, colour=INK, width=STROKE, joint="curve"):
        if len(pts) < 2:
            return
        self.d.line(self._p(pts), fill=colour,
                    width=int(round(width * SS)), joint=joint)

    def circle(self, cx, cy, r, colour=INK, width=STROKE):
        ox, oy = OFFSET
        self.d.ellipse([(cx + ox - r) * SS, (cy + oy - r) * SS,
                        (cx + ox + r) * SS, (cy + oy + r) * SS],
                       outline=colour, width=int(round(width * SS)))

    def disc(self, cx, cy, r, colour=INK):
        ox, oy = OFFSET
        self.d.ellipse([(cx + ox - r) * SS, (cy + oy - r) * SS,
                        (cx + ox + r) * SS, (cy + oy + r) * SS], fill=colour)

    def dashed_arc(self, cx, cy, r, a0, a1, step, colour=INK, width=STROKE):
        a = a0
        while a < a1:
            ox, oy = OFFSET
            self.d.arc([(cx + ox - r) * SS, (cy + oy - r) * SS,
                        (cx + ox + r) * SS, (cy + oy + r) * SS],
                       a, min(a + step * 0.55, a1),
                       fill=colour, width=int(round(width * SS)))
            a += step

    def text(self, pos, s, font, colour=INK, anchor="la"):
        ox, oy = OFFSET
        self.d.text(((pos[0] + ox) * SS, (pos[1] + oy) * SS), s, font=font,
                    fill=colour, anchor=anchor)


def bezier(p0, p1, p2, p3, n=30):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
                    u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]))
    return out


# -- the network ------------------------------------------------------------
# Four operand digits in, three hidden units, two outputs: the smallest shape
# that still reads as a network rather than a diagram of one.

LAYERS = ((4, 0.10), (3, 0.50), (2, 0.90))


def units():
    x0, x1 = P2
    y0, y1 = PANEL_Y
    inner_w, inner_h = (x1 - x0) - 120, (y1 - y0) - 130
    out = []
    for count, fx in LAYERS:
        col = []
        for k in range(count):
            fy = (k + 0.5) / count
            col.append((x0 + 60 + fx * inner_w, y0 + 65 + fy * inner_h))
        out.append(col)
    return out


UNITS = units()


def wrap(text, per_line):
    return [text[i:i + per_line] for i in range(0, len(text), per_line)]


def panel(pen, box):
    x0, x1 = box
    pen.d.rounded_rectangle([x0 * SS + OFFSET[0] * SS,
                             PANEL_Y[0] * SS + OFFSET[1] * SS,
                             x1 * SS + OFFSET[0] * SS,
                             PANEL_Y[1] * SS + OFFSET[1] * SS],
                            radius=RADIUS * SS, outline=INK,
                            width=int(round(STROKE * SS)))


def operands(pen):
    """a, b and p in full. Nothing is elided: the whole point of the task is
    that every digit counts, and a number shown with dots in it cannot be
    checked against the residue."""
    panel(pen, P1)
    x = P1[0] + 24
    y = PANEL_Y[0] + 30
    for name, value in (("a", A), ("b", B), ("p", P)):
        for ln in wrap(f"{name}={value}", 17):
            pen.text((x, y), ln, MONO, dim(y, 0.78), "lt")
            y += 30
        y += 22


def network(pen, t):
    panel(pen, P2)
    for i in range(len(UNITS) - 1):
        for j, aa in enumerate(UNITS[i]):
            for k, bb in enumerate(UNITS[i + 1]):
                pen.line([aa, bb], dim((aa[1] + bb[1]) / 2, 0.62), 2)
                phase = (t - 0.17 * i - 0.055 * j - 0.031 * k) % 1.0
                u = phase / 0.34
                if 0.0 <= u <= 1.0:
                    e = ease(u)
                    pen.disc(aa[0] + (bb[0] - aa[0]) * e,
                             aa[1] + (bb[1] - aa[1]) * e, 4,
                             dim(aa[1], 0.95))
    for i, col in enumerate(UNITS):
        for j, (cx, cy) in enumerate(col):
            wave = 0.5 + 0.5 * math.sin(2 * math.pi * (t - i * 0.16 - j * 0.04))
            pen.circle(cx, cy, 34, dim(cy, 0.74 + 0.26 * wave))


def residue(pen, t):
    """The answer, and the only thing the judge scores: exact digits."""
    panel(pen, P3)
    x = P3[0] + 24
    pen.text((x, PANEL_Y[0] + 30), FORMULA, FORMULA_FONT,
             dim(PANEL_Y[0] + 30, 0.78), "lt")
    y = PANEL_Y[0] + 116
    for ln in wrap(f"r={R}", 13):
        pen.text((x, y), ln, BIG, INK, "lt")
        y += 46


def connectors(pen, t):
    """Dashes carrying the operands into the network and the residue out."""
    for k, (gx0, gx1) in enumerate((GAP1, GAP2)):
        for row in range(3):
            yy = PANEL_Y[0] + 96 + row * 128
            px = gx0 + 6
            while px < gx1 - 6:
                pen.line([(px, yy), (min(px + 12, gx1 - 6), yy)],
                         dim(yy, 0.7), 3, None)
                px += 22
            u = ((t - 0.10 * row - 0.22 * k) % 1.0) / 0.40
            if u <= 1.0:
                pen.disc(gx0 + 6 + (gx1 - gx0 - 12) * ease(u), yy, 5,
                         dim(yy, 0.98))


# -- type -------------------------------------------------------------------

MONO = f(UIB, fit_width(UIB, "a=78169539682036545", (P1[1] - P1[0]) - 48))
BIG = f(UIB, fit_width(UIB, "r=7816953968203", (P3[1] - P3[0]) - 48))
FORMULA_FONT = f(MATH, fit_width(MATH, FORMULA, (P3[1] - P3[0]) - 60, MATH_IDX),
                 MATH_IDX)



def background():
    im = Image.new("RGB", (1, H))
    px = im.load()
    for py in range(H):
        px[0, py] = bg_at(py)
    return im.resize((W * SS, H * SS), Image.BILINEAR)


BG = background()


def logo(im):
    """SAIR's own mark and wordmark, keyed off the gradient they were drawn on
    and recomposited onto ours."""
    art = Image.open(os.path.join(ASSETS, "sair-logo.png")).convert("RGBA")
    w = LOGO[2] - LOGO[0]
    h = round(art.height * (w / art.width))
    art = art.resize((round(w * SS), h * SS), Image.LANCZOS)
    ox, oy = OFFSET
    im.paste(art, (round((LOGO[0] + ox) * SS), round((LOGO[1] + oy) * SS)), art)


def centre_offset():
    """Measure one probe frame and return the shift that equalises the margins."""
    import numpy as np

    OFFSET[0] = OFFSET[1] = 0.0
    a = np.asarray(frame(0).convert("RGB")).astype(int)
    ink = (a.min(axis=2) > 165) & ((a.max(axis=2) - a.min(axis=2)) < 50)
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    dx = ((W - 1 - cols.max()) - cols.min()) / 2.0
    dy = ((H - 1 - rows.max()) - rows.min()) / 2.0
    print(f"  centring by ({dx:+.1f}, {dy:+.1f})")
    return dx, dy


def check_layout(im):
    import numpy as np
    a = np.asarray(im.convert("RGB")).astype(int)
    ink = (a.min(axis=2) > 165) & ((a.max(axis=2) - a.min(axis=2)) < 50)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    left, right = int(cols.min()), int(W - 1 - cols.max())
    top, bottom = int(rows.min()), int(H - 1 - rows.max())
    print(f"  margins  left {left}  right {right}  top {top}  bottom {bottom}")
    used = (cols.max() - cols.min()) * (rows.max() - rows.min()) / (W * H)
    print(f"  the artwork spans {used * 100:.0f}% of the card")
    if min(left, right, top, bottom) < 24:
        raise SystemExit("ink runs too close to an edge")
    if abs(left - right) > 6:
        raise SystemExit(f"card is not symmetric: left {left}, right {right}")
    if abs(top - bottom) > 6:
        raise SystemExit(f"card is not balanced: top {top}, bottom {bottom}")
    if used < 0.74:
        raise SystemExit(f"the card is not using its space ({used*100:.0f}%)")


def quantise(frames):
    """One shared palette, matched exactly. Sharing it makes consecutive frames
    differ only where something moved; matching by hand keeps the gradient
    smooth, which Pillow's approximate matcher does not."""
    import numpy as np

    col = np.asarray(frames[0].convert("RGB")).astype(int)[:, 4, :]
    entries, seen = [], set()
    for c in map(tuple, col):
        if c not in seen:
            seen.add(c)
            entries.append(c)
    for a in (0.45, 0.55, 0.62, 0.72, 0.8, 0.85, 0.9, 0.95):
        for py in (0, H * 0.3, H * 0.55, H * 0.8, H - 1):
            c = dim(py, a)
            if c not in seen:
                seen.add(c)
                entries.append(c)
    if (255, 255, 255) not in seen:
        entries.append((255, 255, 255))
    entries = entries[:256]
    table = [v for c in entries for v in c] + [0, 0, 0] * (256 - len(entries))
    exact = {c: i for i, c in enumerate(entries)}
    arr = np.array(entries, dtype=np.int32)

    out, cache = [], {}
    for fr in frames:
        a = np.asarray(fr.convert("RGB"))
        flat = a.reshape(-1, 3).astype(np.int32)
        keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
        uniq, inverse = np.unique(keys, return_inverse=True)
        lut = np.empty(len(uniq), dtype=np.uint8)
        for j, k in enumerate(uniq.tolist()):
            c = ((k >> 16) & 255, (k >> 8) & 255, k & 255)
            i = exact.get(c)
            if i is None:
                i = cache.get(c)
                if i is None:
                    i = int(((arr - np.array(c)) ** 2).sum(axis=1).argmin())
                    cache[c] = i
            lut[j] = i
        im = Image.fromarray(lut[inverse].reshape(a.shape[:2]), mode="P")
        im.putpalette(table)
        out.append(im)
    return out


def frame(i):
    t = i / FRAMES
    im = BG.copy()
    pen = Pen(ImageDraw.Draw(im))
    logo(im)
    operands(pen)
    network(pen, t)
    residue(pen, t)
    connectors(pen, t)
    return im.resize((W, H), Image.LANCZOS)


def check_glyphs():
    """Ask the font's character map, not the rasteriser.

    A face with no glyph for a codepoint can still return a rendered box, which
    is exactly how U+2217 shipped as a hollow rectangle the first time."""
    from fontTools.ttLib import TTCollection, TTFont

    def covered(path, index):
        fonts = TTCollection(path).fonts if path.endswith(".ttc") else [TTFont(path)]
        cm = set()
        for t in fonts[index]["cmap"].tables:
            cm |= set(t.cmap.keys())
        return cm

    missing = set()
    for ch in "0123456789=abpr":
        if ord(ch) not in covered(UIB, 0):
            missing.add(("segoeuib", ch))
    for ch in FORMULA:
        if ch != " " and ord(ch) not in covered(MATH, MATH_IDX):
            missing.add(("cambria-math", ch))
    if missing:
        raise SystemExit("font is missing glyphs: " + repr(sorted(missing)))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ASSETS
    os.makedirs(out, exist_ok=True)
    check_glyphs()
    assert_no_overlap()
    OFFSET[0], OFFSET[1] = centre_offset()
    frames = [frame(i) for i in range(FRAMES)]
    check_layout(frames[0])
    path = os.path.join(out, "modular-arithmetic-sair.gif")
    q = quantise(frames)
    q[0].save(path, save_all=True, append_images=q[1:], duration=DURATION,
              loop=0, optimize=True, disposal=1)
    print(f"  {os.path.basename(path)}: {os.path.getsize(path) // 1024} KB, "
          f"{len(frames)} frames, {W}x{H}")

