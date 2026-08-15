"""Build the figures used by the READMEs.

Both figures explain mechanism rather than reporting a result, because the
official evaluation runs on a secret seed and this repository has no score to
plot. Nothing here is copied from the competition site.

    python .github/assets/build_assets.py

Standard library only, so it runs anywhere with no setup.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent

# House palette, shared by all three SAIR repositories so they read as one set.
PAPER = "#FAFBFC"
INK = "#12141A"
SOFT = "#4A5663"
FAINT = "#6B7684"
RULE = "#D8DEE6"
ACCENT = "#3949AB"      # this repository: digits and place value
ACCEPT = "#2EA043"
REJECT = "#CF222E"

FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"
MONO = "SFMono-Regular, Consolas, Liberation Mono, monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, fill=INK, weight=400, anchor="start", font=FONT,
         spacing=None, opacity=None):
    extra = ""
    if spacing:
        extra += f' letter-spacing="{spacing}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"'
            f'{extra}>{esc(s)}</text>')


def hero():
    """One product, reduced one digit at a time, on a loop."""
    W, H = 1200, 440
    a, b, p = 4821, 7396, 1009

    # Derived, never written by hand, so a frame cannot disagree with the
    # arithmetic it illustrates. This is the reduction the scratchpad performs:
    # walk the product left to right, keeping only the running remainder.
    product = a * b
    digits = str(product)
    frames = [(f"{a} × {b}", "the operands, as decimal strings"),
              (f"= {digits}", f"the product, {len(digits)} digits wide")]
    r = 0
    for i, d in enumerate(digits):
        prev, r = r, (r * 10 + int(d)) % p
        if i in (3, 4, len(digits) - 2):
            frames.append((f"{prev}{d} mod {p} → {r}",
                           "take the next digit, reduce, keep going"))
    frames.append((f"ANS = {r}", "one answer, exactly right or wrong"))
    trace = frames
    a, b, p = str(a), str(b), str(p)
    p_ = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
          f'width="{W}" height="{H}" role="img" aria-label="A neural network '
          f'computing a times b mod p one digit at a time, showing the running '
          f'remainder staying the width of the prime while the operands grow">']
    p_.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')
    p_.append(f'<rect x="0" y="0" width="10" height="{H}" fill="{ACCENT}"/>')

    p_.append(text(56, 62, "EXACT ARITHMETIC, LEARNED", 12, ACCENT, 600,
                   spacing="0.34em"))
    p_.append(text(56, 104, "(a × b) mod p, with no arithmetic in the code",
                   30, INK, 600))
    p_.append(text(56, 136,
                   "Operands run to hundreds of digits. Every answer is exact "
                   "or it is wrong.", 16, SOFT))

    # The operands, held throughout.
    p_.append(f'<rect x="56" y="186" width="300" height="176" rx="10" '
              f'fill="#FFFFFF" stroke="{RULE}"/>')
    p_.append(text(80, 218, "GIVEN", 11, FAINT, 600, spacing="0.28em"))
    for i, (name, val) in enumerate((("a", a), ("b", b), ("p", p))):
        y = 254 + i * 34
        p_.append(text(80, y, name, 14, FAINT, font=MONO))
        p_.append(text(112, y, val, 20, INK, 500, font=MONO))
    p_.append(text(196, 254, "hundreds of digits", 11, FAINT))
    p_.append(text(196, 288, "hundreds of digits", 11, FAINT))
    p_.append(text(196, 322, "prime, and shorter", 11, FAINT))

    # The trace, one frame at a time.
    dur = len(trace) * 1.5
    p_.append(f'<rect x="392" y="186" width="752" height="176" rx="10" '
              f'fill="#FFFFFF" stroke="{RULE}"/>')
    p_.append(text(418, 218, "SCRATCHPAD", 11, ACCENT, 600, spacing="0.28em"))
    for i, (line, note) in enumerate(trace):
        last = i == len(trace) - 1
        colour = ACCEPT if last else INK
        p_.append(
            f'<g opacity="0"><animate attributeName="opacity" '
            f'values="0;1;1;0" keyTimes="0;0.02;0.15;0.17" dur="{dur}s" '
            f'begin="{i * 1.5}s" repeatCount="indefinite"/>'
            f'{text(418, 268, line, 26, colour, 500, font=MONO)}'
            f'{text(418, 300, note, 14, SOFT)}</g>')

    # Progress ticks, so the loop reads as a sequence.
    for i in range(len(trace)):
        x = 418 + i * 26
        p_.append(f'<rect x="{x}" y="330" width="18" height="4" rx="2" '
                  f'fill="{ACCENT}" opacity="0.2"/>')
        p_.append(f'<rect x="{x}" y="330" width="18" height="4" rx="2" '
                  f'fill="{ACCENT}" opacity="0"><animate '
                  f'attributeName="opacity" values="0;1;1;0" '
                  f'keyTimes="0;0.02;0.15;0.17" dur="{dur}s" '
                  f'begin="{i * 1.5}s" repeatCount="indefinite"/></rect>')

    p_.append(text(700, 334,
                   "the remainder never grows past the width of p", 13, FAINT))

    p_.append(f'<path d="M 56 {H - 34} H {W - 56}" stroke="{RULE}"/>')
    p_.append(text(56, H - 12,
                   "The answer has to come from trained parameters. Native "
                   "big-integer arithmetic on the inputs is disqualifying.",
                   13, FAINT))
    p_.append("</svg>")
    (OUT / "hero.svg").write_text("\n".join(p_), encoding="utf-8")


def place_value():
    """Why absolute position fails and place value does not."""
    W, H = 1100, 430
    p_ = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
          f'width="{W}" height="{H}" role="img" aria-label="The same digit '
          f'receives a different absolute position when the operand is padded, '
          f'but the same place value, which is why place-value embeddings '
          f'generalise to longer inputs">']
    p_.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')
    p_.append(text(72, 52, "LENGTH GENERALISATION", 12, ACCENT, 600,
                   spacing="0.34em"))
    p_.append(text(72, 80, "The same digit, told two different things",
                   20, INK, 600))

    short, long = "4821", "00000004821"
    cell = 34

    def strip(y, digits, label, mode):
        p_.append(text(72, y - 14, label, 13, SOFT, 600))
        for i, d in enumerate(digits):
            x = 72 + i * cell
            live = i >= len(digits) - 4
            fill = ACCENT if live else FAINT
            op = 0.12 if live else 0.05
            p_.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" '
                      f'height="{cell - 4}" rx="4" fill="{fill}" '
                      f'opacity="{op}"/>')
            p_.append(text(x + (cell - 4) / 2, y + 21, d, 16,
                           INK if live else FAINT, 500, "middle", MONO))
            if mode == "position":
                tag = str(i)
                colour = ACCENT if live else FAINT
            else:
                tag = str(len(digits) - 1 - i)
                colour = ACCENT if live else FAINT
            p_.append(text(x + (cell - 4) / 2, y + cell + 14, tag, 11,
                           colour, 500, "middle", MONO))

    p_.append(text(72, 128, "Absolute position: counts from the left", 14,
                   INK, 600))
    strip(160, short, "operand as written", "position")
    strip(240, long, "the same operand, padded", "position")
    p_.append(text(72 + len(long) * cell + 20, 178, "the 4 is at index 0",
                   13, REJECT))
    p_.append(text(72 + len(long) * cell + 20, 258, "now the 4 is at index 7",
                   13, REJECT))

    p_.append(text(72, 322, "Place value: counts from the right", 14, INK, 600))
    strip(354, long, "the same padded operand", "significance")
    p_.append(text(72 + len(long) * cell + 20, 372,
                   "the 4 keeps significance 3", 13, ACCEPT))

    p_.append(text(72, H - 10,
                   "A model trained on short operands has never seen index 7. "
                   "It has seen significance 3 on every example.", 13, FAINT))
    p_.append("</svg>")
    (OUT / "place-value.svg").write_text("\n".join(p_), encoding="utf-8")


if __name__ == "__main__":
    hero()
    place_value()
    for f in sorted(OUT.glob("*.svg")):
        print(f"  {f.name}  {f.stat().st_size:,} bytes")
