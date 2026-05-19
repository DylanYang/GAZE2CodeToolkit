"""Build a token-level OCR ground-truth CSV from a text source file.

Renders the input text with a chosen font / size, computes pixel
bounding boxes for each whitespace-separated token, and writes a CSV
in the schema consumed by ``cli/evaluate_ocr.py``::

    line_num, x, y, width, height, text

where ``line_num`` is ``"Line N Part M"`` (N = visual line, M = token
index within that line).

A single stimulus typically combines several text regions in
different fonts (HTML header in sans-serif, problem statement in
serif, code block in monospace, answer choices in sans-serif). Run
the script once per region with ``--append`` and per-region
``--font`` / ``--x-offset`` / ``--y-offset`` / ``--line-offset``::

    # 1) Question header (bold sans, top of page)
    python -m cli.build_groundtruth \\
        --text regions/Q5_header.txt --out output/ocr_groundtruth/Q5_gt.csv \\
        --font DejaVuSans-Bold.ttf --font-size 16 \\
        --x-offset 118 --y-offset 62

    # 2) Problem description (serif, below header)
    python -m cli.build_groundtruth \\
        --text regions/Q5_problem.txt --out output/ocr_groundtruth/Q5_gt.csv \\
        --append --font DejaVuSerif.ttf --font-size 16 \\
        --x-offset 119 --y-offset 108 --line-offset 1

    # 3) Code block (monospace)
    python -m cli.build_groundtruth \\
        --text regions/Q5_code.java --out output/ocr_groundtruth/Q5_gt.csv \\
        --append --font DejaVuSansMono.ttf --font-size 14 \\
        --x-offset 140 --y-offset 220 --line-offset 3

    # 4) Answer choices (sans, near bottom)
    python -m cli.build_groundtruth \\
        --text regions/Q5_answers.txt --out output/ocr_groundtruth/Q5_gt.csv \\
        --append --font DejaVuSans.ttf --font-size 14 \\
        --x-offset 239 --y-offset 838 --line-offset 16

To visually verify alignment, pass ``--debug-overlay <stimulus.png>``
and the script writes ``<out>.debug.png`` showing every predicted
box drawn on top of the stimulus.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

# Cross-platform fallback chain for the default font. Listed in the
# order most likely to be installed on Linux / macOS / Windows.
_DEFAULT_FONT_CANDIDATES = (
    "DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
)

_CSV_FIELDS = ["line_num", "x", "y", "width", "height", "text"]


def _resolve_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """Load `font_path` if given; otherwise walk a default fallback chain."""
    if font_path:
        return ImageFont.truetype(font_path, size)
    for candidate in _DEFAULT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise RuntimeError(
        "No monospace font found. Pass --font /path/to/font.ttf explicitly."
    )


def _tokenize_with_columns(line: str) -> list[tuple[int, str]]:
    """Split `line` on whitespace, returning (column_index, token) pairs.

    Column index is the character offset of the token's first
    non-whitespace character — needed to compute the token's pixel x
    via a prefix textbbox measurement.
    """
    tokens: list[tuple[int, str]] = []
    cur: list[str] = []
    cur_start: int | None = None
    for col, ch in enumerate(line):
        if ch.isspace():
            if cur:
                tokens.append((cur_start, "".join(cur)))  # type: ignore[arg-type]
                cur = []
                cur_start = None
        else:
            if cur_start is None:
                cur_start = col
            cur.append(ch)
    if cur:
        tokens.append((cur_start, "".join(cur)))  # type: ignore[arg-type]
    return tokens


def build_groundtruth_rows(
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    x_offset: int,
    y_offset: int,
    line_height: int | None,
    line_offset: int,
    align: str = "left",
) -> list[dict]:
    """Compute token bboxes for one text block and return CSV rows.

    The function never draws onto a real canvas: it instantiates a
    1x1 dummy image only so PIL exposes `ImageDraw.textbbox`, which
    is what actually does the metric measurement.

    `align`:
        "left"   — `x_offset` is the left edge of every line (default).
        "right"  — `x_offset` is the right edge; each line's start_x
                   is computed as `x_offset - line_width`. Use for
                   right-justified output panels (e.g. Q4A/Q4B
                   numeric pyramid).
        "center" — `x_offset` is the centre x; line is centred on it.
    """
    if align not in ("left", "right", "center"):
        raise ValueError(f"Unknown align mode: {align}")

    dummy = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(dummy)

    if line_height is None:
        # Reasonable default that matches most monospace renderings.
        # Override with --line-height for tighter / looser layouts.
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + 2

    rows: list[dict] = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        y_top = y_offset + (line_idx - 1) * line_height

        # Per-line `line_x_start` (the x of the line's first character).
        # For left-align this is just `x_offset`. For right/center we
        # need the full line's ink width, then shift accordingly.
        if align == "left" or not line.strip():
            line_x_start = x_offset
        else:
            l0, _, r0, _ = draw.textbbox((0, 0), line, font=font)
            line_width = r0 - l0
            if align == "right":
                line_x_start = x_offset - line_width
            else:  # center
                line_x_start = x_offset - line_width // 2

        for part_idx, (col, tok) in enumerate(
            _tokenize_with_columns(line), start=1
        ):
            # Prefix-measure to get the exact pixel-x of this token,
            # honouring variable-width glyphs in proportional fonts.
            prefix = line[:col]
            if prefix:
                _, _, prefix_right, _ = draw.textbbox(
                    (line_x_start, y_top), prefix, font=font
                )
                tok_x = prefix_right
            else:
                tok_x = line_x_start

            l, t, r, b = draw.textbbox((tok_x, y_top), tok, font=font)
            rows.append({
                "line_num": f"Line {line_offset + line_idx} Part {part_idx}",
                "x":        int(l),
                "y":        int(t),
                "width":    int(r - l),
                "height":   int(b - t),
                "text":     tok,
            })
    return rows


def _write_rows(out_csv: Path, rows: Iterable[dict], *, append: bool) -> int:
    """Write `rows` to `out_csv`. Returns the number of rows written.

    In append mode the header is reused, and `line_num` collisions are
    *not* deduped — the caller is responsible for choosing
    `--line-offset` so visual line numbers don't overlap.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    write_header = (not append) or (not out_csv.exists())
    mode = "a" if append and out_csv.exists() else "w"
    with out_csv.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _save_debug_overlay(
    stimulus_path: Path,
    out_csv: Path,
    rows: Iterable[dict],
    debug_out: Path,
    *,
    only_new: bool = False,
    new_rows: Iterable[dict] | None = None,
) -> None:
    """Draw every row's bbox on top of `stimulus_path` for visual QA.

    `only_new` colours rows just added by *this* run differently from
    rows already in `out_csv` from previous --append calls.
    """
    img = Image.open(stimulus_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)

    new_keys = set()
    if only_new and new_rows is not None:
        new_keys = {(r["line_num"], r["text"]) for r in new_rows}

    for r in rows:
        key = (r["line_num"], r["text"])
        is_new = key in new_keys
        colour = (0, 200, 0, 220) if is_new else (255, 90, 0, 180)
        l, t = int(r["x"]), int(r["y"])
        rgt, btm = l + int(r["width"]), t + int(r["height"])
        drw.rectangle([l, t, rgt, btm], outline=colour, width=2)

    debug_out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, overlay).save(debug_out)


def _read_existing_rows(out_csv: Path) -> list[dict]:
    if not out_csv.exists():
        return []
    with out_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text", required=True,
        help="Path to a plain-text source file (one region per run).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output ground-truth CSV path.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to --out instead of overwriting (compose multiple regions).",
    )
    parser.add_argument(
        "--font", default=None,
        help="Font file path (.ttf/.ttc/.otf). "
             "Default: first available monospace from a fallback chain.",
    )
    parser.add_argument(
        "--font-size", type=int, default=14,
        help="Font size in pixels (default: 14).",
    )
    parser.add_argument(
        "--x-offset", type=int, default=0,
        help="Pixel x anchor for each line. Meaning depends on "
             "--align: with `left` (default) it is the left edge; with "
             "`right` it is the right edge; with `center` it is the "
             "centre x.",
    )
    parser.add_argument(
        "--align", choices=["left", "right", "center"], default="left",
        help="Per-line horizontal alignment. Use `right` for "
             "right-justified content (e.g. Q4A/Q4B numeric pyramid).",
    )
    parser.add_argument(
        "--y-offset", type=int, default=0,
        help="Pixel y of the first line's top (top margin).",
    )
    parser.add_argument(
        "--line-height", type=int, default=None,
        help="Pixels between line tops "
             "(default: font ascent + descent + 2).",
    )
    parser.add_argument(
        "--line-offset", type=int, default=0,
        help="Add this to every line index in line_num "
             "(use when --append'ing after earlier regions). "
             "Example: previous region ended at Line 5 → pass --line-offset 5 "
             "so this region starts at Line 6.",
    )
    parser.add_argument(
        "--debug-overlay", default=None,
        help="If set, also write a PNG that draws every bbox over the "
             "given stimulus image. New rows are green; pre-existing "
             "rows from previous --append runs are orange.",
    )
    parser.add_argument(
        "--debug-out", default=None,
        help="Explicit path for the debug overlay PNG. "
             "Default: <out>.debug.png.",
    )

    args = parser.parse_args(argv)

    text_path = Path(args.text)
    if not text_path.exists():
        print(f"Source text not found: {text_path}", file=sys.stderr)
        return 1
    out_csv = Path(args.out)

    try:
        font = _resolve_font(args.font, args.font_size)
    except (OSError, RuntimeError) as exc:
        print(f"Could not load font: {exc}", file=sys.stderr)
        return 2

    text = text_path.read_text(encoding="utf-8")
    new_rows = build_groundtruth_rows(
        text,
        font=font,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        line_height=args.line_height,
        line_offset=args.line_offset,
        align=args.align,
    )

    n_written = _write_rows(out_csv, new_rows, append=args.append)
    print(f"Wrote {n_written} token row(s) → {out_csv}"
          + (" (appended)" if args.append and out_csv.exists() else ""))

    if args.debug_overlay:
        stim = Path(args.debug_overlay)
        if not stim.exists():
            print(f"Stimulus for debug overlay not found: {stim}",
                  file=sys.stderr)
            return 3
        debug_out = (Path(args.debug_out)
                     if args.debug_out
                     else out_csv.with_suffix(out_csv.suffix + ".debug.png"))
        # Read back the full (possibly appended) CSV so the overlay
        # shows historical rows too — new rows highlighted green.
        all_rows = _read_existing_rows(out_csv)
        _save_debug_overlay(
            stim, out_csv, all_rows, debug_out,
            only_new=True, new_rows=new_rows,
        )
        print(f"Debug overlay → {debug_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
