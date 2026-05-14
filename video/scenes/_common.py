"""Shared palette, timing helper, and grid builder for every scene.

Each scene file imports from here. The implementation contract ("one Scene
subclass per file") still holds — this is a plain helper module, not a scene.
"""

import os

from manim import (
    VGroup, RoundedRectangle, Text, BLACK, UP, DOWN,
)

# --- palette: three colours for structure, four for the learnable params ----
BG = "#0E0E13"          # background
AMBER = "#E8A33D"       # forward pass
CYAN = "#46C8E6"        # backward pass
STRUCT = "#9AA0AE"      # neutral structure / grid lines
INK = "#E8E8EC"         # captions / text

# the four learnable scalars — lock these and reuse everywhere a cell's
# internals are drawn (log tau, w_left, w_top, bias)
C_LOGTAU = "#E0566A"
C_WL = "#5FD08A"
C_WT = "#5B9BE0"
C_B = "#E6C24A"

GRID_ROWS = 28
GRID_COLS = 10


def target_duration(default: float) -> float:
    """Scene length in seconds, supplied by build.py via TARGET_DURATION."""
    return float(os.environ.get("TARGET_DURATION", str(default)))


def hold(t: float, lo: float = 0.05) -> float:
    """Clamp a wait duration to a positive minimum (manim rejects wait(0))."""
    return max(float(t), lo)


def caption(text: str, scale: float = 0.55) -> Text:
    """A bottom-anchored caption in the house style."""
    t = Text(text, color=INK, weight="LIGHT").scale(scale)
    t.to_edge(DOWN, buff=0.45)
    return t


def make_grid(rows: int = GRID_ROWS, cols: int = GRID_COLS,
              cell: float = 0.2, gap: float = 0.03,
              stroke=STRUCT, stroke_width: float = 0.8):
    """Build a rows x cols grid of rounded cells.

    Returns ``(group, cells)`` where ``cells[r][c]`` is the RoundedRectangle at
    row ``r`` (top = 0), column ``c`` (left = 0). The group is centred on the
    origin.
    """
    cells = []
    group = VGroup()
    step = cell + gap
    for r in range(rows):
        row = []
        for c in range(cols):
            sq = RoundedRectangle(
                width=cell, height=cell, corner_radius=cell * 0.22,
                stroke_color=stroke, stroke_width=stroke_width,
                fill_color=BLACK, fill_opacity=1.0,
            )
            sq.move_to([c * step, -r * step, 0])
            row.append(sq)
            group.add(sq)
        cells.append(row)
    group.move_to([0, 0, 0])
    return group, cells
