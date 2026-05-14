"""Act 1 — the cell, and how to tile it. Three sub-beats: cell, tile, signal."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from manim import (
    Scene, VGroup, RoundedRectangle, Text, Arrow, FadeIn, FadeOut, Indicate,
    BLACK, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN, interpolate_color, ManimColor,
)
from _common import (
    BG, AMBER, CYAN, STRUCT, INK, C_LOGTAU, C_WL, C_WT, C_B,
    caption, target_duration, hold,
)

MONO = "Consolas"

_CYAN = ManimColor(CYAN)
_AMBER = ManimColor(AMBER)
_GREY = ManimColor("#3A3A44")


def divergent(t: float):
    """t in [0,1] -> blue .. grey .. amber."""
    if t < 0.5:
        return interpolate_color(_CYAN, _GREY, t * 2)
    return interpolate_color(_GREY, _AMBER, (t - 0.5) * 2)


class Act1CellAndGrid(Scene):
    def construct(self):
        self.camera.background_color = BG
        total = target_duration(135.0)
        b1, b2, b3 = total * 0.46, total * 0.18, total * 0.36

        # ===== Beat 2a: the cell ============================================
        cell = RoundedRectangle(
            width=2.2, height=2.2, corner_radius=0.25,
            stroke_color=STRUCT, stroke_width=2.5,
            fill_color=BLACK, fill_opacity=1.0,
        ).shift(LEFT * 3.4)
        y_label = Text("y", font=MONO, color=INK).scale(0.7).move_to(cell)
        in_L = Arrow(cell.get_left() + LEFT * 0.9, cell.get_left(),
                     color=STRUCT, buff=0.05, stroke_width=3)
        in_T = Arrow(cell.get_top() + UP * 0.9, cell.get_top(),
                     color=STRUCT, buff=0.05, stroke_width=3)
        lab_L = Text("y_left", font=MONO, color=STRUCT).scale(0.4).next_to(in_L, LEFT, buff=0.1)
        lab_T = Text("y_top", font=MONO, color=STRUCT).scale(0.4).next_to(in_T, UP, buff=0.1)

        self.play(FadeIn(cell), FadeIn(y_label), run_time=0.8)
        self.play(FadeIn(in_L), FadeIn(lab_L), FadeIn(in_T), FadeIn(lab_T), run_time=0.8)

        eq = VGroup(
            Text("drive   = w_L · y_left + w_T · y_top", font=MONO,
                 t2c={"w_L": C_WL, "w_T": C_WT}).scale(0.5),
            Text("tau · ds/dt = -s + drive", font=MONO,
                 t2c={"tau": C_LOGTAU}).scale(0.5),
            Text("y       = tanh(s + b)", font=MONO,
                 t2c={"+ b": C_B}).scale(0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(RIGHT * 2.6)

        self.play(FadeIn(eq[0]), run_time=b1 * 0.18)
        self.play(Indicate(eq[0], color=C_WL, scale_factor=1.06), run_time=b1 * 0.12)
        self.play(FadeIn(eq[1]), run_time=b1 * 0.16)
        self.play(Indicate(eq[1], color=C_LOGTAU, scale_factor=1.06), run_time=b1 * 0.12)
        self.play(FadeIn(eq[2]), run_time=b1 * 0.16)
        self.play(Indicate(eq[2], color=C_B, scale_factor=1.06), run_time=b1 * 0.12)

        legend = VGroup(
            Text("log tau", font=MONO, color=C_LOGTAU).scale(0.42),
            Text("w_L", font=MONO, color=C_WL).scale(0.42),
            Text("w_T", font=MONO, color=C_WT).scale(0.42),
            Text("b", font=MONO, color=C_B).scale(0.42),
        ).arrange(RIGHT, buff=0.6).next_to(eq, DOWN, buff=0.7)
        four = Text("four learnable numbers per cell", font=MONO, color=STRUCT).scale(0.4)
        four.next_to(legend, DOWN, buff=0.35)
        self.play(FadeIn(legend), FadeIn(four), run_time=b1 * 0.14)
        self.wait(hold(b1 * 0.18))

        cell_group = VGroup(cell, y_label, in_L, in_T, lab_L, lab_T)
        self.play(
            FadeOut(eq), FadeOut(legend), FadeOut(four),
            FadeOut(in_L), FadeOut(in_T), FadeOut(lab_L), FadeOut(lab_T),
            run_time=0.6,
        )

        # ===== Beat 2b: the tile ============================================
        rows, cols = 28, 10
        csize = 0.20
        gap = 0.03
        st = csize + gap
        rng = np.random.default_rng(3)
        wl = rng.standard_normal((rows, cols))
        wl = (wl - wl.min()) / (wl.max() - wl.min())

        grid = VGroup()
        cells = []
        for r in range(rows):
            row = []
            for c in range(cols):
                sq = RoundedRectangle(
                    width=csize, height=csize, corner_radius=csize * 0.22,
                    stroke_color=STRUCT, stroke_width=0.6,
                    fill_color=divergent(float(wl[r, c])), fill_opacity=1.0,
                )
                sq.move_to([(c - cols / 2) * st, (rows / 2 - r) * st, 0])
                row.append(sq)
                grid.add(sq)
            cells.append(row)
        grid.move_to(ORIGIN)

        self.play(
            FadeOut(VGroup(cell, y_label)),
            FadeIn(grid, lag_ratio=0.002, run_time=b2 * 0.55),
        )
        dims = caption("28 cells tall, 10 wide — every cell identical in structure")
        self.play(FadeIn(dims), run_time=b2 * 0.2)
        self.wait(hold(b2 * 0.25))

        # ===== Beat 2c: signal flow =========================================
        self.play(FadeOut(dims), run_time=0.4)

        # external input markers on the left edge
        in_markers = VGroup(*[
            Arrow(cells[r][0].get_left() + LEFT * 0.35, cells[r][0].get_left(),
                  color=AMBER, buff=0.02, stroke_width=2)
            for r in range(rows)
        ])
        self.play(FadeIn(in_markers), run_time=0.5)

        flow_cap = caption("forward pass — activity sweeps in raster order")
        self.play(FadeIn(flow_cap), run_time=0.4)

        # wavefront: brightness pulse cell by cell in raster order
        order = [(r, c) for r in range(rows) for c in range(cols)]
        sweep_time = max(b3 - 2.2, 1.0)
        chunk = 14  # cells lit per play call, to keep the call count sane
        per = sweep_time / (len(order) / chunk)
        for i in range(0, len(order), chunk):
            anims = []
            for (r, c) in order[i:i + chunk]:
                anims.append(cells[r][c].animate.set_stroke(AMBER, width=2.0))
            self.play(*anims, run_time=per)

        # bottom row outputs
        out_row = VGroup(*[cells[rows - 1][c] for c in range(cols)])
        out_markers = VGroup(*[
            Arrow(cells[rows - 1][c].get_bottom(),
                  cells[rows - 1][c].get_bottom() + DOWN * 0.35,
                  color=AMBER, buff=0.02, stroke_width=2)
            for c in range(cols)
        ])
        self.play(
            out_row.animate.set_fill(AMBER, opacity=0.9),
            FadeIn(out_markers),
            run_time=0.6,
        )
        count_cap = caption("28 x 10 = 280 cells.  1,120 parameters total.  That's the entire model.")
        self.play(FadeOut(flow_cap), FadeIn(count_cap), run_time=0.5)
        self.wait(hold(b3 - sweep_time - 1.5))
        self.wait(0.3)
