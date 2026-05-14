"""Act 4 — why this matters. Chip mapping, cortex aside, what's next."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from manim import (
    Scene, VGroup, RoundedRectangle, Rectangle, Text, Line, Dot, Arrow,
    FadeIn, FadeOut, Create, Indicate, BLACK, WHITE, UP, DOWN, LEFT, RIGHT,
    ORIGIN,
)
from _common import (
    BG, AMBER, CYAN, STRUCT, INK, C_LOGTAU, C_WL, C_WT, C_B,
    make_grid, caption, target_duration, hold,
)

MONO = "Consolas"


class Act4Outro(Scene):
    def construct(self):
        self.camera.background_color = BG
        total = target_duration(135.0)
        b1, b2, b3 = total * 0.42, total * 0.22, total * 0.36

        # ===== Scene 5a: the chip mapping ===================================
        cell = RoundedRectangle(width=2.0, height=2.0, corner_radius=0.22,
                                stroke_color=STRUCT, stroke_width=2.5,
                                fill_color=BLACK, fill_opacity=1.0)
        cell.shift(LEFT * 4.0)
        cell_lbl = Text("one cell", font=MONO, color=STRUCT).scale(0.4)
        cell_lbl.next_to(cell, DOWN, buff=0.3)
        self.play(FadeIn(cell), FadeIn(cell_lbl), run_time=0.7)

        # analog schematic on the right
        def cap(label, color):
            plates = VGroup(
                Line(LEFT * 0.25, RIGHT * 0.25, color=color, stroke_width=4),
                Line(LEFT * 0.25 + DOWN * 0.14, RIGHT * 0.25 + DOWN * 0.14,
                     color=color, stroke_width=4),
            )
            t = Text(label, font=MONO, color=color).scale(0.3)
            t.next_to(plates, DOWN, buff=0.12)
            return VGroup(plates, t)

        integ = cap("integrator C", CYAN)
        tanh_box = VGroup(
            Rectangle(width=0.9, height=0.7, stroke_color=AMBER,
                      stroke_width=2.5, fill_color=BLACK, fill_opacity=1.0),
            Text("tanh", font=MONO, color=AMBER).scale(0.3),
        )
        param_caps = VGroup(
            cap("log tau", C_LOGTAU), cap("w_L", C_WL),
            cap("w_T", C_WT), cap("b", C_B),
        ).arrange(RIGHT, buff=0.45)

        schematic = VGroup(integ, tanh_box).arrange(RIGHT, buff=0.9)
        schematic.shift(RIGHT * 1.6 + UP * 0.8)
        param_caps.next_to(schematic, DOWN, buff=0.9)
        readout = Arrow(tanh_box.get_right(), tanh_box.get_right() + RIGHT * 1.2,
                        color="#E0566A", buff=0.05, stroke_width=4)
        readout_lbl = Text("backward-message readout\n(must be low-noise)",
                           font=MONO, color="#E0566A").scale(0.3)
        readout_lbl.next_to(readout, RIGHT, buff=0.2)

        morph = Arrow(cell.get_right(), schematic.get_left() + LEFT * 0.3,
                      color=STRUCT, buff=0.2, stroke_width=3)
        self.play(Create(morph), run_time=0.5)
        self.play(FadeIn(schematic), run_time=b1 * 0.2)
        self.play(FadeIn(param_caps), run_time=b1 * 0.2)
        self.play(Create(readout), FadeIn(readout_lbl), run_time=b1 * 0.16)
        chip_cap = caption("every operation maps onto a standard analog crossbar primitive")
        self.play(FadeIn(chip_cap), run_time=0.4)
        self.play(Indicate(readout, color="#E0566A", scale_factor=1.1),
                  Indicate(readout_lbl, color="#E0566A"), run_time=b1 * 0.18)
        self.wait(hold(b1 * 0.2))

        self.play(
            FadeOut(VGroup(cell, cell_lbl, schematic, param_caps, readout,
                           readout_lbl, morph, chip_cap)),
            run_time=0.6,
        )

        # ===== Scene 5b: cortex aside =======================================
        sheet = VGroup()
        dots = {}
        nx, ny = 16, 8
        for r in range(ny):
            for c in range(nx):
                d = Dot([(c - nx / 2) * 0.55, (ny / 2 - r) * 0.55, 0],
                        radius=0.045, color=STRUCT)
                dots[(r, c)] = d
                sheet.add(d)
        edges = VGroup()
        for (r, c), d in dots.items():
            for dr, dc in ((0, 1), (1, 0)):
                if (r + dr, c + dc) in dots:
                    edges.add(Line(d.get_center(),
                                   dots[(r + dr, c + dc)].get_center(),
                                   color=STRUCT, stroke_width=0.8,
                                   stroke_opacity=0.4))
        sheet_all = VGroup(edges, sheet).move_to(ORIGIN)
        self.play(FadeIn(sheet_all, lag_ratio=0.005), run_time=b2 * 0.4)
        cortex_cap = caption("cortex doesn't backpropagate either")
        self.play(FadeIn(cortex_cap), run_time=0.4)
        self.wait(hold(b2 * 0.5))
        self.play(FadeOut(sheet_all), FadeOut(cortex_cap), run_time=0.6)

        # ===== Scene 5c: what's next ========================================
        grid, cells = make_grid(28, 10, cell=0.16, gap=0.025)
        grid.scale(0.9).shift(DOWN * 0.6)
        self.play(FadeIn(grid, lag_ratio=0.002), run_time=b3 * 0.18)

        # faint tree above the grid plane
        root = np.array([0, 2.6, 0])
        tree = VGroup()
        l1 = [np.array([x, 1.7, 0]) for x in (-2.0, 0.0, 2.0)]
        for p in l1:
            tree.add(Line(root, p, color=CYAN, stroke_width=2,
                          stroke_opacity=0.6))
        top = grid.get_top()
        for p in l1:
            for dx in (-0.9, 0.0, 0.9):
                leaf = np.array([p[0] + dx, top[1] + 0.1, 0])
                tree.add(Line(p, leaf, color=CYAN, stroke_width=1.5,
                              stroke_opacity=0.45))
        tree.add(Dot(root, radius=0.07, color=CYAN))
        self.play(Create(tree), run_time=b3 * 0.25)
        next_cap = caption("hierarchical projection — next")
        self.play(FadeIn(next_cap), run_time=0.4)
        self.wait(hold(b3 * 0.2))

        # end card
        self.play(
            FadeOut(grid), FadeOut(tree), FadeOut(next_cap), run_time=0.6,
        )
        title = Text("A Cellular Automaton That Learns to Read",
                     font=MONO, color=INK).scale(0.6)
        sub = Text("a continuous CA learns row-wise MNIST — and we know "
                   "what it takes to put it on a chip",
                   font=MONO, color=STRUCT).scale(0.34)
        repo = Text("code & paper:  grid-neurons repository", font=MONO,
                    color=AMBER).scale(0.36)
        card = VGroup(title, sub, repo).arrange(DOWN, buff=0.45).move_to(ORIGIN)
        self.play(FadeIn(card), run_time=0.8)
        self.wait(hold(b3 * 0.3))
        self.wait(0.4)
