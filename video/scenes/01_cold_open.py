"""Cold open — 'The wrong question'. Three beats of roughly equal weight."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from manim import (
    Scene, VGroup, RoundedRectangle, Rectangle, Arrow, Text, FadeIn, FadeOut,
    BLACK, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)
from _common import (
    BG, AMBER, CYAN, STRUCT, INK, caption, target_duration, hold,
)

LIFE_GREEN = "#5C9C6E"


class ColdOpen(Scene):
    def construct(self):
        self.camera.background_color = BG
        total = target_duration(45.0)
        beat = total / 3.0

        # ----- Beat 1: Conway's Life on a grid --------------------------------
        n = 24
        cell = 0.22
        step = cell + 0.025
        squares = []
        grp = VGroup()
        for r in range(n):
            row = []
            for c in range(n):
                sq = RoundedRectangle(
                    width=cell, height=cell, corner_radius=cell * 0.2,
                    stroke_color=STRUCT, stroke_width=0.6,
                    fill_color=BLACK, fill_opacity=1.0,
                )
                sq.move_to([(c - n / 2) * step, (n / 2 - r) * step, 0])
                row.append(sq)
                grp.add(sq)
            squares.append(row)
        grp.move_to(ORIGIN)

        rng = np.random.default_rng(7)
        state = (rng.random((n, n)) < 0.32).astype(np.int8)

        def render(st):
            return [
                squares[r][c].animate.set_fill(
                    LIFE_GREEN if st[r, c] else BLACK,
                    opacity=1.0 if st[r, c] else 1.0,
                )
                for r in range(n) for c in range(n)
            ]

        def life_step(st):
            nb = sum(
                np.roll(np.roll(st, dr, 0), dc, 1)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            )
            return ((nb == 3) | ((st == 1) & (nb == 2))).astype(np.int8)

        cap1 = caption("Cellular Automaton")
        self.play(FadeIn(grp, run_time=0.6))
        for r in range(n):
            for c in range(n):
                squares[r][c].set_fill(LIFE_GREEN if state[r, c] else BLACK, opacity=1.0)
        self.add(grp)
        self.play(FadeIn(cap1, run_time=0.5))

        gens = 6
        per = max((beat - 1.5) / gens, 0.25)
        for _ in range(gens):
            state = life_step(state)
            self.play(*render(state), run_time=per)
        self.wait(hold(beat - 1.5 - gens * per))

        # ----- Beat 2: NCA — a neural net per cell ----------------------------
        self.play(FadeOut(cap1, run_time=0.4))
        # collapse the Life grid into a smaller, tidy NCA grid
        m = 8
        small = VGroup()
        ncells = []
        cs = 0.5
        cstep = cs + 0.12
        for r in range(m):
            row = []
            for c in range(m):
                sq = RoundedRectangle(
                    width=cs, height=cs, corner_radius=cs * 0.2,
                    stroke_color=STRUCT, stroke_width=1.2,
                    fill_color=BLACK, fill_opacity=1.0,
                )
                sq.move_to([(c - m / 2) * cstep, (m / 2 - r) * cstep, 0])
                row.append(sq)
                small.add(sq)
            ncells.append(row)
        small.move_to(ORIGIN).shift(LEFT * 1.2)

        self.play(FadeOut(grp, run_time=0.5), FadeIn(small, run_time=0.6))

        def mlp_icon():
            bars = VGroup(*[
                Rectangle(width=0.34, height=0.06, fill_color=CYAN,
                          fill_opacity=1.0, stroke_width=0)
                for _ in range(3)
            ]).arrange(DOWN, buff=0.05)
            return bars

        icons = VGroup()
        arrows = VGroup()
        for r in range(m):
            for c in range(m):
                ic = mlp_icon().scale(0.5)
                ic.next_to(ncells[r][c], UP, buff=0.02)
                icons.add(ic)
        cap2 = caption("Neural Cellular Automaton — a neural network per cell")
        self.play(FadeIn(icons, run_time=0.7), FadeIn(cap2, run_time=0.4))
        self.wait(hold(beat * 0.35, 0.3))

        # highlight one cell, pull its MLP out, backprop arrows to a GPU box
        hr, hc = 3, 3
        hi_cell = ncells[hr][hc]
        big_mlp = mlp_icon().scale(1.4).move_to([3.4, 1.2, 0])
        gpu = VGroup(
            Rectangle(width=1.5, height=1.0, stroke_color=AMBER,
                      stroke_width=2.0, fill_color=BLACK, fill_opacity=1.0),
            Text("GPU", color=AMBER, weight="BOLD").scale(0.45),
        )
        gpu.move_to([3.4, -1.4, 0])
        bp = Arrow(big_mlp.get_bottom(), gpu.get_top(), color=AMBER,
                   buff=0.15, stroke_width=3.0)
        cap3 = caption("Learning happens off-grid")
        self.play(
            hi_cell.animate.set_stroke(AMBER, width=3.0),
            FadeIn(big_mlp, run_time=0.5),
        )
        self.play(FadeIn(gpu), FadeIn(bp), FadeOut(cap2), FadeIn(cap3), run_time=0.6)
        self.wait(hold(beat - beat * 0.35 - 2.4, 0.3))

        # ----- Beat 3: the MLPs dissolve, the grid remains --------------------
        self.play(
            FadeOut(icons), FadeOut(big_mlp), FadeOut(gpu), FadeOut(bp),
            FadeOut(cap3), hi_cell.animate.set_stroke(STRUCT, width=1.2),
            run_time=0.8,
        )
        cap4 = caption("What if the grid is the network?")
        self.play(small.animate.move_to(ORIGIN), run_time=0.7)
        self.play(FadeIn(cap4, run_time=0.5))
        self.wait(hold(beat - 2.0, 0.5))
        self.wait(0.3)
