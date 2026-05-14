"""Aside — why snap one. The cone, the truncation, the update rule."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import (
    Scene, VGroup, RoundedRectangle, Text, FadeIn, FadeOut, Indicate,
    BLACK, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)
from _common import (
    BG, AMBER, CYAN, STRUCT, INK, C_LOGTAU, C_WL, C_WT, C_B,
    make_grid, caption, target_duration, hold,
)

MONO = "Consolas"


class AsideSnap1(Scene):
    def construct(self):
        self.camera.background_color = BG
        total = target_duration(75.0)
        b1, b2, b3 = total * 0.34, total * 0.30, total * 0.36

        rows, cols = 28, 10
        grid, cells = make_grid(rows, cols, cell=0.2, gap=0.03)
        grid.scale(0.95).move_to(ORIGIN)
        self.play(FadeIn(grid, lag_ratio=0.002), run_time=0.8)

        # ===== Scene A-a: the cone ==========================================
        sr, sc = 6, 2
        src = cells[sr][sc]
        self.play(src.animate.set_fill(WHITE, opacity=1.0), run_time=0.5)

        title = Text("a parameter only reaches cells downstream of it",
                     font=MONO, color=INK).scale(0.45).to_edge(UP, buff=0.5)
        self.play(FadeIn(title), run_time=0.4)

        counter = Text("reached: 0", font=MONO, color=AMBER).scale(0.5)
        counter.to_corner(UP + RIGHT, buff=0.7)
        self.play(FadeIn(counter), run_time=0.3)

        max_k = 6
        cone_time = b1 - 1.7
        per = cone_time / max_k
        reached = 0
        for k in range(1, max_k + 1):
            ring = [
                (r, c) for r in range(sr, rows) for c in range(sc, cols)
                if (r - sr) + (c - sc) == k
            ]
            reached += len(ring)
            new_counter = Text(f"reached: {reached}", font=MONO,
                               color=AMBER).scale(0.5).move_to(counter)
            self.play(
                *[cells[r][c].animate.set_fill(AMBER, opacity=0.55 + 0.05 * k)
                  for (r, c) in ring],
                counter.animate.become(new_counter),
                run_time=per,
            )
        cone_cap = caption("the downstream cone grows quadratically with distance")
        self.play(FadeIn(cone_cap), run_time=0.4)
        self.wait(hold(b1 * 0.1))

        # ===== Scene A-b: the truncation ====================================
        self.play(FadeOut(cone_cap), FadeOut(counter), FadeOut(title), run_time=0.4)
        # fade the cone away, keep source + immediate below/right neighbour
        keep = {(sr, sc), (sr + 1, sc), (sr, sc + 1)}
        fade_anims = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in keep and cells[r][c].get_fill_opacity() > 0.05:
                    fade_anims.append(cells[r][c].animate.set_fill(BLACK, opacity=1.0))
        cells[sr + 1][sc].set_stroke(CYAN, width=2)
        cells[sr][sc + 1].set_stroke(CYAN, width=2)
        self.play(
            *fade_anims,
            cells[sr + 1][sc].animate.set_fill(CYAN, opacity=0.7),
            cells[sr][sc + 1].animate.set_fill(CYAN, opacity=0.7),
            run_time=b2 * 0.35,
        )
        trunc_cap = caption("snap one — keep only one hop")
        self.play(FadeIn(trunc_cap), run_time=0.4)

        budget = VGroup(
            Text("self-traces        : 3", font=MONO, color=INK).scale(0.42),
            Text("cross-traces (x2)  : 8", font=MONO, color=INK).scale(0.42),
            Text("-------------------", font=MONO, color=STRUCT).scale(0.42),
            Text("total              : 11", font=MONO, color=AMBER).scale(0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        panel = VGroup(
            RoundedRectangle(width=4.6, height=2.2, corner_radius=0.15,
                             stroke_color=STRUCT, stroke_width=1.5,
                             fill_color=BLACK, fill_opacity=1.0),
            budget,
        )
        panel.to_edge(RIGHT, buff=0.9)
        self.play(FadeIn(panel), run_time=0.6)
        self.wait(hold(b2 * 0.4))

        # ===== Scene A-c: the update rule ===================================
        self.play(
            FadeOut(panel), FadeOut(trunc_cap), FadeOut(grid), run_time=0.6,
        )
        eq = Text("e(t+1)  =  a · e(t)  +  c · kappa · (1 - y²) · f",
                  font=MONO, color=INK,
                  t2c={"a · e(t)": CYAN, "kappa": AMBER}).scale(0.6)
        eq.shift(UP * 0.6)
        self.play(FadeIn(eq), run_time=b3 * 0.18)

        ann1 = Text("leaky memory of this trace", font=MONO, color=CYAN).scale(0.4)
        ann1.next_to(eq, DOWN, buff=0.9).shift(LEFT * 3.0)
        ann2 = Text("how my parameter pushes my neighbour", font=MONO,
                    color=AMBER).scale(0.4)
        ann2.next_to(eq, DOWN, buff=0.9).shift(RIGHT * 2.2)
        self.play(FadeIn(ann1), run_time=b3 * 0.16)
        self.play(FadeIn(ann2), run_time=b3 * 0.16)
        self.play(Indicate(eq, scale_factor=1.04), run_time=b3 * 0.14)

        local_cap = caption("local rule.  no global state.")
        self.play(FadeIn(local_cap), run_time=0.4)
        self.wait(hold(b3 * 0.25))
        self.wait(0.3)
