"""Act 2 — learning, without leaving the grid. Forward+backward, one cell, curve."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import (
    Scene, VGroup, RoundedRectangle, Text, Axes, Dot, DashedLine, FadeIn,
    FadeOut, Create, Indicate, BLACK, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)
from _common import (
    BG, AMBER, CYAN, STRUCT, INK, C_LOGTAU, C_WL, C_WT, C_B,
    make_grid, caption, target_duration, hold,
)

MONO = "Consolas"
ACC = [0.374, 0.324, 0.438, 0.500, 0.503, 0.494, 0.430, 0.479, 0.512, 0.522]


class Act2Learning(Scene):
    def construct(self):
        self.camera.background_color = BG
        total = target_duration(150.0)
        b1, b2, b3 = total * 0.40, total * 0.30, total * 0.30

        # ===== Beat 3a: forward + backward sweep ============================
        grid, cells = make_grid(28, 10, cell=0.2, gap=0.03)
        grid.scale(0.92).move_to(ORIGIN)
        rows, cols = 28, 10
        self.play(FadeIn(grid, lag_ratio=0.002), run_time=1.0)

        q = Text("So how does it learn?", font=MONO, color=INK).scale(0.6)
        q.to_edge(UP, buff=0.6)
        self.play(FadeIn(q), run_time=0.6)
        self.wait(hold(b1 * 0.12))

        fwd_cap = caption("forward pass — raster order, left-to-right, top-to-bottom")
        self.play(FadeIn(fwd_cap), run_time=0.4)
        order = [(r, c) for r in range(rows) for c in range(cols)]
        fwd_time = b1 * 0.34
        chunk = 20
        per = fwd_time / max(len(order) / chunk, 1)
        for i in range(0, len(order), chunk):
            self.play(*[cells[r][c].animate.set_fill(AMBER, opacity=0.85)
                        for (r, c) in order[i:i + chunk]], run_time=per)

        bwd_cap = caption("reverse raster — messages, not gradients-of-the-whole-graph")
        self.play(FadeOut(fwd_cap), FadeIn(bwd_cap), run_time=0.4)
        bwd_time = b1 * 0.34
        per = bwd_time / max(len(order) / chunk, 1)
        for i in range(0, len(order), chunk):
            batch = order[::-1][i:i + chunk]
            self.play(*[cells[r][c].animate.set_stroke(CYAN, width=2.2)
                        for (r, c) in batch], run_time=per)
        self.wait(hold(b1 * 0.1))

        # ===== Beat 3b: zoom into one cell during backward ==================
        self.play(FadeOut(bwd_cap), FadeOut(q), run_time=0.4)
        hr, hc = 14, 5
        target = cells[hr][hc]
        big = RoundedRectangle(
            width=2.4, height=2.4, corner_radius=0.25,
            stroke_color=CYAN, stroke_width=3, fill_color=BLACK, fill_opacity=1,
        ).shift(LEFT * 3.2)
        self.play(
            grid.animate.scale(0.55).to_edge(RIGHT, buff=1.0),
            FadeIn(big),
            target.animate.set_stroke(WHITE, width=3),
            run_time=0.9,
        )

        # eleven trace scalars: 3 self + 8 cross
        self_traces = VGroup(*[
            Text(f"e_self[{i}]", font=MONO, color=STRUCT).scale(0.34)
            for i in range(3)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        cross_traces = VGroup(*[
            Text(f"e_cross[{i}]", font=MONO, color=STRUCT).scale(0.34)
            for i in range(8)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        traces = VGroup(
            Text("11 traces", font=MONO, color=INK).scale(0.4),
            self_traces, cross_traces,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        traces.next_to(big, RIGHT, buff=0.5).shift(UP * 0.1)
        self.play(FadeIn(traces), run_time=0.6)

        params = VGroup(
            Text("log tau", font=MONO, color=C_LOGTAU).scale(0.4),
            Text("w_L", font=MONO, color=C_WL).scale(0.4),
            Text("w_T", font=MONO, color=C_WT).scale(0.4),
            Text("b", font=MONO, color=C_B).scale(0.4),
        ).arrange(DOWN, buff=0.18).move_to(big)
        self.play(FadeIn(params), run_time=0.5)

        msg_cap = caption("local rule — only this cell's traces, only its neighbours' messages")
        self.play(FadeIn(msg_cap), run_time=0.4)
        # backward message arrives; parameters update (flash)
        n_updates = 3
        upd_time = max(b2 - 3.4, 0.9)
        for _ in range(n_updates):
            self.play(
                Indicate(cross_traces, color=CYAN, scale_factor=1.05),
                run_time=upd_time / n_updates * 0.5,
            )
            self.play(
                *[Indicate(p, scale_factor=1.15) for p in params],
                run_time=upd_time / n_updates * 0.5,
            )
        self.wait(hold(b2 * 0.08))

        # ===== Beat 3c: the training curve ==================================
        self.play(
            FadeOut(big), FadeOut(traces), FadeOut(params), FadeOut(msg_cap),
            FadeOut(grid), run_time=0.6,
        )
        axes = Axes(
            x_range=[0, 9, 1], y_range=[0, 0.6, 0.1],
            x_length=8.5, y_length=4.6,
            axis_config={"color": STRUCT, "stroke_width": 2,
                         "include_numbers": True, "font_size": 22},
            tips=False,
        ).shift(DOWN * 0.3)
        x_lbl = Text("epoch", font=MONO, color=STRUCT).scale(0.4)
        x_lbl.next_to(axes.x_axis, DOWN, buff=0.3)
        y_lbl = Text("validation accuracy", font=MONO, color=STRUCT).scale(0.4)
        y_lbl.rotate(1.5708).next_to(axes.y_axis, LEFT, buff=0.3)
        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl), run_time=1.0)

        chance = DashedLine(
            axes.c2p(0, 0.10), axes.c2p(9, 0.10),
            color=STRUCT, stroke_width=2,
        )
        chance_lbl = Text("chance = 0.10", font=MONO, color=STRUCT).scale(0.36)
        chance_lbl.next_to(chance, UP, buff=0.08).align_to(chance, RIGHT)
        self.play(Create(chance), FadeIn(chance_lbl), run_time=0.6)

        from manim import Line
        pts = [axes.c2p(i, ACC[i]) for i in range(len(ACC))]
        dots = VGroup(*[Dot(p, color=AMBER, radius=0.06) for p in pts])
        curve = VGroup(*[
            Line(pts[i], pts[i + 1], color=AMBER, stroke_width=3)
            for i in range(len(pts) - 1)
        ])

        draw_time = max(b3 - 3.6, 1.2)
        per = draw_time / len(pts)
        self.play(FadeIn(dots[0]), run_time=per)
        for i in range(len(pts) - 1):
            self.play(Create(curve[i]), FadeIn(dots[i + 1]), run_time=per)

        final_lbl = Text("0.522", font=MONO, color=AMBER).scale(0.5)
        final_lbl.next_to(dots[-1], UP, buff=0.15)
        self.play(FadeIn(final_lbl), Indicate(dots[-1], scale_factor=1.6), run_time=0.6)
        end_cap = caption("1,120 parameters.  Local rule.  Row-wise MNIST.  52% accuracy.")
        self.play(FadeIn(end_cap), run_time=0.5)
        self.wait(hold(b3 * 0.12))
        self.wait(0.3)
