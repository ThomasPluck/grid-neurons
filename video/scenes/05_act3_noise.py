"""Act 3 — where it breaks. Six knobs, the verdict (hero shot), the intuition."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from manim import (
    Scene, VGroup, RoundedRectangle, Rectangle, Text, Axes, Dot, Line,
    DashedLine, FadeIn, FadeOut, Create, Indicate, FunctionGraph,
    BLACK, WHITE, UP, DOWN, LEFT, RIGHT, ORIGIN,
)
from _common import (
    BG, AMBER, CYAN, STRUCT, INK, caption, target_duration, hold,
)

MONO = "Consolas"

KNOBS = [
    ("trace leak", "bias"),
    ("message gain asymmetry", "bias"),
    ("subtraction mismatch", "bias"),
    ("trace noise", "precision"),
    ("message noise", "precision"),
    ("combined", "precision"),
]

# stylised sweeps (knob 0..1 -> val accuracy). Robust stays near baseline 0.27;
# the message-noise path collapses to chance once sigma/RMS passes ~0.01.
BASELINE = 0.27
CHANCE = 0.10


def robust_curve(t):
    return BASELINE - 0.015 * t


def fragile_curve(t):
    # collapses fast: by t~0.1 of the knob range it is already at chance
    return CHANCE + (BASELINE - CHANCE) * np.exp(-t * 28.0)


class Act3Noise(Scene):
    def construct(self):
        self.camera.background_color = BG
        total = target_duration(165.0)
        b1, b2, b3 = total * 0.28, total * 0.40, total * 0.32

        # ===== Scene 4a: the six knobs ======================================
        intro = Text("real chips are noisy — the question is which noise",
                     font=MONO, color=INK).scale(0.5).to_edge(UP, buff=0.6)
        self.play(FadeIn(intro), run_time=0.6)

        tiles = VGroup()
        for idx, (label, kind) in enumerate(KNOBS):
            r, c = idx // 3, idx % 3
            col = AMBER if kind == "bias" else CYAN
            box = RoundedRectangle(width=3.3, height=1.7, corner_radius=0.15,
                                   stroke_color=col, stroke_width=2,
                                   fill_color=BLACK, fill_opacity=1.0)
            kindt = Text(kind, font=MONO, color=col).scale(0.32)
            namet = Text(label, font=MONO, color=INK).scale(0.34)
            VGroup(kindt, namet).arrange(DOWN, buff=0.18).move_to(box)
            tile = VGroup(box, kindt, namet)
            tile.move_to([(c - 1) * 3.6, 0.9 - r * 2.0, 0])
            tiles.add(tile)
        self.play(FadeIn(tiles, lag_ratio=0.1), run_time=b1 * 0.4)
        knob_cap = caption("three bias-style errors, three precision-style errors")
        self.play(FadeIn(knob_cap), run_time=0.4)
        self.wait(hold(b1 * 0.4))

        # ===== Scene 4b: the verdict (hero shot) ============================
        self.play(FadeOut(tiles), FadeOut(intro), FadeOut(knob_cap), run_time=0.5)

        def panel(title_txt, title_col, curve_fn, mark, mark_col, x_shift):
            ax = Axes(
                x_range=[0, 1, 0.25], y_range=[0, 0.32, 0.1],
                x_length=4.6, y_length=3.0,
                axis_config={"color": STRUCT, "stroke_width": 1.8,
                             "include_numbers": False},
                tips=False,
            )
            ax.shift(RIGHT * x_shift + DOWN * 0.3)
            head = Text(title_txt, font=MONO, color=title_col).scale(0.42)
            head.next_to(ax, UP, buff=0.35)
            xl = Text("noise knob ->", font=MONO, color=STRUCT).scale(0.3)
            xl.next_to(ax.x_axis, DOWN, buff=0.2)
            yl = Text("val acc", font=MONO, color=STRUCT).scale(0.3)
            yl.rotate(1.5708).next_to(ax.y_axis, LEFT, buff=0.15)
            base = DashedLine(ax.c2p(0, BASELINE), ax.c2p(1, BASELINE),
                              color=STRUCT, stroke_width=1.5)
            chance = DashedLine(ax.c2p(0, CHANCE), ax.c2p(1, CHANCE),
                                color=STRUCT, stroke_width=1.2)
            ts = np.linspace(0, 1, 60)
            pts = [ax.c2p(t, max(min(curve_fn(t), 0.32), 0.0)) for t in ts]
            line = VGroup(*[Line(pts[i], pts[i + 1], color=mark_col,
                                 stroke_width=3) for i in range(len(pts) - 1)])
            badge = Text(mark, color=mark_col).scale(0.9)
            badge.next_to(ax, DOWN, buff=0.55)
            return ax, head, xl, yl, base, chance, line, badge

        L = panel("robust paths", "#5FD08A", robust_curve, "OK", "#5FD08A", -3.6)
        R = panel("message noise", "#E0566A", fragile_curve, "X", "#E0566A", 3.6)

        self.play(
            Create(L[0]), Create(R[0]),
            FadeIn(L[1]), FadeIn(R[1]),
            FadeIn(L[2]), FadeIn(L[3]), FadeIn(R[2]), FadeIn(R[3]),
            run_time=b2 * 0.18,
        )
        self.play(
            Create(L[4]), Create(L[5]), Create(R[4]), Create(R[5]),
            run_time=b2 * 0.14,
        )
        cap_l = Text("trace leak 20%, trace noise 0.5 RMS, sub-mismatch 10%",
                     font=MONO, color="#5FD08A").scale(0.3)
        cap_l.next_to(L[0], DOWN, buff=1.15)
        cap_r = Text("collapses past sigma/RMS = 0.01", font=MONO,
                     color="#E0566A").scale(0.3)
        cap_r.next_to(R[0], DOWN, buff=1.15)
        self.play(Create(L[6]), FadeIn(L[7]), FadeIn(cap_l), run_time=b2 * 0.3)
        self.play(Create(R[6]), FadeIn(R[7]), FadeIn(cap_r), run_time=b2 * 0.3)
        self.wait(hold(b2 * 0.08))

        # ===== Scene 4c: the intuition ======================================
        self.play(
            *[FadeOut(m) for grp in (L, R) for m in grp],
            FadeOut(cap_l), FadeOut(cap_r), run_time=0.5,
        )
        # cross-trace = slow integrator (low-pass); message = single-step value
        ax2 = Axes(
            x_range=[0, 10, 2], y_range=[-1.5, 1.5, 1],
            x_length=9.0, y_length=3.4,
            axis_config={"color": STRUCT, "stroke_width": 1.5},
            tips=False,
        ).shift(DOWN * 0.2)
        rng = np.random.default_rng(11)
        xs = np.linspace(0, 10, 220)
        noise = rng.standard_normal(len(xs)) * 0.55
        # integrator: leaky-cumulative of noise -> smooth
        smooth = np.zeros_like(xs)
        acc = 0.0
        for i, nz in enumerate(noise):
            acc = 0.92 * acc + 0.08 * nz
            smooth[i] = acc * 3.0
        trace_pts = [ax2.c2p(xs[i], smooth[i]) for i in range(len(xs))]
        msg_pts = [ax2.c2p(xs[i], noise[i]) for i in range(len(xs))]
        trace_line = VGroup(*[Line(trace_pts[i], trace_pts[i + 1], color=CYAN,
                                   stroke_width=3) for i in range(len(xs) - 1)])
        msg_line = VGroup(*[Line(msg_pts[i], msg_pts[i + 1], color="#E0566A",
                                 stroke_width=1.8) for i in range(len(xs) - 1)])

        lt = Text("cross-trace — integrative, noise averages out", font=MONO,
                  color=CYAN).scale(0.36).to_edge(UP, buff=0.7)
        lm = Text("backward message — single-step, noise hits directly",
                  font=MONO, color="#E0566A").scale(0.36).next_to(lt, DOWN, buff=0.2)
        self.play(Create(ax2), FadeIn(lt), run_time=b3 * 0.2)
        self.play(Create(trace_line), run_time=b3 * 0.22)
        self.play(FadeIn(lm), run_time=b3 * 0.1)
        self.play(Create(msg_line), run_time=b3 * 0.22)
        intuition = caption("noise on the message is noise on the direction of learning itself")
        self.play(FadeIn(intuition), run_time=0.4)
        self.wait(hold(b3 * 0.2))
        self.wait(0.3)
