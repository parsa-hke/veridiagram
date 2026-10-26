"""Figures for the methodology report (METHODOLOGY_PLAN.md §4), drawn with matplotlib as PDF.

  fig_recap.pdf     one construction, two readings: printed marks vs relations true in the drawing
  fig_design.pdf    the research design: training conditions x test sets x label sets
  fig_pipeline.pdf  the data pipeline from raw annotations to batches
plus copies of three existing figures from reports/figures/ (dataset examples, tolerance
calibration, synthetic examples) so the Overleaf project is self-contained.

    python scripts/methodology_figures.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")   # fixed PDF creation date: the figures regenerate byte for byte

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "methodology" / "figures"
COPIES = ["dataset_examples.png", "tolerance_all.png", "synthetic_examples.png"]

INK, GRAPHITE, LIGHT = "#16181D", "#8A9099", "#C9CDD3"
GREEN, GREEN_TEXT, GREEN_FILL = "#00C800", "#008A2E", "#E6F7E6"
MARK = "#D64541"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "pdf.fonttype": 42})


# ----------------------------------------------------------------------------- recap

def recap(path: Path) -> None:
    """Circle centre O(0,0) r=1, A(-1,0), B(1,0), C(0,1); t: x=-1 is tangent at A."""
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.75, 1.55); ax.set_ylim(-1.3, 1.5)
    A, B, C, O = (-1, 0), (1, 0), (0, 1), (0, 0)
    # true-but-unmarked relations: highlight behind the ink
    for seg in [((-1, -0.6), (-1, 1.3)), (A, B), (O, C)]:
        ax.plot(*zip(*seg), color=GREEN, lw=7, alpha=0.3, solid_capstyle="round", zorder=1)
    ax.add_patch(Circle(C, 0.075, fill=False, ec=GREEN, lw=1.8, zorder=1))
    # ink
    ax.add_patch(Circle(O, 1, fill=False, ec=INK, lw=1.1, zorder=2))
    for seg in [(A, B), (B, C), (C, A), (O, C), ((-1, -0.6), (-1, 1.3))]:
        ax.plot(*zip(*seg), color=INK, lw=1.1, zorder=3)
    for p, name, dx, dy in [(A, "A", -0.14, 0.08), (B, "B", 0.07, 0.08), (C, "C", 0.05, 0.08), (O, "O", 0.06, -0.17)]:
        ax.plot(*p, "o", color=INK, ms=3.5, zorder=4)
        ax.text(p[0] + dx, p[1] + dy, name, style="italic", fontsize=10, zorder=5)
    ax.text(-1.03, 1.34, "t", style="italic", fontsize=10)
    # printed marks (red): right angle at C, one tick on CA and CB
    s = 0.09
    ax.plot([-s, 0, s], [1 - s, 1 - 2 * s, 1 - s], color=MARK, lw=1.2, zorder=6)
    for mid, d in [((-0.5, 0.5), (1, 1)), ((0.5, 0.5), (1, -1))]:
        n = (d[0] * 0.05, d[1] * 0.05)
        ax.plot([mid[0] - n[0], mid[0] + n[0]], [mid[1] - n[1], mid[1] + n[1]], color=MARK, lw=1.4, zorder=6)
    # green marks and labels for the unmarked relations
    b = 0.1
    ax.plot([0, b, b], [b, b, 0], color=GREEN, lw=1.2, zorder=6)                # OC ⊥ AB at O
    ax.plot([-1, -1 + b, -1 + b], [-b, -b, 0], color=GREEN, lw=1.2, zorder=6)   # t ⊥ AB at A
    for x in (-1, 0):                                                           # parallel arrows t ∥ OC
        for y in (0.48, 0.56):
            ax.plot([x - 0.05, x, x + 0.05], [y - 0.05, y, y - 0.05], color=GREEN, lw=1.2, zorder=6)
    labels = [(-1.7, 0.52, "t $\\parallel$ OC"), (-1.7, -0.17, "t $\\perp$ AB"), (0.1, 0.33, "OC $\\perp$ AB"), (0.55, 1.22, "C on circle"),
              (-1.7, -0.6, "t tangent at A"), (0.0, -0.45, "A, O, B collinear")]
    for x, y, txt in labels:
        ax.text(x, y, txt, color=GREEN_TEXT, fontsize=8, zorder=7)
    for p, txt in [(A, "(−1, 0)"), (B, "(1, 0)"), (C, "(0, 1)"), (O, "(0, 0)")]:
        ax.text(p[0] + 0.05, p[1] - 0.33 if p != C else p[1] + 0.2, txt, color=GRAPHITE, fontsize=6.5, zorder=7)
    ax.plot([-1.55, -1.35], [-1.1, -1.1], color=GRAPHITE, lw=1.1); ax.plot([-1.45, -1.45], [-1.15, -1.05], color=MARK, lw=1.4)
    ax.text(-1.28, -1.13, "printed marks", color=GRAPHITE, fontsize=8)
    ax.plot([-0.1, 0.25], [-1.1, -1.1], color=GREEN, lw=7, alpha=0.3, solid_capstyle="round")
    ax.plot([-0.1, 0.25], [-1.1, -1.1], color=INK, lw=1.1)
    ax.text(0.32, -1.13, "true in the drawing", color=GREEN_TEXT, fontsize=8)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- design

def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8, lw=0.9, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, linespacing=1.25)


def arrow(ax, p, q, color=GRAPHITE, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, shrinkA=2, shrinkB=2))


def design(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.axis("off"); ax.set_xlim(0, 13.2); ax.set_ylim(0, 8.2)
    # Left: the intervention (training conditions)
    ax.text(0.2, 7.85, "Independent variable: training condition", fontsize=8.5, weight="bold")
    conds = ["real only (control, H0)", "real + synthetic 1k", "real + synthetic 10k", "real + synthetic 50k",
             "self-training: confidence", "self-training: + consistency", "self-training: no pseudo-labels"]
    for i, c in enumerate(conds):
        y = 6.9 - i * 0.78
        box(ax, 0.2, y, 3.9, 0.6, c, fc=GREEN_FILL if "synthetic" in c else "white", ec=LIGHT, fs=7.5)
    ax.text(0.2, 1.05, "Same budget per run (LP stage + 2 × 40k images),\nsame seeds (0, 1, 2), figure-disjoint data.",
            fontsize=7, color=GRAPHITE, va="top")
    # Middle: the model and the instrument
    box(ax, 4.9, 4.55, 3.0, 1.3, "DINOv3 ViT-L/16\n+ linear head\n(fallback: DINOv2-L/14)", ec=INK, fs=8)
    box(ax, 4.9, 2.55, 3.0, 1.2, "coordinate verifier\n(labels every query\nfrom point positions)", fc=GREEN_FILL, ec=GREEN, fs=8, color=GREEN_TEXT)
    ax.text(6.4, 6.15, "Model and instrument", fontsize=8.5, weight="bold", ha="center")
    for i in range(len(conds)):
        arrow(ax, (4.1, 7.2 - i * 0.78), (4.9, 5.2))
    # Right: outcomes
    ax.text(8.6, 7.85, "Dependent variable: macro-F1 on", fontsize=8.5, weight="bold")
    tests = ["geometry3k_test (same source)", "pgdp5k_heldout (unseen source)", "synth_extrap (12–20 points)",
             "not-to-scale subset"]
    for i, t in enumerate(tests):
        box(ax, 8.6, 6.55 - i * 0.78, 4.4, 0.6, t, ec=LIGHT, fs=7.5)
        arrow(ax, (7.9, 5.2), (8.6, 6.85 - i * 0.78))
    box(ax, 8.6, 2.6, 2.1, 0.75, "geometric labels\n(headline, H1)", fc=GREEN_FILL, ec=GREEN, fs=7.5, color=GREEN_TEXT)
    box(ax, 10.9, 2.6, 2.1, 0.75, "asserted labels\n(secondary, H2)", ec=LIGHT, fs=7.5)
    arrow(ax, (7.9, 3.15), (8.6, 2.98), color=GREEN)
    ax.text(8.6, 1.9, "Comparisons at fixed cost (one seed): linear probe, no overlay,\nDINOv2-L backbone, 224 px, no augmentation.",
            fontsize=7, color=GRAPHITE, va="top")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- pipeline

def pipeline(path: Path) -> None:
    steps = [("Raw annotations", "Geometry3K logic forms\nPGDP5K / PGPS9K primitives"),
             ("Unified schema", "points, segments, circles,\nasserted relations;\noffset recovery"),
             ("Figures and splits", "duplicate clusters;\n70/10/20 by figure;\nheld-out source; pool"),
             ("Verifier labels", "margins and tolerances;\ngeometric set;\nnot-to-scale flag"),
             ("Queries", "pairs by arity;\ncap 60 per diagram;\nmulti-label + mask"),
             ("Overlay", "resize to 448;\nhighlight the queried\nprimitives in green"),
             ("Balance and augment", "negatives ≤ 3× positives;\nrotation, flips, scale,\nnoise (training only)"),
             ("Batches", "normalised tensors,\ntwo label vectors,\napplicability mask")]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.axis("off"); ax.set_xlim(0, 16.4); ax.set_ylim(0, 8.0)
    w, gap, h = 3.85, 0.3, 3.2
    for i, (title, body) in enumerate(steps):
        row, col = divmod(i, 4)
        col = col if row == 0 else 3 - col                      # second row runs right to left
        x, y = 0.1 + col * (w + gap), 4.5 - row * (h + 0.9)
        green = title in ("Verifier labels", "Overlay")
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=GREEN_FILL if green else "white", ec=GREEN if green else LIGHT, lw=0.9))
        ax.text(x + w / 2, y + h - 0.45, f"{i + 1}. {title}", ha="center", va="center", fontsize=8, weight="bold",
                color=GREEN_TEXT if green else INK)
        ax.text(x + w / 2, y + h / 2 - 0.35, body, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.35)
        if i < len(steps) - 1:
            if i == 3:                                           # turn down to the second row
                arrow(ax, (x + w / 2, y), (x + w / 2, y - 0.9))
            elif row == 0:
                arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2))
            else:
                arrow(ax, (x, y + h / 2), (x - gap, y + h / 2))
    ax.text(0.1, 0.05, "Synthetic diagrams enter at step 3 with exact coordinates; their asserted set is the injected constructions.",
            fontsize=6.8, color=GRAPHITE)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recap(OUT / "fig_recap.pdf")
    design(OUT / "fig_design.pdf")
    pipeline(OUT / "fig_pipeline.pdf")
    for name in COPIES:
        shutil.copyfile(ROOT / "reports" / "figures" / name, OUT / name)
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
