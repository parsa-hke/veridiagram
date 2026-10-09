"""Coordinate sanity checks (plan §2.7).

For each source: image-size distribution (reports/figures/image_sizes.png) and the number of
annotated points / circle centres outside the image (reports/tables/coordinate_sanity.csv).
Also measures Geometry3K coordinate accuracy against PGDP5K, which re-annotated the same images
in pixels: for exact duplicates of equal size, distance between same-named points
(reports/tables/g3k_vs_pgdp5k_coords.csv, reports/figures/g3k_vs_pgdp5k_coords.png).

    python -m data.sanity
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.schema import read_jsonl  # noqa: E402
from experiments.run import REPO_ROOT, load_config  # noqa: E402

SOURCES = ("geometry3k", "pgdp5k", "pgps9k")
MARGIN_PX = 2.0  # a point this far outside the border still counts as inside (anti-aliased edges)


def outside(x: float, y: float, w: int, h: int, m: float = MARGIN_PX) -> bool:
    return x < -m or y < -m or x > w + m or y > h + m


def size_figure(diagrams: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(diagrams), figsize=(4.2 * len(diagrams), 3.8), sharex=True, sharey=True)
    for ax, (src, ds) in zip(axes, diagrams.items()):
        w = np.array([d.width for d in ds])
        h = np.array([d.height for d in ds])
        ax.scatter(w, h, s=3, alpha=0.3, color="#2a6fdb", linewidths=0)
        ax.set_title(f"{src} (n={len(ds)})\nmedian {int(np.median(w))}x{int(np.median(h))}", fontsize=9)
        ax.set_xlabel("width (px)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("height (px)")
    fig.suptitle("Image size distribution per source", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def bounds_table(diagrams: dict) -> list[dict]:
    rows = []
    for src, ds in diagrams.items():
        n_pts = n_out = n_diag_out = n_c = n_c_out = 0
        worst = []
        for d in ds:
            bad = [n for n, (x, y) in d.points.items() if outside(x, y, d.width, d.height)]
            n_pts += len(d.points)
            n_out += len(bad)
            n_diag_out += bool(bad)
            if bad:
                worst.append(d.id)
            for c in d.circles:
                if c.center_xy:
                    n_c += 1
                    n_c_out += outside(*c.center_xy, d.width, d.height)
        rows.append({"source": src, "diagrams": len(ds), "points": n_pts, "points_outside": n_out,
                     "pct_points_outside": round(100 * n_out / max(n_pts, 1), 2),
                     "diagrams_with_outside_point": n_diag_out,
                     "pct_diagrams": round(100 * n_diag_out / max(len(ds), 1), 2),
                     "circle_centres": n_c, "centres_outside": n_c_out,
                     "examples": " ".join(worst[:5])})
    return rows


def g3k_vs_pgdp(diagrams: dict, proc: Path) -> tuple[list[float], list[dict]]:
    g = {d.id: d for d in diagrams["geometry3k"]}
    p = {d.id: d for d in diagrams["pgdp5k"]}
    dists, per = [], []
    with open(proc / "overlap.csv") as f:
        for r in csv.DictReader(f):
            if not (r["id_a"] in g and r["id_b"] in p and r["size_a"] == r["size_b"] and float(r["mad64"]) <= 1.0):
                continue
            a, b = g[r["id_a"]], p[r["id_b"]]
            common = [n for n in a.points if n in b.points and not n.startswith("_")]
            if not common:
                continue
            ds = [math.dist(a.points[n], b.points[n]) for n in common]
            dists += ds
            per.append({"geometry3k_id": a.id, "pgdp5k_id": b.id, "n_common_points": len(common),
                        "median_px": round(float(np.median(ds)), 2), "max_px": round(max(ds), 2),
                        "diag_px": round(math.hypot(a.width, a.height), 1)})
    return dists, per


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    figs, tabs = REPO_ROOT / cfg["paths"]["figures"], REPO_ROOT / cfg["paths"]["tables"]
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    diagrams = {s: list(read_jsonl(proc / f"{s}.jsonl")) for s in SOURCES}

    size_figure(diagrams, figs / "image_sizes.png")
    rows = bounds_table(diagrams)
    with open(tabs / "coordinate_sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: v for k, v in r.items() if k != "examples"})

    if (proc / "overlap.csv").exists():
        dists, per = g3k_vs_pgdp(diagrams, proc)
        with open(tabs / "g3k_vs_pgdp5k_coords.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per[0]))
            w.writeheader()
            w.writerows(per)
        d = np.array(dists)
        q = np.percentile(d, [50, 75, 90, 95, 99]).round(2)
        print(f"G3K vs PGDP5K same-named points: n={len(d)} in {len(per)} image pairs; "
              f"quantiles 50/75/90/95/99 = {q} px; >5px {100 * (d > 5).mean():.1f}%, >10px {100 * (d > 10).mean():.1f}%, "
              f">20px {100 * (d > 20).mean():.1f}%; image pairs with any point >10px: "
              f"{sum(r['max_px'] > 10 for r in per)}")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.hist(np.clip(d, 0.1, 200), bins=np.logspace(-1, np.log10(200), 50), color="#2a6fdb")
        ax.set_xscale("log")
        ax.set_xlabel("distance between Geometry3K and PGDP5K position of the same point (px)")
        ax.set_ylabel("points")
        ax.set_title(f"Geometry3K coordinate error vs PGDP5K re-annotation (n={len(d)})", fontsize=9)
        fig.tight_layout()
        fig.savefig(figs / "g3k_vs_pgdp5k_coords.png", dpi=130)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
