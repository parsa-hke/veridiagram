"""Which saturated colours already occur in real diagrams (plan §5.3: why the highlight is green).

    python -m data.colour_check [--n 600] [--seed 0]

Samples n real diagrams (equal shares of Geometry3K, PGDP5K and PGPS9K, fixed seed), flattens each
on white and counts (a) its saturated pixels by hue family and (b) its pixels within RGB distance r of
the highlight colour (configs overlay.color), for several fixed r. An image "has" a family or a
near-highlight colour when at least `min_pixels` of its pixels qualify. Writes
reports/tables/highlight_colour_check.csv, one row per test: images that have it, and their share.
Re-derives the Phase 5.3 finding (2026-10-11, recorded without a script) on a scripted sample: the
2026-10-13 run finds green-family ink in 1% of images (darker / teal greens in coloured textbook
figures) but nothing within distance 80 of the highlight's (0, 200, 0).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from data.overlay import flatten
from data.schema import read_jsonl
from experiments.run import REPO_ROOT, load_config

# hue family -> [lo, hi) in degrees; red wraps around 0
HUES = {"red": (345, 15), "orange": (15, 45), "yellow": (45, 70), "green": (70, 165), "cyan": (165, 200),
        "blue": (200, 260), "magenta": (260, 345)}
SOURCES = ("geometry3k", "pgdp5k", "pgps9k")
RADII = (40, 60, 80, 100)
MIN_PIXELS = 20


def hue_families(im: Image.Image, sat: float = 0.5, val: float = 0.35, min_pixels: int = MIN_PIXELS) -> dict[str, int]:
    """Saturated pixel count per hue family (families under `min_pixels` count as 0)."""
    hsv = np.asarray(flatten(im).convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    on = (s >= sat) & (v >= val)
    out = {}
    for name, (lo, hi) in HUES.items():
        m = on & (((h >= lo) | (h < hi)) if lo > hi else ((h >= lo) & (h < hi)))
        c = int(m.sum())
        out[name] = c if c >= min_pixels else 0
    return out


def near_colour(im: Image.Image, colour: tuple, radius: float) -> int:
    """Pixels within RGB distance `radius` of `colour`."""
    a = np.asarray(flatten(im), dtype=np.float32)
    return int((np.sqrt(((a - np.asarray(colour, np.float32)) ** 2).sum(-1)) < radius).sum())


def sample(processed: Path, n: int, seed: int) -> list:
    rng = random.Random(seed)
    per = [n // len(SOURCES) + (i < n % len(SOURCES)) for i in range(len(SOURCES))]
    out = []
    for src, k in zip(SOURCES, per):
        ds = list(read_jsonl(processed / f"{src}.jsonl"))
        out += rng.sample(ds, min(k, len(ds)))
    return out


def table(images: list[Image.Image], highlight: tuple) -> list[dict]:
    counts = {f"hue_{k}": 0 for k in HUES} | {f"near_highlight_r{r}": 0 for r in RADII}
    for im in images:
        for k, c in hue_families(im).items():
            counts[f"hue_{k}"] += c > 0
        for r in RADII:
            counts[f"near_highlight_r{r}"] += near_colour(im, highlight, r) >= MIN_PIXELS
    n = len(images)
    return [{"test": k, "images": n, "images_with": c, "pct": round(100 * c / n, 1) if n else None,
             "min_pixels": MIN_PIXELS} for k, c in counts.items()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    raw, processed = REPO_ROOT / cfg["paths"]["raw"], REPO_ROOT / cfg["paths"]["processed"]
    diagrams = sample(processed, args.n, args.seed)
    images = []
    for d in diagrams:
        with Image.open(raw / d.image_path) as im:
            images.append(im.copy())
    rows = table(images, tuple(cfg["overlay"]["color"]))
    out = REPO_ROOT / cfg["paths"]["tables"] / "highlight_colour_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['test']:22s} {r['images_with']:4d} / {r['images']}  ({r['pct']}%)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
