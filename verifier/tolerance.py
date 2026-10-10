"""Tolerance calibration from margin distributions (plan §3.4).

For each relation, compute the margin of every candidate query in the training split of
`geometry3k_figures` (plan §0.2 #9 revised), split by whether the query is asserted. Line
relations (parallel, perpendicular, collinear, tangent) are counted once per pair of carrier lines,
so a long line cut into many sub-segments does not dominate.

The threshold is the valley of the smoothed log-margin density of all candidates between the
asserted mode and the bulk of the non-asserted ones (centre of the gap if the valley is a flat,
empty gap). Where there is no valley (equal_length: unmarked equal lengths are common, so the
non-asserted density rises smoothly), the knee of the asserted CDF is used. Youden's J (max of
asserted recall minus non-asserted positive rate) is reported alongside as a cross-check. tangent has no asserted
positives, so tangent_rel = on_circle_rel (same quantity: |distance - r| / r).

Writes reports/figures/tolerance_<relation>.png, reports/figures/tolerance_all.png,
reports/tables/tolerance_calibration.csv and the `tolerances` section of the config.

    python -m verifier.tolerance [--write-config]

The config is only written with --write-config: the values in configs/base.yaml were revised after
the gold set (plan §0.2 #5), and a plain rerun must not overwrite them.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.queries import asserted_lookup, enumerate_queries  # noqa: E402
from data.schema import read_jsonl  # noqa: E402
from experiments.run import REPO_ROOT, load_config  # noqa: E402
from verifier.checks import evaluate  # noqa: E402
from verifier.primitives import bbox_diagonal  # noqa: E402

LINE_RELATIONS = {"parallel", "perpendicular", "collinear", "tangent"}
ANGLE_RELATIONS = {"parallel", "perpendicular"}
CALIBRATED = ("parallel", "perpendicular", "equal_length", "on_circle", "collinear")
LOG_MIN, LOG_MAX, BINS = -5.0, 0.5, 110
ASSERTED_C, OTHER_C, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"


def collect(diagrams, tier2=("collinear",)) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """relation -> (margins, asserted flags), deduplicated per carrier pair for line relations."""
    groups: dict[tuple, list] = defaultdict(lambda: [[], False])
    for d in diagrams:
        scale = bbox_diagonal(d.points.values())
        is_asserted = asserted_lookup(d)
        for q in enumerate_queries(d, tier2):
            for rel in q.relations:
                _, m = evaluate(d, rel, q.refs, scale=scale)
                if math.isnan(m):
                    continue
                key = (d.id, rel, q.carrier_key if rel in LINE_RELATIONS else q.key())
                g = groups[key]
                g[0].append(m)
                g[1] = g[1] or is_asserted(q, rel)
    out = defaultdict(lambda: ([], []))
    for (_, rel, _), (ms, a) in groups.items():
        out[rel][0].append(float(np.median(ms)))
        out[rel][1].append(a)
    return {r: (np.array(m), np.array(a, dtype=bool)) for r, (m, a) in out.items()}


def _logx(m: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(m, 10 ** LOG_MIN, 10 ** LOG_MAX))


def _smooth(h: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    k = np.arange(-4 * int(sigma), 4 * int(sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    return np.convolve(h, w / w.sum(), mode="same")


def valley(m: np.ndarray, a: np.ndarray) -> float | None:
    """Threshold at a real density valley between the asserted mode and the non-asserted bulk.

    The smoothed log-margin density of all candidates must dip to a minimum that is at most half
    the density on its left (towards the asserted mode) and that is not at the edge of the search
    region. If the minimum is a flat, near-empty gap, the threshold is the gap's centre (log scale).
    Returns None when there is no such valley.
    """
    if a.sum() < 10:
        return None
    edges = np.linspace(LOG_MIN, LOG_MAX, BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    x = _logx(m)
    h_all = _smooth(np.histogram(x, edges)[0].astype(float))
    h_neg = _smooth(np.histogram(x[~a], edges)[0].astype(float))
    mode_a = float(np.median(x[a]))
    right = np.nonzero(centres > mode_a)[0]
    if len(right) == 0:
        return None
    i_n = right[np.argmax(h_neg[right])]
    region = np.arange(right[0], i_n)
    if len(region) < 5:
        return None
    i_min = region[np.argmin(h_all[region])]
    left_peak = h_all[region[0]:i_min + 1].max()
    if i_min - region[0] < 2 or h_all[i_min] > 0.5 * left_peak:
        return None
    band = max(1.5 * h_all[i_min], h_all[i_min] + 0.01 * h_all[region].max())
    lo = hi = i_min
    while lo - 1 >= region[0] and h_all[lo - 1] <= band:
        lo -= 1
    while hi + 1 < i_n and h_all[hi + 1] <= band:
        hi += 1
    return float(10 ** ((centres[lo] + centres[hi]) / 2))


def knee(m_asserted: np.ndarray) -> float:
    """Knee of the asserted-margin CDF on a log axis (max distance above the chord between the
    1st and 99th percentiles): beyond it, widening the tolerance captures few more asserted
    relations, which are mostly drawn not to scale. Used when there is no valley."""
    x = np.log10(np.clip(np.sort(m_asserted), 10 ** LOG_MIN, None))
    y = np.arange(1, len(x) + 1) / len(x)
    lo, hi = np.quantile(x, [0.01, 0.99])
    sel = (x >= lo) & (x <= hi)
    x, y = x[sel], y[sel]
    xn = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    yn = (y - y[0]) / max(y[-1] - y[0], 1e-12)
    return float(10 ** x[int(np.argmax(yn - xn))])


def youden(m: np.ndarray, a: np.ndarray) -> float:
    cand = np.unique(np.quantile(m, np.linspace(0, 1, 400)))
    j = [(a & (m <= t)).sum() / max(a.sum(), 1) - (~a & (m <= t)).sum() / max((~a).sum(), 1) for t in cand]
    return float(cand[int(np.argmax(j))])


def to_tolerance(rel: str, thr: float) -> float:
    if rel in ANGLE_RELATIONS:
        return round(math.degrees(math.asin(min(thr, 1.0))), 2)
    return float(f"{thr:.3g}")


def plot(ax, rel: str, m: np.ndarray, a: np.ndarray, thr: float | None, tol: float | None) -> None:
    edges = np.linspace(LOG_MIN, LOG_MAX, BINS + 1)
    x = _logx(m)
    for sel, colour, label in ((a, ASSERTED_C, f"asserted (n={a.sum()})"), (~a, OTHER_C, f"not asserted (n={(~a).sum()})")):
        if sel.any():
            h, _ = np.histogram(x[sel], edges, density=True)
            ax.stairs(h, 10 ** edges, color=colour, linewidth=2, label=label)
    ax.set_xscale("log")
    if thr is not None:
        ax.axvline(thr, color=INK, linewidth=1, linestyle="--")
        unit = "°" if rel in ANGLE_RELATIONS else ""
        ax.text(thr, ax.get_ylim()[1] * 0.97, f" tol = {tol}{unit}", color=INK, fontsize=8, va="top")
    xl = {"parallel": "|sin θ|", "perpendicular": "|cos θ|", "equal_length": "| |s1|-|s2| | / max",
          "on_circle": "| |p-c|-r | / r", "collinear": "dist(p, line) / diagram scale"}[rel]
    ax.set_xlabel(f"margin: {xl} (log)", fontsize=8, color=MUTED)
    ax.set_ylabel("density (each class sums to 1)", fontsize=8, color=MUTED)
    ax.set_title(rel, fontsize=10, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c1")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def write_config(path: Path, tolerances: dict[str, float]) -> None:
    text = path.read_text()
    for key, value in tolerances.items():
        def repl(m, value=value):
            body = f"{m.group(1)} {value}"
            return body + " " * max(1, len(m.group(0)) - len(body)) if m.group(2) else body
        text, n = re.subn(rf"^(  {key}:)\s*[^\s#]+(\s+(?=#))?", repl, text, count=1, flags=re.M)
        if n != 1:
            raise KeyError(f"tolerances.{key} not found in {path}")
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--write-config", action="store_true", help="overwrite the tolerances section of the config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    figs, tabs = REPO_ROOT / cfg["paths"]["figures"], REPO_ROOT / cfg["paths"]["tables"]
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    train = [d for d in read_jsonl(proc / "geometry3k_figures.jsonl") if d.split == "train"]
    data = collect(train, tuple(cfg["relations"].get("tier2_included", [])))

    rows, tol, chosen = [], {}, {}
    for rel in CALIBRATED:
        if rel not in data:
            continue
        m, a = data[rel]
        v, y = valley(m, a), youden(m, a)
        thr = v if v is not None else knee(m[a])
        chosen[rel] = thr
        tol[rel] = to_tolerance(rel, thr)
        rows.append({"relation": rel, "n_asserted": int(a.sum()), "n_not_asserted": int((~a).sum()),
                     "valley_threshold": "" if v is None else f"{v:.4g}", "youden_threshold": f"{y:.4g}",
                     "method": "valley" if v is not None else "knee (no valley)", "margin_threshold": f"{thr:.4g}",
                     "tolerance": tol[rel], "unit": "deg" if rel in ANGLE_RELATIONS else "relative",
                     "asserted_within": f"{(a & (m <= thr)).sum() / max(a.sum(), 1):.3f}",
                     "not_asserted_within": f"{(~a & (m <= thr)).sum() / max((~a).sum(), 1):.3f}"})
        fig, ax = plt.subplots(figsize=(6, 3.4))
        plot(ax, rel, m, a, thr, tol[rel])
        fig.tight_layout()
        fig.savefig(figs / f"tolerance_{rel}.png", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, rel in zip(axes.flat, CALIBRATED):
        if rel in data:
            plot(ax, rel, *data[rel], chosen.get(rel), tol.get(rel))
    axes.flat[-1].axis("off")
    fig.suptitle("Verifier margins on the training split: asserted vs not asserted queries", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "tolerance_all.png", dpi=130)
    plt.close(fig)

    with open(tabs / "tolerance_calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)

    values = {"parallel_deg": tol.get("parallel"), "perpendicular_deg": tol.get("perpendicular"),
              "equal_length_rel": tol.get("equal_length"), "on_circle_rel": tol.get("on_circle"),
              "tangent_rel": tol.get("on_circle"), "collinear_rel": tol.get("collinear")}
    values = {k: v for k, v in values.items() if v is not None}
    print("tolerances:", values)
    if args.write_config:
        write_config(Path(args.config), values)
        print(f"written to {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
