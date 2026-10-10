"""Dataset audit: asserted (human annotation) vs geometric (verifier) labels (plan §3.8, Result 1).

For each source and relation type, counts over candidate queries:
  asserted ∧ geometric       the annotation and the drawing agree
  asserted ∧ ¬geometric      marked but not drawn so (not to scale, or annotation/coordinate error)
  ¬asserted ∧ geometric      true in the drawing but unmarked ("true but unmarked")
  ¬asserted ∧ ¬geometric
Line relations (parallel, perpendicular, collinear, tangent) are counted once per pair of carrier
lines (median margin of its sub-segment queries), as in the calibration, so long lines cut into
many sub-segments do not dominate. Reads data/processed/queries/<set>.csv.gz (python -m verifier.label).

Writes reports/tables/audit_<source>.csv and, per source, example figures
reports/figures/audit_<source>_asserted_not_geometric.png and
reports/figures/audit_<source>_geometric_not_asserted.png.

    python -m verifier.audit [--sources geometry3k pgdp5k pgps9k geometry3k_figures]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.schema import read_jsonl  # noqa: E402
from data.visualize import draw_query  # noqa: E402
from experiments.run import REPO_ROOT, load_config  # noqa: E402
from verifier.checks import TOL_KEYS  # noqa: E402

LINE_RELATIONS = {"parallel", "perpendicular", "collinear", "tangent"}
SOURCES = ("geometry3k", "pgdp5k", "pgps9k", "geometry3k_figures")
N_EXAMPLES = 8


def margin_threshold(rel: str, tol: dict) -> float:
    t = tol[TOL_KEYS[rel]]
    return math.sin(math.radians(t)) if rel in ("parallel", "perpendicular") else t


def grouped(path: Path, tol: dict) -> list[dict]:
    """One record per (diagram, relation, carrier pair | query) with asserted / geometric flags."""
    groups: dict[tuple, dict] = {}
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            rel = r["relation"]
            key = (r["diagram_id"], rel, r["carrier_key"] if rel in LINE_RELATIONS else r["refs"])
            g = groups.setdefault(key, {"diagram_id": r["diagram_id"], "relation": rel, "margins": [],
                                        "geo": [], "asserted": False, "refs": r["refs"]})
            g["asserted"] = g["asserted"] or r["asserted"] == "1"
            g["geo"].append(r["geometric"] == "1")
            if r["margin"]:
                g["margins"].append(float(r["margin"]))
    out = []
    for g in groups.values():
        m = float(np.median(g["margins"])) if g["margins"] else float("nan")
        if g["relation"] == "tangent":
            geometric = sum(g["geo"]) * 2 > len(g["geo"])     # tangent also needs the foot on the segment
        else:
            geometric = not math.isnan(m) and m <= margin_threshold(g["relation"], tol)
        out.append({**g, "margin": m, "geometric": geometric})
    return out


def table(records: list[dict]) -> list[dict]:
    rows = []
    for rel in sorted({r["relation"] for r in records}):
        R = [r for r in records if r["relation"] == rel]
        aa = sum(r["asserted"] and r["geometric"] for r in R)
        an = sum(r["asserted"] and not r["geometric"] for r in R)
        na = sum(not r["asserted"] and r["geometric"] for r in R)
        nn = len(R) - aa - an - na
        rows.append({"relation": rel, "candidates": len(R),
                     "asserted_and_geometric": aa, "asserted_not_geometric": an,
                     "geometric_not_asserted": na, "neither": nn,
                     "pct_asserted_confirmed": round(100 * aa / max(aa + an, 1), 1),
                     "pct_geometric_unmarked": round(100 * na / max(aa + na, 1), 1)})
    return rows


def examples(records: list[dict], cond, k: int, seed: int = 0) -> list[dict]:
    """Up to k records satisfying cond, spread round-robin over relation types."""
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for r in records:
        if cond(r):
            by_rel[r["relation"]].append(r)
    for v in by_rel.values():
        rng.shuffle(v)
    out, rels = [], sorted(by_rel)
    while len(out) < k and any(by_rel[x] for x in rels):
        for x in rels:
            if by_rel[x] and len(out) < k:
                out.append(by_rel[x].pop())
    return out


def example_figure(recs: list[dict], diagrams: dict, raw: Path, title: str, out: Path) -> None:
    if not recs:
        return
    cols = 4
    rows = math.ceil(len(recs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, r in zip(axes.flat, recs):
        d = diagrams[r["diagram_id"]]
        draw_query(ax, d, json.loads(r["refs"]), raw, title=f"{r['relation']} | {d.id}\nmargin {r['margin']:.3g}")
    for ax in axes.flat[len(recs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc, raw = REPO_ROOT / cfg["paths"]["processed"], REPO_ROOT / cfg["paths"]["raw"]
    figs, tabs = REPO_ROOT / cfg["paths"]["figures"], REPO_ROOT / cfg["paths"]["tables"]
    for src in args.sources:
        records = grouped(proc / "queries" / f"{src}.csv.gz", cfg["tolerances"])
        rows = table(records)
        with open(tabs / f"audit_{src}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"== {src}")
        for r in rows:
            print("  " + ", ".join(f"{k}={v}" for k, v in r.items()))
        diagrams = {d.id: d for d in read_jsonl(proc / f"{src}.jsonl")}
        example_figure(examples(records, lambda r: r["asserted"] and not r["geometric"], N_EXAMPLES), diagrams, raw,
                       f"{src}: asserted but not true in the drawing", figs / f"audit_{src}_asserted_not_geometric.png")
        example_figure(examples(records, lambda r: r["geometric"] and not r["asserted"], N_EXAMPLES), diagrams, raw,
                       f"{src}: true in the drawing but not asserted", figs / f"audit_{src}_geometric_not_asserted.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
