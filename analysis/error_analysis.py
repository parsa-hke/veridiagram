"""Phase 8.4: where the model fails (plan §8.4), from the saved predictions of the square-padded runs.

    python -m analysis.error_analysis [--experiments sq_finetune_colab,sq_finetune_synth10k,...] [--runs runs]

For each experiment (seeds pooled; each run's own val-tuned thresholds, as in 8.1/8.2; geometric
labels) and each test unit, macro-F1 and per-relation F1 broken down by
  points      number of points in the diagram (bins: <=4, 5-6, 7-8, 9-12, 13-16, 17-20)
  segment     shortest queried segment relative to the diagram scale (bounding-box diagonal; the two
              segments of a seg_pair, the segment of a seg_circle, the line ab of a collinear query;
              pt_circle queries have no segment) - short segments are hard
  relation    per-relation precision / recall / F1 (the breakdown by relation)
  source      annotation source of the diagram (Geometry3K test figures carry PGDP5K or Geometry3K
              annotations, §0.2 #9c; PGDP5K held-out; synthetic)
  not_to_scale  the diagram's not-to-scale flag (Phase 3.6)
and, for every relation, the share of correct decisions against the verifier margin in units of the
relation's tolerance (margin / threshold; <= 1 is a geometric positive): bins [0, .25), [.25, .5),
[.5, 1), [1, 1.5), [1.5, 2), [2, 4), [4, inf). Near the tolerance the label is decided by a hair, so
this is where the model should fail relative to the verifier. The augmentation condition of the plan's
list is a training ablation (7.7), not a property of a test query, and is not broken down here.

Writes
  reports/tables/error_analysis.csv   experiment x unit x breakdown x group: n, macro-F1, per-relation F1 and support
  reports/tables/error_margin.csv     experiment x unit x relation x margin bin: n, positives, accuracy
  reports/tables/error_analysis.md    the first experiment's tables, readable
  reports/figures/error_margin.png    accuracy against margin / tolerance, one panel per unit (first experiment)
  reports/figures/error_breakdowns.png macro-F1 by points and by segment length (first experiment)
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import metrics as M
from analysis.phase7 import eval_rows, latest_runs, split_name, write_csv
from analysis.phase8 import TEST_UNITS, load_predictions, run_thresholds
from data.queries import vocabulary
from experiments.run import REPO_ROOT, load_config
from verifier.audit import margin_threshold
from verifier.primitives import bbox_diagonal

DEFAULT_EXPERIMENTS = ["sq_finetune_colab", "sq_finetune_synth10k", "sq_selftrain_control", "sq_probe", "sq_ablation_dinov2"]
POINT_BINS = [(0, 4, "<=4"), (5, 6, "5-6"), (7, 8, "7-8"), (9, 12, "9-12"), (13, 16, "13-16"), (17, 10 ** 9, "17-20")]
SEGMENT_BINS = [(0.0, 0.1, "<0.10"), (0.1, 0.2, "0.10-0.20"), (0.2, 0.35, "0.20-0.35"), (0.35, math.inf, ">=0.35")]
MARGIN_BINS = [(0.0, 0.25, "[0, 0.25)"), (0.25, 0.5, "[0.25, 0.5)"), (0.5, 1.0, "[0.5, 1)"), (1.0, 1.5, "[1, 1.5)"),
               (1.5, 2.0, "[1.5, 2)"), (2.0, 4.0, "[2, 4)"), (4.0, math.inf, "[4, inf)")]
NO_SEGMENT = "no segment"
BREAKDOWNS = ("points", "segment", "relation", "source", "not_to_scale")


def _bin(x: float, bins: list[tuple]) -> str:
    for lo, hi, name in bins:
        if lo <= x <= hi if bins is POINT_BINS else lo <= x < hi:
            return name
    return "other"


def _xy(d, name: str):
    p = d.points[name]
    return p if not hasattr(p, "xy") else p.xy


def queried_segments(d, refs: list) -> list[float]:
    """Lengths of the segments a query is about: ['seg', A, B] refs, and the line ab of a collinear
    query (three point refs: p, a, b)."""
    segs = [(r[1], r[2]) for r in refs if r[0] == "seg"]
    pts = [r[1] for r in refs if r[0] == "pt"]
    if not segs and len(pts) == 3:
        segs = [(pts[1], pts[2])]
    out = []
    for a, b in segs:
        (x1, y1), (x2, y2) = _xy(d, a), _xy(d, b)
        out.append(math.hypot(x1 - x2, y1 - y2))
    return out


def query_features(rows: list[dict], diagrams: dict) -> dict[str, np.ndarray]:
    """Per query: diagram id, point-count bin, relative segment-length bin, source, not-to-scale flag."""
    scale = {k: bbox_diagonal([_xy(d, n) for n in d.points]) for k, d in diagrams.items()}
    feats = defaultdict(list)
    for r in rows:
        d = diagrams[r["diagram_id"]]
        segs = queried_segments(d, r["refs"])
        feats["diagram"].append(r["diagram_id"])
        feats["points"].append(_bin(len(d.points), POINT_BINS))
        feats["segment"].append(_bin(min(segs) / scale[r["diagram_id"]], SEGMENT_BINS) if segs and scale[r["diagram_id"]] > 0
                                else NO_SEGMENT)
        feats["source"].append(str(d.meta.get("annotation") or d.source))
        feats["not_to_scale"].append("not to scale" if d.flags.get("not_to_scale") else "to scale")
        feats["margins"].append([math.nan if m is None else m for m in r["margins"]])
    return {k: np.array(v) for k, v in feats.items()}


def group_order(breakdown: str, groups: set[str]) -> list[str]:
    fixed = {"points": [b[2] for b in POINT_BINS], "segment": [b[2] for b in SEGMENT_BINS] + [NO_SEGMENT],
             "not_to_scale": ["to scale", "not to scale"]}
    if breakdown in fixed:
        return [g for g in fixed[breakdown] if g in groups] + sorted(groups - set(fixed[breakdown]))
    return sorted(groups)


def breakdown(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray, groups: np.ndarray, vocab: list[str],
              name: str) -> list[dict]:
    out = []
    for g in group_order(name, set(groups.tolist())):
        sel = groups == g
        per = M.per_relation(pred[sel], labels[sel], mask[sel], vocab)
        out.append({"breakdown": name, "group": g, "n_queries": int(sel.sum()), "n_decisions": int(mask[sel].sum()),
                    "macro_f1": round(M.macro_f1(per), 4),
                    **{f"f1_{r}": round(per[r]["f1"], 4) if per[r]["support"] else None for r in vocab},
                    **{f"support_{r}": per[r]["support"] for r in vocab}})
    return out


def relation_rows(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray, vocab: list[str]) -> list[dict]:
    per = M.per_relation(pred, labels, mask, vocab)
    return [{"breakdown": "relation", "group": r, "n_queries": int(mask[:, j].sum()), "n_decisions": int(mask[:, j].sum()),
             "macro_f1": None, "f1": round(per[r]["f1"], 4), "precision": round(per[r]["precision"], 4),
             "recall": round(per[r]["recall"], 4), "support": per[r]["support"]} for j, r in enumerate(vocab)]


def margin_table(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray, margins: np.ndarray, thresholds: dict[str, float],
                 vocab: list[str]) -> list[dict]:
    """Share of correct decisions per relation and margin bin (margin in units of the tolerance)."""
    out = []
    correct = (pred.astype(bool) == (labels > 0))
    for j, r in enumerate(vocab):
        ratio = margins[:, j] / thresholds[r]
        ok = (mask[:, j] > 0) & np.isfinite(ratio)
        for lo, hi, name in MARGIN_BINS:
            sel = ok & (ratio >= lo) & (ratio < hi)
            n = int(sel.sum())
            out.append({"relation": r, "bin": name, "bin_lo": lo, "n": n, "positives": int((labels[sel, j] > 0).sum()),
                        "accuracy": round(float(correct[sel, j].mean()), 4) if n else None})
    return out


def analyse(runs: Path, latest: dict, experiments: list[str], units: dict, val_qid: np.ndarray | None, vocab: list[str],
            thresholds: dict[str, float]) -> tuple[list[dict], list[dict], list[str]]:
    """units: test unit -> {"rows", "features"} from eval_rows / query_features; val_qid checks the
    validation predictions the thresholds are tuned on (None: no check)."""
    table, margin, problems = [], [], []
    for exp in experiments:
        members = sorted(r for r in latest if split_name(r)[0] == exp)
        if not members:
            problems.append(f"{exp}: no runs")
            continue
        for u, info in units.items():
            qid = np.array([r["qid"] for r in info["rows"]])
            parts, used = defaultdict(list), []
            for run in members:
                try:
                    p = load_predictions(runs, run, u, qid)
                    th = run_thresholds(runs, run, vocab, val_qid)
                except (ValueError, FileNotFoundError) as ex:
                    problems.append(str(ex))
                    continue
                if p is None:
                    continue
                parts["pred"].append(p["probs"] >= th)
                parts["labels"].append(p["geo"])
                parts["mask"].append(p["mask"])
                used.append(run)
            if not used:
                continue
            pred, labels, mask = (np.concatenate(parts[k]) for k in ("pred", "labels", "mask"))
            f = {k: np.concatenate([v] * len(used)) for k, v in info["features"].items()}
            base = {"experiment": exp, "unit": u, "seeds": " ".join(str(split_name(r)[1]) for r in used)}
            for name in ("points", "segment", "source", "not_to_scale"):
                table += [{**base, **row} for row in breakdown(pred, labels, mask, f[name], vocab, name)]
            table += [{**base, **row} for row in relation_rows(pred, labels, mask, vocab)]
            margin += [{**base, **row} for row in margin_table(pred, labels, mask, f["margins"], thresholds, vocab)]
    return table, margin, problems


def markdown(table: list[dict], margin: list[dict], vocab: list[str], experiment: str) -> str:
    units = list(dict.fromkeys(r["unit"] for r in table if r["experiment"] == experiment))
    lines = [f"## Error analysis of {experiment} (seeds pooled, geometric labels, val-tuned thresholds)", ""]
    for name in ("points", "segment", "source", "not_to_scale"):
        lines += [f"### Macro-F1 by {name}", "", "| group | " + " | ".join(f"{u} (n)" for u in units) + " |", "|---|" + "---|" * len(units)]
        by = defaultdict(dict)
        for r in table:
            if r["experiment"] == experiment and r["breakdown"] == name:
                by[r["group"]][r["unit"]] = r
        for g in group_order(name, set(by)):
            cells = [f"{by[g][u]['macro_f1']:.3f} ({by[g][u]['n_queries']})" if u in by[g] else "" for u in units]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Per-relation F1 by segment length (shortest queried segment / diagram scale)", ""]
    for u in units:
        lines += [f"{u}:", "", "| segment | " + " | ".join(vocab) + " |", "|---|" + "---|" * len(vocab)]
        for r in table:
            if r["experiment"] == experiment and r["unit"] == u and r["breakdown"] == "segment":
                lines.append(f"| {r['group']} | " + " | ".join("" if r[f"f1_{v}"] is None else f"{r[f'f1_{v}']:.3f} ({r[f'support_{v}']})" for v in vocab) + " |")
        lines.append("")
    lines += ["### Share of correct decisions against the verifier margin (in units of the tolerance; <= 1 is a geometric positive)", ""]
    for u in units:
        lines += [f"{u}:", "", "| relation | " + " | ".join(b[2] for b in MARGIN_BINS) + " |", "|---|" + "---|" * len(MARGIN_BINS)]
        for v in vocab:
            cells = [next((f"{r['accuracy']:.3f} ({r['n']})" if r["accuracy"] is not None else "-" for r in margin
                           if r["experiment"] == experiment and r["unit"] == u and r["relation"] == v and r["bin"] == b[2]), "")
                     for b in MARGIN_BINS]
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def figures(table: list[dict], margin: list[dict], vocab: list[str], experiment: str, figures_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    units = list(dict.fromkeys(r["unit"] for r in margin if r["experiment"] == experiment))
    out = []
    # 1. accuracy against margin / tolerance
    fig, axes = plt.subplots(1, len(units), figsize=(4.2 * len(units), 3.6), sharey=True, squeeze=False)
    xs = np.arange(len(MARGIN_BINS))
    for ax, u in zip(axes[0], units):
        for v in vocab:
            ys = [next((r["accuracy"] for r in margin if r["experiment"] == experiment and r["unit"] == u
                        and r["relation"] == v and r["bin"] == b[2]), None) for b in MARGIN_BINS]
            ax.plot(xs, [np.nan if y is None else y for y in ys], marker="o", ms=3, label=v)
        ax.axvline(2.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(xs, [b[2] for b in MARGIN_BINS], rotation=45, ha="right", fontsize=7)
        ax.set_title(u, fontsize=9)
        ax.set_xlabel("verifier margin / tolerance", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("share of correct decisions")
    axes[0][0].set_ylim(0, 1.02)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"{experiment}: decisions against the verifier margin (dashed: the tolerance)", fontsize=9)
    fig.tight_layout()
    p = figures_dir / "error_margin.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    # 2. macro-F1 by points and by segment length
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, name in zip(axes, ("points", "segment")):
        for u in units:
            rows = [r for r in table if r["experiment"] == experiment and r["unit"] == u and r["breakdown"] == name
                    and r["group"] != NO_SEGMENT]
            ax.plot([r["group"] for r in rows], [r["macro_f1"] for r in rows], marker="o", label=u)
        ax.set_xlabel("points in the diagram" if name == "points" else "shortest queried segment / diagram scale", fontsize=8)
        ax.set_ylabel("macro-F1 (geometric)")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{experiment}: macro-F1 by diagram size and segment length", fontsize=9)
    fig.tight_layout()
    p = figures_dir / "error_breakdowns.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS), help="comma-separated; the first one gets the figures")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    runs = Path(args.runs) if args.runs else REPO_ROOT / cfg["paths"]["runs"]
    tables, figs = REPO_ROOT / cfg["paths"]["tables"], REPO_ROOT / cfg["paths"]["figures"]
    vocab = vocabulary(cfg)
    thresholds = {r: margin_threshold(r, cfg["tolerances"]) for r in vocab}
    experiments = [e for e in args.experiments.split(",") if e]
    units = {}
    for u in TEST_UNITS:
        rows, diagrams = eval_rows(cfg, u)
        units[u] = {"rows": rows, "features": query_features(rows, diagrams)}
    val_qid = np.array([r["qid"] for r in eval_rows(cfg, "val")[0]])
    table, margin, problems = analyse(runs, latest_runs(runs), experiments, units, val_qid, vocab, thresholds)
    write_csv(table, tables / "error_analysis.csv")
    write_csv(margin, tables / "error_margin.csv")
    md = markdown(table, margin, vocab, experiments[0])
    (tables / "error_analysis.md").write_text(md)
    print(md)
    paths = figures(table, margin, vocab, experiments[0], figs)
    for p in problems:
        print("WARNING:", p)
    print(f"-> {tables}/error_analysis.{{csv,md}}, error_margin.csv, " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
