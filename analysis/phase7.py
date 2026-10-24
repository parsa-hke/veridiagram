"""Phase 7 wrap-up: results tables, the not-to-scale subset (7.4), the scaling curve (7.9) and the
Part B tables (7.11), regenerated from runs/ (plan §7, §0.2 #10).

    python -m analysis.phase7 [--runs runs]

Reads the latest `ok` line of every non-smoke run in <runs>/log.csv, its metrics.json and
predictions/<unit>.npz, and writes
  reports/tables/phase7_runs.csv        one row per run x unit x label set: macro-F1, per-relation F1, n
  reports/tables/phase7_summary.csv     per experiment x unit x label set: seeds, mean, sd, min, max
  reports/tables/phase7_summary.md      the same, geometric and asserted macro-F1, readable
  reports/tables/not_to_scale_eval.csv  7.4: every run on the queries of not-to-scale diagrams
                                        (geometry3k_test + pgdp5k_heldout pooled; 7 diagrams), both label sets
  reports/tables/selftrain_pseudo_labels.csv  7.11 post-hoc: pseudo-label precision per filter and relation
  reports/figures/scaling.png           7.9: macro-F1 against synthetic training size (0, 1k, 10k, 50k)
  reports/tables/padding_effect.csv     §0.2 #10 note: every run with a square-padded re-evaluation
                                        (sq_<run>) against its original batch-padded one, per unit:
                                        macro-F1 both ways, the change, and the share of decisions that flip
  reports/figures/scaling_square.png    7.9 from the square-padded re-evaluations (0 = finetune_colab)
Predictions are aligned to diagrams by rebuilding each unit's evaluation rows (same index, same
caps) and checking the query ids, so a stale prediction file is reported, not silently mismatched.
Thresholds are 0.5 here; Phase 8 adds val-tuned thresholds, bootstrap CIs and ECE.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.metrics import evaluate
from data.queries import vocabulary
from experiments.run import REPO_ROOT, load_config

UNITS = ["val", "geometry3k_test", "pgdp5k_heldout", "synth_extrap"]
NTS_UNITS = ["geometry3k_test", "pgdp5k_heldout"]
# experiment name -> (plan item, synthetic training size for the 7.9 curve or None)
EXPERIMENTS = {
    "probe": ("7.1 linear probe", None),
    "finetune": ("7.2 fine-tune (Part A: 0 synthetic)", 0),
    "scaling_1k": ("7.9 Part A: 1k synthetic", 1000),
    "finetune_synth10k": ("7.3 fine-tune + 10k synthetic (Part A: 10k)", 10000),
    "scaling_50k": ("7.9 Part A: 50k synthetic", 50000),
    "ablation_dinov2": ("7.5 DINOv2-L backbone", None),
    "ablation_res224": ("7.6 224 px", None),
    "ablation_noaug": ("7.7 no augmentation", None),
    "ablation_nooverlay": ("7.8 no overlay (query as tokens)", None),
    "selftrain_confidence": ("7.11 (a) confidence filter", None),
    "selftrain_consistency": ("7.11 (b) confidence + consistency filter", None),
    "selftrain_control": ("7.11 control: no pseudo-labels", None),
    "finetune_colab": ("7.2 fine-tune, retrained on Colab (same config)", None),
}
# §0.2 #10 note: square-padded re-evaluations sq_<experiment> of every experiment with a checkpoint
SQUARE = "sq_"
EXPERIMENTS.update({SQUARE + e: (f"{plan}, square padding", None) for e, (plan, _) in list(EXPERIMENTS.items())
                    if e not in ("ablation_res224", "ablation_noaug")})
SCALING_SIZES = {e: s for e, (_, s) in EXPERIMENTS.items() if s is not None}
SCALING_SIZES_SQUARE = {"sq_finetune_colab": 0, "sq_scaling_1k": 1000, "sq_finetune_synth10k": 10000,
                        "sq_scaling_50k": 50000}


def latest_runs(runs: Path) -> dict[str, dict]:
    """run name -> its latest ok log row (non-smoke) whose run dir has metrics.json."""
    out = {}
    with open(runs / "log.csv") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" and not row["name"].startswith("smoke") and (runs / row["name"] / "metrics.json").exists():
                out[row["name"]] = row
    return out


def split_name(run: str) -> tuple[str, int]:
    name, _, seed = run.rpartition("_s")
    return name, int(seed)


def eval_rows(cfg: dict, unit: str) -> tuple[list[dict], dict]:
    """The evaluation rows of a unit in prediction order, and their diagrams (no images loaded)."""
    from data.dataset import QueryDataset
    ds = QueryDataset(cfg, unit, train=False)
    return ds.rows, ds.diagrams


def per_run_table(runs: Path, latest: dict, vocab: list[str]) -> list[dict]:
    rows = []
    for run in sorted(latest):
        m = json.loads((runs / run / "metrics.json").read_text())
        name, seed = split_name(run)
        for unit, res in (m.get("results") or {}).items():
            for ls in ("geometric", "asserted"):
                r = res[ls]
                rows.append({"run": run, "experiment": name, "seed": seed, "unit": unit, "label_set": ls,
                             "macro_f1": r["macro_f1"], "n": res.get("n"),
                             **{f"f1_{k}": r["f1"].get(k) for k in vocab},
                             "backbone": m.get("backbone_loaded")})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["unit"], r["label_set"])].append(r)
    out = []
    for (exp, unit, ls), g in groups.items():
        v = np.array([r["macro_f1"] for r in g], float)
        out.append({"experiment": exp, "plan": EXPERIMENTS.get(exp, ("", None))[0], "unit": unit, "label_set": ls,
                    "seeds": " ".join(str(r["seed"]) for r in sorted(g, key=lambda r: r["seed"])),
                    "n_seeds": len(v), "mean": round(float(v.mean()), 4),
                    "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                    "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)})
    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    return sorted(out, key=lambda r: (order.get(r["experiment"], 99), r["label_set"], UNITS.index(r["unit"])
                                      if r["unit"] in UNITS else 9))


def markdown(summary: list[dict]) -> str:
    lines = []
    for ls in ("geometric", "asserted"):
        lines += [f"### Macro-F1 against {ls} labels (mean ± sd over seeds; threshold 0.5)", "",
                  "| experiment | plan | seeds | " + " | ".join(UNITS) + " |", "|---|---|---|" + "---|" * len(UNITS)]
        by = defaultdict(dict)
        for r in summary:
            if r["label_set"] == ls:
                by[r["experiment"]][r["unit"]] = r
        for exp in [e for e in EXPERIMENTS if e in by] + [e for e in by if e not in EXPERIMENTS]:
            cells = []
            for u in UNITS:
                r = by[exp].get(u)
                cells.append("" if r is None else f"{r['mean']:.3f}" + (f" ± {r['sd']:.3f}" if r["sd"] is not None else ""))
            any_r = next(iter(by[exp].values()))
            lines.append(f"| {exp} | {any_r['plan']} | {any_r['n_seeds']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def not_to_scale(runs: Path, latest: dict, cfg: dict, vocab: list[str]) -> tuple[list[dict], list[str]]:
    """7.4: metrics of every run on the queries of not-to-scale diagrams (real test units pooled)."""
    rows_of, flag_of = {}, {}
    for u in NTS_UNITS:
        rows, diagrams = eval_rows(cfg, u)
        rows_of[u] = rows
        flag_of[u] = np.array([bool(diagrams[r["diagram_id"]].flags.get("not_to_scale")) for r in rows])
    n_diagrams = sum(len({r["diagram_id"] for r, f in zip(rows_of[u], flag_of[u]) if f}) for u in NTS_UNITS)
    out, problems = [], []
    for run in sorted(latest):
        parts = defaultdict(list)
        ok = True
        for u in NTS_UNITS:
            p = runs / run / "predictions" / f"{u}.npz"
            if not p.exists():
                ok = False
                break
            z = np.load(p)
            if len(z["qid"]) != len(rows_of[u]) or (z["qid"] != np.array([r["qid"] for r in rows_of[u]])).any():
                problems.append(f"{run}/{u}: predictions do not match the current evaluation rows")
                ok = False
                break
            sel = flag_of[u]
            for k in ("probs", "geo", "asr", "mask"):
                parts[k].append(z[k][sel])
        if not ok:
            continue
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        name, seed = split_name(run)
        for ls, key in (("geometric", "geo"), ("asserted", "asr")):
            m = evaluate(cat["probs"], cat[key], cat["mask"], vocab)
            out.append({"run": run, "experiment": name, "seed": seed, "label_set": ls, "macro_f1": round(m["macro_f1"], 4),
                        "n_queries": int(len(cat["probs"])), "n_diagrams": n_diagrams,
                        **{f"f1_{r}": round(v["f1"], 4) for r, v in m["per_relation"].items()},
                        **{f"support_{r}": v["support"] for r, v in m["per_relation"].items()}})
    return out, problems


def pseudo_label_table(runs: Path, latest: dict, vocab: list[str]) -> list[dict]:
    """7.11 post-hoc: one block per (teacher, tau) from the first self-training run that has it."""
    seen, out = set(), []
    for run in sorted(latest):
        if not run.startswith("selftrain_"):
            continue
        m = json.loads((runs / run / "metrics.json").read_text())
        rep = m.get("pseudo_labels")
        if not rep or (m.get("teacher"), rep["tau"]) in seen:
            continue
        seen.add((m.get("teacher"), rep["tau"]))
        for part in ("confidence", "consistency", "removed_by_consistency"):
            s = rep[part]
            for rel in [*vocab, "all"]:
                r = s if rel == "all" else s[rel]
                out.append({"teacher": Path(m.get("teacher", "")).parent.parent.name, "tau": rep["tau"],
                            "consistency_negatives": rep.get("consistency_negatives"), "kept_by": part, "relation": rel,
                            "pseudo_pos": r["pos"], "pos_precision": r["pos_precision"],
                            "pseudo_neg": r.get("neg"), "neg_accuracy": r.get("neg_accuracy"),
                            "accuracy": s["accuracy"] if rel == "all" else None,
                            "pool_diagrams": rep["pool_diagrams"], "contradiction_diagrams": rep["contradiction_diagrams"],
                            "truth_contradiction_rate": rep["truth_contradiction_rate"]})
    return out


SCALING_UNITS = [("pgdp5k_heldout", "PGDP5K held-out source"), ("synth_extrap", "synthetic extrapolation"),
                 ("geometry3k_test", "Geometry3K test (in-distribution)")]


def scaling_points(rows: list[dict], size_of: dict[str, int] | None = None) -> list[dict]:
    """7.9: geometric macro-F1 per unit and synthetic training size, over seeds."""
    size_of = SCALING_SIZES if size_of is None else size_of
    pts = []
    for u, _ in SCALING_UNITS:
        for exp, size in sorted(size_of.items(), key=lambda kv: kv[1]):
            g = sorted((r for r in rows if r["experiment"] == exp and r["unit"] == u and r["label_set"] == "geometric"),
                       key=lambda r: r["seed"])
            if not g:
                continue
            v = [r["macro_f1"] for r in g]
            pts.append({"unit": u, "synthetic": size, "experiment": exp, "n_seeds": len(v),
                        "mean": round(float(np.mean(v)), 4),
                        "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
                        "seeds": [r["seed"] for r in g], "values": v})
    return pts


def scaling_figure(rows: list[dict], out: Path, size_of: dict[str, int] | None = None, title: str = "") -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = scaling_points(rows, size_of)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (u, label), c in zip(SCALING_UNITS, ("C0", "C3", "C7")):
        up = [p for p in pts if p["unit"] == u]
        xs = [max(p["synthetic"], 300) for p in up]             # 0 drawn at 300 on the log axis, labelled "0"
        for x, p in zip(xs, up):
            ax.scatter([x] * len(p["values"]), p["values"], color=c, s=12, alpha=0.5)
        if xs:
            ax.errorbar(xs, [p["mean"] for p in up], yerr=[p["sd"] or 0.0 for p in up], color=c, marker="o",
                        capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xticks([300, 1000, 10000, 50000], ["0", "1k", "10k", "50k"])
    ax.set_xlabel("synthetic training diagrams (50% of every batch when > 0)")
    ax.set_ylabel("macro-F1 (geometric labels)")
    ax.set_title("Part A: verifier-labelled synthetic data" + title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return pts


def padding_effect(runs: Path, latest: dict, vocab: list[str]) -> tuple[list[dict], list[str]]:
    """§0.2 #10 note: each square-padded re-evaluation sq_<run> against the original <run>, per unit.
    Decisions are (query, applicable relation) entries at threshold 0.5."""
    out, problems = [], []
    for sq in sorted(r for r in latest if r.startswith(SQUARE)):
        orig = sq[len(SQUARE):]
        if orig not in latest:
            continue
        ms = json.loads((runs / sq / "metrics.json").read_text())["results"]
        mo = json.loads((runs / orig / "metrics.json").read_text())["results"]
        name, seed = split_name(orig)
        for unit in [u for u in UNITS if u in ms and u in mo]:
            zs, zo = np.load(runs / sq / "predictions" / f"{unit}.npz"), np.load(runs / orig / "predictions" / f"{unit}.npz")
            if len(zs["qid"]) != len(zo["qid"]) or (zs["qid"] != zo["qid"]).any():
                problems.append(f"{sq}/{unit}: query ids differ from {orig}")
                continue
            m = zo["mask"] > 0
            flips = (zs["probs"] > 0.5) != (zo["probs"] > 0.5)
            row = {"run": orig, "experiment": name, "seed": seed, "unit": unit, "n_queries": int(len(zo["qid"])),
                   "n_decisions": int(m.sum()), "decisions_changed": int((flips & m).sum()),
                   "share_changed": round(float((flips & m).sum() / max(m.sum(), 1)), 4),
                   "max_abs_dp": round(float(np.abs(zs["probs"] - zo["probs"])[m].max()), 4) if m.any() else None}
            for ls in ("geometric", "asserted"):
                b_, s_ = mo[unit][ls]["macro_f1"], ms[unit][ls]["macro_f1"]
                row.update({f"{ls}_batch": b_, f"{ls}_square": s_, f"{ls}_change": round(s_ - b_, 4)})
            out.append(row)
    return out, problems


def padding_markdown(rows: list[dict]) -> str:
    """Mean change of geometric macro-F1 (square - batch) and share of decisions changed, per experiment and unit."""
    if not rows:
        return ""
    lines = ["### Square vs batch padding (§0.2 #10 note): geometric macro-F1 change, mean over seeds "
             "(share of decisions changed)", "",
             "| experiment | seeds | " + " | ".join(UNITS) + " |", "|---|---|" + "---|" * len(UNITS)]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["experiment"]][r["unit"]].append(r)
    for exp in [e for e in EXPERIMENTS if e in by]:
        cells = []
        for u in UNITS:
            g = by[exp].get(u, [])
            cells.append("" if not g else f"{np.mean([r['geometric_change'] for r in g]):+.3f} "
                         f"({100 * sum(r['decisions_changed'] for r in g) / sum(r['n_decisions'] for r in g):.1f}%)")
        lines.append(f"| {exp} | {len(next(iter(by[exp].values())))} | " + " | ".join(cells) + " |")
    return "\n".join(lines + [""])


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="runs directory (default: paths.runs of the base config)")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    runs = Path(args.runs) if args.runs else REPO_ROOT / cfg["paths"]["runs"]
    tables, figures = REPO_ROOT / cfg["paths"]["tables"], REPO_ROOT / cfg["paths"]["figures"]
    vocab = vocabulary(cfg)
    latest = latest_runs(runs)
    print(f"{len(latest)} runs: {', '.join(sorted(latest))}")
    rows = per_run_table(runs, latest, vocab)
    write_csv(rows, tables / "phase7_runs.csv")
    summary = summarise(rows)
    write_csv(summary, tables / "phase7_summary.csv")
    pad, pad_problems = padding_effect(runs, latest, vocab)
    if pad:
        write_csv(pad, tables / "padding_effect.csv")
    (tables / "phase7_summary.md").write_text(markdown(summary) + ("\n" + padding_markdown(pad) if pad else ""))
    nts, problems = not_to_scale(runs, latest, cfg, vocab)
    write_csv(nts, tables / "not_to_scale_eval.csv")
    pl = pseudo_label_table(runs, latest, vocab)
    if pl:
        write_csv(pl, tables / "selftrain_pseudo_labels.csv")
    pts = scaling_figure(rows, figures / "scaling.png")
    if any(r["experiment"] in SCALING_SIZES_SQUARE for r in rows):
        scaling_figure(rows, figures / "scaling_square.png", SCALING_SIZES_SQUARE, " (square padding)")
    print(markdown(summary))
    print(padding_markdown(pad))
    print("7.9 scaling points:", ", ".join(f"{p['unit']}@{p['synthetic']}: {p['mean']} ({p['n_seeds']} seeds)" for p in pts))
    missing = [e for e in EXPERIMENTS if not any(r["experiment"] == e for r in rows)]
    print("experiments without results:", ", ".join(missing) or "none")
    for p in problems + pad_problems:
        print("WARNING:", p)
    print(f"-> {tables}/phase7_*.{{csv,md}}, not_to_scale_eval.csv"
          + (", selftrain_pseudo_labels.csv" if pl else "") + (", padding_effect.csv" if pad else "")
          + f", {figures}/scaling.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
