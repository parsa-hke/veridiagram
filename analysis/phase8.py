"""Phase 8.1: the evaluation table of every run with saved predictions, from runs/ (plan §8.1, §0.2 #10).

    python -m analysis.phase8 [--runs runs] [--resamples 1000] [--seed 0]

For every non-smoke `ok` run whose predictions match the current evaluation rows, per unit and label
set (geometric, asserted), at two operating points: threshold 0.5 (as Phase 7) and the per-relation
thresholds tuned on the run's own validation predictions against geometric labels (plan §8.2;
`analysis.metrics.tune_thresholds`):
  macro-F1 with a 95% bootstrap CI over diagrams (1,000 resamples), per-relation precision / recall /
  F1 with CIs, ECE (15 bins, pooled and per relation), supports, the thresholds used.
Per experiment (seeds pooled): mean and sd over seeds, and the bootstrap CI of the seed-mean macro-F1
(diagrams resampled jointly for all seeds), so the CI measures test-set sampling and the sd measures
training noise.

Writes
  reports/tables/phase8_metrics.csv    one row per run x unit x label set x operating point
  reports/tables/phase8_summary.csv    one row per experiment x unit x label set x operating point
  reports/tables/phase8_summary.md     the report's tables: geometric macro-F1 with CIs at the tuned
                                       thresholds and at 0.5, ECE, asserted macro-F1, per-relation F1
                                       of the report set
Report set (§0.2 #10 note): the square-padded re-evaluation `sq_<experiment>` wherever one exists
(7.2 = sq_finetune_colab); 7.6 and 7.7 keep their batch-padded runs and are compared with the
batch-padded `finetune`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import metrics as M
from analysis.phase7 import EXPERIMENTS, UNITS, eval_rows, latest_runs, split_name, write_csv
from data.queries import vocabulary
from experiments.run import REPO_ROOT, load_config

TEST_UNITS = [u for u in UNITS if u != "val"]
LABEL_SETS = (("geometric", "geo"), ("asserted", "asr"))
OPERATING = ("tuned", "0.5")
REPORT_SET = ["sq_probe", "sq_finetune_colab", "sq_scaling_1k", "sq_finetune_synth10k", "sq_scaling_50k",
              "sq_ablation_dinov2", "sq_ablation_nooverlay", "sq_selftrain_confidence", "sq_selftrain_consistency",
              "sq_selftrain_control", "finetune", "ablation_res224", "ablation_noaug"]


def unit_rows(cfg: dict, units: list[str] = UNITS) -> dict[str, dict]:
    """unit -> {"qid": array, "diagram": array of diagram ids} from the current evaluation rows."""
    out = {}
    for u in units:
        rows, _ = eval_rows(cfg, u)
        out[u] = {"qid": np.array([r["qid"] for r in rows]), "diagram": np.array([r["diagram_id"] for r in rows])}
    return out


def load_predictions(runs: Path, run: str, unit: str, expect_qid: np.ndarray | None = None) -> dict | None:
    p = runs / run / "predictions" / f"{unit}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if expect_qid is not None and (len(z["qid"]) != len(expect_qid) or (z["qid"] != expect_qid).any()):
        raise ValueError(f"{run}/{unit}: predictions do not match the current evaluation rows")
    return {k: z[k] for k in ("probs", "geo", "asr", "mask")}


def run_thresholds(runs: Path, run: str, vocab: list[str], val_qid: np.ndarray | None = None) -> np.ndarray:
    """Per-relation thresholds tuned on the run's own validation predictions (geometric labels)."""
    val = load_predictions(runs, run, "val", val_qid)
    if val is None:
        raise FileNotFoundError(f"{run}: no validation predictions to tune thresholds on")
    return M.tune_thresholds(val["probs"], val["geo"], val["mask"], vocab)


def metrics_row(preds: dict, labels_key: str, thresholds: np.ndarray | float, diagram: np.ndarray, vocab: list[str],
                n_resamples: int, seed: int) -> dict:
    pred = preds["probs"] >= thresholds
    per = M.per_relation(pred, preds[labels_key], preds["mask"], vocab)
    boot = M.bootstrap_macro_f1([(pred, preds[labels_key], preds["mask"])], diagram, vocab, n_resamples, seed)
    e = M.ece_per_relation(preds["probs"], preds[labels_key], preds["mask"], vocab)
    row = {"macro_f1": round(M.macro_f1(per), 4), "ci_lo": round(boot["lo"], 4), "ci_hi": round(boot["hi"], 4),
           "ece": round(M.ece(preds["probs"], preds[labels_key], preds["mask"]), 4),
           "n_queries": int(len(preds["probs"])), "n_diagrams": boot["n_diagrams"]}
    for j, r in enumerate(vocab):
        b = boot["per_relation"][r]
        row.update({f"f1_{r}": round(per[r]["f1"], 4), f"f1_lo_{r}": None if b["lo"] is None else round(b["lo"], 4),
                    f"f1_hi_{r}": None if b["hi"] is None else round(b["hi"], 4),
                    f"precision_{r}": round(per[r]["precision"], 4), f"recall_{r}": round(per[r]["recall"], 4),
                    f"support_{r}": per[r]["support"], f"ece_{r}": None if np.isnan(e[r]) else round(e[r], 4),
                    f"threshold_{r}": float(np.broadcast_to(thresholds, (len(vocab),))[j])})
    return row


def per_run_metrics(runs: Path, latest: dict, rows_of: dict, vocab: list[str], n_resamples: int = 1000,
                    seed: int = 0, units: list[str] = TEST_UNITS) -> tuple[list[dict], list[str]]:
    out, problems = [], []
    for run in sorted(latest):
        name, s = split_name(run)
        try:
            th = run_thresholds(runs, run, vocab, rows_of["val"]["qid"])
            preds = {u: load_predictions(runs, run, u, rows_of[u]["qid"]) for u in units}
        except (ValueError, FileNotFoundError) as ex:
            problems.append(str(ex))
            continue
        for u in units:
            if preds[u] is None:
                continue
            for ls, key in LABEL_SETS:
                for op, t in (("tuned", th), ("0.5", 0.5)):
                    out.append({"run": run, "experiment": name, "seed": s, "unit": u, "label_set": ls, "operating": op,
                                **metrics_row(preds[u], key, t, rows_of[u]["diagram"], vocab, n_resamples, seed)})
    return out, problems


def per_experiment_metrics(runs: Path, latest: dict, rows_of: dict, vocab: list[str], n_resamples: int = 1000,
                           seed: int = 0, units: list[str] = TEST_UNITS) -> list[dict]:
    """Seeds pooled: mean ± sd over seeds and the bootstrap CI of the seed-mean (joint resampling)."""
    groups = defaultdict(list)
    for run in sorted(latest):
        groups[split_name(run)[0]].append(run)
    out = []
    for exp, members in groups.items():
        try:
            ths = {r: run_thresholds(runs, r, vocab, rows_of["val"]["qid"]) for r in members}
        except (ValueError, FileNotFoundError):
            continue
        for u in units:
            try:
                preds = {r: load_predictions(runs, r, u, rows_of[u]["qid"]) for r in members}
            except ValueError:
                continue
            preds = {r: p for r, p in preds.items() if p is not None}
            if not preds:
                continue
            for ls, key in LABEL_SETS:
                for op in OPERATING:
                    sets = [(p["probs"] >= (ths[r] if op == "tuned" else 0.5), p[key], p["mask"]) for r, p in preds.items()]
                    boot = M.bootstrap_macro_f1(sets, rows_of[u]["diagram"], vocab, n_resamples, seed)
                    pts = np.array(boot["points"])
                    eces = [M.ece(p["probs"], p[key], p["mask"]) for p in preds.values()]
                    row = {"experiment": exp, "plan": EXPERIMENTS.get(exp, ("",))[0], "unit": u, "label_set": ls,
                           "operating": op, "n_seeds": len(pts), "seeds": " ".join(str(split_name(r)[1]) for r in preds),
                           "mean": round(float(pts.mean()), 4),
                           "sd": round(float(pts.std(ddof=1)), 4) if len(pts) > 1 else None,
                           "ci_lo": round(boot["lo"], 4), "ci_hi": round(boot["hi"], 4),
                           "ece_mean": round(float(np.mean(eces)), 4), "n_diagrams": boot["n_diagrams"]}
                    for r in vocab:
                        b = boot["per_relation"][r]
                        row.update({f"f1_{r}": None if b["f1"] is None else round(b["f1"], 4),
                                    f"f1_lo_{r}": None if b["lo"] is None else round(b["lo"], 4),
                                    f"f1_hi_{r}": None if b["hi"] is None else round(b["hi"], 4)})
                    out.append(row)
    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    return sorted(out, key=lambda r: (order.get(r["experiment"], 99), r["unit"] not in UNITS, UNITS.index(r["unit"])
                                      if r["unit"] in UNITS else 9, r["label_set"], r["operating"]))


def _cell(r: dict | None, ci: bool = True) -> str:
    if r is None:
        return ""
    s = f"{r['mean']:.3f}" + (f" ± {r['sd']:.3f}" if r["sd"] is not None else "")
    return s + (f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if ci else "")


def markdown(summary: list[dict], vocab: list[str], report_set: list[str] = REPORT_SET,
             units: list[str] = TEST_UNITS) -> str:
    by = defaultdict(dict)
    for r in summary:
        by[(r["experiment"], r["label_set"], r["operating"])][r["unit"]] = r
    exps = [e for e in report_set if any(k[0] == e for k in by)]
    lines = []

    def table(title, ls, op, ci=True, ece=False):
        lines.extend([f"### {title}", "", "| experiment | plan | seeds | " + " | ".join(units) + " |",
                      "|---|---|---|" + "---|" * len(units)])
        for e in exps:
            cells = by.get((e, ls, op), {})
            if not cells:
                continue
            any_r = next(iter(cells.values()))
            vals = [(f"{cells[u]['ece_mean']:.3f}" if u in cells else "") if ece else _cell(cells.get(u), ci) for u in units]
            lines.append(f"| {e} | {any_r['plan']} | {any_r['n_seeds']} | " + " | ".join(vals) + " |")
        lines.append("")

    table("Geometric macro-F1 at val-tuned thresholds: mean ± sd over seeds [95% bootstrap CI over diagrams of the seed-mean]",
          "geometric", "tuned")
    table("Geometric macro-F1 at threshold 0.5 (Phase 7's operating point)", "geometric", "0.5")
    table("ECE (15 bins, applicable entries pooled, geometric labels; mean over seeds)", "geometric", "tuned", ece=True)
    table("Asserted macro-F1 at val-tuned thresholds", "asserted", "tuned")
    # per-relation F1 of the report set, geometric, tuned
    for u in units:
        lines.extend([f"### Per-relation F1 on {u} (geometric labels, val-tuned thresholds; seed-mean [95% CI])", "",
                      "| experiment | " + " | ".join(vocab) + " |", "|---|" + "---|" * len(vocab)])
        for e in exps:
            r = by.get((e, "geometric", "tuned"), {}).get(u)
            if r is None:
                continue
            cells = []
            for v in vocab:
                f = r.get(f"f1_{v}")
                cells.append("" if f is None else f"{f:.3f} [{r[f'f1_lo_{v}']:.3f}, {r[f'f1_hi_{v}']:.3f}]")
            lines.append(f"| {e} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    runs = Path(args.runs) if args.runs else REPO_ROOT / cfg["paths"]["runs"]
    tables = REPO_ROOT / cfg["paths"]["tables"]
    vocab = vocabulary(cfg)
    latest = latest_runs(runs)
    rows_of = unit_rows(cfg)
    per_run, problems = per_run_metrics(runs, latest, rows_of, vocab, args.resamples, args.seed)
    write_csv(per_run, tables / "phase8_metrics.csv")
    summary = per_experiment_metrics(runs, latest, rows_of, vocab, args.resamples, args.seed)
    write_csv(summary, tables / "phase8_summary.csv")
    md = markdown(summary, vocab)
    (tables / "phase8_summary.md").write_text(md)
    print(md)
    for p in problems:
        print("WARNING:", p)
    print(f"{len(per_run)} per-run rows, {len(summary)} per-experiment rows -> {tables}/phase8_*.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
