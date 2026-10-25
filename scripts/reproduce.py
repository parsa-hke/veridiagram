"""Regenerate every table in reports/tables/ and every figure in reports/figures/, then the reports.

One command (plan §10.1). Each step is an existing CLI, run in-process in dependency order and
timed; the timings go to reports/tables/reproduce_timings.csv so the README's runtime table is a
measurement, not a guess. Steps whose inputs are missing are skipped with the reason (for example
the analysis steps on a fresh clone without runs/, or attention maps without a checkpoint).

    python -m scripts.reproduce                    # everything whose inputs exist
    python -m scripts.reproduce --stages analysis report
    python -m scripts.reproduce --list             # print the steps and what each needs, run nothing

Stages:
  data      dataset, verifier and synthetic tables/figures   needs data/processed (python -m data.prepare)
  analysis  Phase 7-8 tables and figures                     needs runs/ (the experiment outputs)
  report    numbers.tex and figure copies for both reports, the Overleaf zips, the word count
Training is not a step: it is python -m experiments.queue <queue.yaml> (GPU hours, see README).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ROOT / "reports" / "tables" / "reproduce_timings.csv"


@dataclass
class Step:
    stage: str
    name: str
    module: str                     # importable module with main(argv)
    argv: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)   # paths relative to ROOT that must exist
    writes: str = ""

    def missing(self, root: Path = ROOT) -> list[str]:
        return [n for n in self.needs if not (root / n).exists()]

    def command(self) -> str:
        return " ".join(["python -m", self.module, *self.argv])


STEPS: list[Step] = [
    # ---- data: tables and figures from the processed datasets (no model)
    Step("data", "loader and coordinate checks", "data.sanity", needs=["data/processed"],
         writes="coordinate_sanity.csv, g3k_vs_pgdp5k_coords.csv, loader_check_*.png, image_sizes.png"),
    Step("data", "source overlap", "data.overlap", needs=["data/processed"], writes="source_overlap.csv"),
    Step("data", "relation census", "data.census", needs=["data/processed"], writes="relation_census.csv"),
    Step("data", "highlight colour check", "data.colour_check", needs=["data/processed"],
         writes="highlight_colour_check.csv"),
    Step("data", "tolerance calibration", "verifier.tolerance", needs=["data/processed"],
         writes="tolerance_calibration.csv, tolerance_*.png (configs/base.yaml untouched)"),
    Step("data", "asserted vs geometric audit", "verifier.audit", needs=["data/processed"],
         writes="audit_*.csv, audit_*_{asserted_not_geometric,geometric_not_asserted}.png, not_to_scale.csv"),
    Step("data", "gold-set scores", "verifier.gold", ["score"], needs=["data/processed", "verifier/gold"],
         writes="verifier_audit.csv, verifier_recheck.csv"),
    Step("data", "synthetic verifier check", "data.synthetic.check", needs=["data/processed"],
         writes="synthetic_verifier_check.csv, synthetic_examples.png"),
    Step("data", "dataset examples", "data.dataset", ["--config", "configs/smoke.yaml", "--examples"],
         needs=["data/processed"], writes="dataset_examples.png"),
    # ---- analysis: from the experiment outputs in runs/
    Step("analysis", "Phase 7 tables", "analysis.phase7", needs=["runs/log.csv"],
         writes="phase7_*.{csv,md}, padding_effect.csv, not_to_scale_eval.csv, selftrain_pseudo_labels.csv, scaling*.png"),
    Step("analysis", "CIs, ECE, tuned thresholds", "analysis.phase8", needs=["runs/log.csv"],
         writes="phase8_metrics.csv, phase8_summary.{csv,md}"),
    Step("analysis", "headline table", "analysis.headline_table", needs=["runs/sq_probe_s0", "runs/sq_finetune_colab_s0"],
         writes="headline_table.{csv,md}"),
    Step("analysis", "error analysis", "analysis.error_analysis", needs=["runs/sq_finetune_colab_s0"],
         writes="error_analysis.{csv,md}, error_margin.csv, error_margin.png, error_breakdowns.png"),
    Step("analysis", "both label sets", "analysis.asserted_vs_geometric", needs=["runs/sq_finetune_colab_s0", "data/processed"],
         writes="asserted_vs_geometric.{csv,md}, label_disagreement.csv, audit_examples_model.png"),
    Step("analysis", "attention maps", "analysis.attention_maps",
         needs=["runs/finetune_synth10k_s2/checkpoints/best.pt", "data/processed"],
         writes="attention_stats.csv, attention_summary.csv, attention_*.png"),
    Step("analysis", "per-relation bars and manifest", "analysis.figures", needs=["reports/tables/phase8_summary.csv"],
         writes="per_relation_f1.png, headline_table.png, reports/figures/README.md"),
    # ---- report: the two LaTeX projects (tables in, macros and figures out)
    Step("report", "methodology numbers", "scripts.methodology_numbers", writes="reports/methodology/numbers.tex"),
    Step("report", "methodology figures", "scripts.methodology_figures", writes="reports/methodology/figures/"),
    Step("report", "methodology zip", "scripts.methodology_zip", writes="reports/methodology/*.zip"),
    Step("report", "report numbers", "scripts.report_numbers", writes="reports/report/numbers.tex"),
    Step("report", "report figures", "scripts.report_figures", writes="reports/report/figures/"),
    Step("report", "report zip", "scripts.report_zip", writes="reports/Report_7744547.zip"),
    Step("report", "report word count", "scripts.report_wordcount", writes="(prints; exit 1 above 2,500)"),
]
STAGES = ["data", "analysis", "report"]


def load_main(module: str) -> Callable[[list[str]], int]:
    """The module's main, called with argv; a few report scripts take no arguments."""
    import importlib
    import inspect
    fn = importlib.import_module(module).main
    if not inspect.signature(fn).parameters:
        return lambda argv: fn()
    return fn


def run(steps: list[Step], root: Path = ROOT, loader=load_main, quiet: bool = False,
        timings: Path | None = TIMINGS) -> list[dict]:
    """Run the steps in order; skip those with missing inputs; stop at the first failure."""
    results = []
    for s in steps:
        miss = s.missing(root)
        if miss:
            results.append({"stage": s.stage, "step": s.name, "command": s.command(), "status": "skipped",
                            "seconds": "", "note": "missing " + ", ".join(miss)})
            print(f"[skip] {s.name}: missing {', '.join(miss)}", flush=True)
            continue
        print(f"[run ] {s.name}: {s.command()}", flush=True)
        t0 = time.time()
        sink = io.StringIO() if quiet else None
        with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
            try:
                code = loader(s.module)(list(s.argv))
            except SystemExit as e:                     # argparse or an explicit sys.exit
                code = e.code if isinstance(e.code, int) else 1
        dt = time.time() - t0
        status = "ok" if not code else f"exit {code}"
        results.append({"stage": s.stage, "step": s.name, "command": s.command(), "status": status,
                        "seconds": f"{dt:.1f}", "note": s.writes})
        print(f"[{'done' if status == 'ok' else 'FAIL'}] {s.name} in {dt:.1f}s", flush=True)
        if status != "ok":
            break
    if timings is not None:
        timings.parent.mkdir(parents=True, exist_ok=True)
        with open(timings, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stage", "step", "command", "status", "seconds", "note"])
            w.writeheader()
            w.writerows(results)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--list", action="store_true", help="print the steps and their inputs, run nothing")
    ap.add_argument("--quiet", action="store_true", help="hide each step's own output")
    args = ap.parse_args(argv)
    steps = [s for s in STEPS if s.stage in args.stages]
    if args.list:
        for s in steps:
            miss = s.missing()
            print(f"{s.stage:9s} {s.command():60s} {'ready' if not miss else 'missing ' + ', '.join(miss)}")
        return 0
    results = run(steps, quiet=args.quiet, timings=TIMINGS if args.stages == STAGES else None)
    failed = [r for r in results if r["status"] not in ("ok", "skipped")]
    total = sum(float(r["seconds"]) for r in results if r["seconds"])
    print(f"\n{sum(r['status'] == 'ok' for r in results)} ok, {sum(r['status'] == 'skipped' for r in results)} skipped, "
          f"{len(failed)} failed, {total / 60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
