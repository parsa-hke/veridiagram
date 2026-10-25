"""Phase 8.2: queries solved by the probe only, the fine-tune only, both, or neither, per test set
(plan §8.2; mirrors the thesis's base-only / RL-only / both table).

    python -m analysis.headline_table [--probe sq_probe] [--finetune sq_finetune_colab] [--runs runs]

A query is *solved* by a model when every applicable relation decision for that query is correct
against the geometric labels, at the model's own operating thresholds tuned on its validation
predictions (`analysis.metrics.tune_thresholds`). Seeds are matched (probe seed k against fine-tune
seed k); counts are reported per seed and averaged over seeds. The same four-way split is also given
per (query, relation) decision, overall and per relation, which separates "the harder relation" from
"the harder diagram".

Writes reports/tables/headline_table.csv and headline_table.md. Defaults follow §0.2 #10: the
square-padded probe and the square-padded Colab retrain of 7.2.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.phase7 import latest_runs, split_name, write_csv
from analysis.phase8 import TEST_UNITS, load_predictions, run_thresholds, unit_rows
from data.queries import vocabulary
from experiments.run import REPO_ROOT, load_config

CATEGORIES = ("both", "probe_only", "finetune_only", "neither")


def four_way(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, int]:
    a, b = np.asarray(correct_a).astype(bool), np.asarray(correct_b).astype(bool)
    return {"both": int((a & b).sum()), "probe_only": int((a & ~b).sum()), "finetune_only": int((~a & b).sum()),
            "neither": int((~a & ~b).sum())}


def decisions(preds: dict, thresholds: np.ndarray, labels_key: str = "geo") -> np.ndarray:
    """(N, R) boolean: decision correct (masked-out entries count as correct so a query's all() ignores them)."""
    m = preds["mask"] > 0
    return ((preds["probs"] >= thresholds) == (preds[labels_key] > 0)) | ~m


def compare(pa: dict, pb: dict, ta: np.ndarray, tb: np.ndarray, vocab: list[str]) -> list[dict]:
    """Rows for one (probe run, fine-tune run, unit): query level and decision level (all + per relation)."""
    ca, cb = decisions(pa, ta), decisions(pb, tb)
    m = pa["mask"] > 0
    rows = [{"level": "query", "relation": "all", "n": int(len(ca)), **four_way(ca.all(1), cb.all(1))},
            {"level": "decision", "relation": "all", "n": int(m.sum()), **four_way(ca[m], cb[m])}]
    for j, r in enumerate(vocab):
        mj = m[:, j]
        rows.append({"level": "decision", "relation": r, "n": int(mj.sum()), **four_way(ca[mj, j], cb[mj, j])})
    return rows


def headline(runs: Path, latest: dict, probe: str, finetune: str, rows_of: dict, vocab: list[str],
             units: list[str] = TEST_UNITS) -> list[dict]:
    seeds_a = {split_name(r)[1]: r for r in latest if split_name(r)[0] == probe}
    seeds_b = {split_name(r)[1]: r for r in latest if split_name(r)[0] == finetune}
    shared = sorted(set(seeds_a) & set(seeds_b))
    if not shared:
        raise SystemExit(f"no matching seeds between {probe} ({sorted(seeds_a)}) and {finetune} ({sorted(seeds_b)})")
    out = []
    for u in units:
        per_seed = []
        for s in shared:
            ra, rb = seeds_a[s], seeds_b[s]
            pa, pb = load_predictions(runs, ra, u, rows_of[u]["qid"]), load_predictions(runs, rb, u, rows_of[u]["qid"])
            if pa is None or pb is None:
                continue
            ta, tb = run_thresholds(runs, ra, vocab, rows_of["val"]["qid"]), run_thresholds(runs, rb, vocab, rows_of["val"]["qid"])
            for row in compare(pa, pb, ta, tb, vocab):
                per_seed.append({"unit": u, "probe": probe, "finetune": finetune, "seed": s, **row})
        out.extend(per_seed)
        groups = defaultdict(list)
        for r in per_seed:
            groups[(r["level"], r["relation"])].append(r)
        for (level, rel), g in groups.items():
            row = {"unit": u, "probe": probe, "finetune": finetune, "seed": "mean", "level": level, "relation": rel,
                   "n": g[0]["n"], **{c: round(float(np.mean([r[c] for r in g])), 1) for c in CATEGORIES}}
            out.append(row)
    for r in out:
        for c in CATEGORIES:
            r[f"share_{c}"] = round(r[c] / r["n"], 4) if r["n"] else None
    return out


def markdown(rows: list[dict], vocab: list[str]) -> str:
    probe, ft = rows[0]["probe"], rows[0]["finetune"]
    seeds = sorted({r["seed"] for r in rows if r["seed"] != "mean"})
    lines = [f"### Queries solved by the probe ({probe}) and the fine-tune ({ft}); geometric labels, val-tuned thresholds; "
             f"mean over seeds {seeds}", "", "| test set | queries | both | fine-tune only | probe only | neither |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        if r["seed"] == "mean" and r["level"] == "query":
            lines.append(f"| {r['unit']} | {r['n']} | " + " | ".join(
                f"{r[c]:.0f} ({100 * r[f'share_{c}']:.1f}%)" for c in ("both", "finetune_only", "probe_only", "neither")) + " |")
    lines += ["", "### The same per (query, relation) decision", "",
              "| test set | relation | decisions | both | fine-tune only | probe only | neither |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["seed"] == "mean" and r["level"] == "decision":
            lines.append(f"| {r['unit']} | {r['relation']} | {r['n']} | " + " | ".join(
                f"{100 * r[f'share_{c}']:.1f}%" for c in ("both", "finetune_only", "probe_only", "neither")) + " |")
    return "\n".join(lines + [""])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--probe", default="sq_probe")
    ap.add_argument("--finetune", default="sq_finetune_colab")
    ap.add_argument("--out", default="headline_table", help="basename under reports/tables")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    runs = Path(args.runs) if args.runs else REPO_ROOT / cfg["paths"]["runs"]
    tables = REPO_ROOT / cfg["paths"]["tables"]
    vocab = vocabulary(cfg)
    rows = headline(runs, latest_runs(runs), args.probe, args.finetune, unit_rows(cfg), vocab)
    write_csv(rows, tables / f"{args.out}.csv")
    md = markdown(rows, vocab)
    (tables / f"{args.out}.md").write_text(md)
    print(md)
    print(f"-> {tables}/{args.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
