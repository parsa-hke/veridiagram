"""Phase 8.3: the model against both label sets, and which "errors" are the labels' (plan §8.3, §0.2 #4).

    python -m analysis.asserted_vs_geometric [--experiments sq_finetune_colab,...] [--runs runs]

Two ground truths exist for every applicable (query, relation) entry: `asserted` (a human marked the
relation) and `geometric` (the coordinate verifier within the fixed tolerance). They agree on most
entries; where they disagree, either the drawing does not show what the text asserts
(asserted-only: drawn out of tolerance, often a not-to-scale figure) or the drawing shows a relation
nobody marked (geometric-only: unmarked but true). The gold set (Phase 3.5) made the verifier the
authority, so a model decision that follows the verifier against a human label is the label's error,
not the model's.

For each experiment (seeds pooled; val-tuned thresholds, 8.1) and test unit:
  asserted_vs_geometric.csv   macro-F1 and per-relation F1 against both label sets, side by side
  label_disagreement.csv      per unit x relation: the four label cells (agree-positive, agree-negative,
                              asserted-only, geometric-only), the model's positive rate in each, and
                              the attribution of its errors: of the errors against asserted labels, the
                              share where the model sides with the verifier (labels' errors); of the
                              errors against geometric labels, the share where it sides with the human
                              mark (the model read the mark, not the drawing)
  asserted_vs_geometric.md    the first experiment's tables, readable
  reports/figures/audit_examples_model.png   four asserted-only and four geometric-only entries from the
                              real test sets with the verifier margin and the model's probability
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

DEFAULT_EXPERIMENTS = ["sq_finetune_colab", "sq_finetune_synth10k", "sq_selftrain_control", "sq_probe", "sq_ablation_dinov2"]
CELLS = ("agree_pos", "agree_neg", "asserted_only", "geometric_only")
REAL_UNITS = ["geometry3k_test", "pgdp5k_heldout"]


def cells(geo: np.ndarray, asr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    g, a, m = geo > 0, asr > 0, mask > 0
    return {"agree_pos": g & a & m, "agree_neg": ~g & ~a & m, "asserted_only": a & ~g & m, "geometric_only": g & ~a & m}


def disagreement_rows(pred: np.ndarray, geo: np.ndarray, asr: np.ndarray, mask: np.ndarray, vocab: list[str],
                      nts: np.ndarray | None = None) -> list[dict]:
    """Per relation and pooled: cell counts, the model's positive rate per cell, error attribution."""
    c = cells(geo, asr, mask)
    pred, g, a, m = pred.astype(bool), geo > 0, asr > 0, mask > 0
    out = []
    for j, rel in enumerate([*vocab, "all"]):
        sel = slice(None) if rel == "all" else [j]
        row = {"relation": rel, "n": int(m[:, sel].sum())}
        for name, cm in c.items():
            cs = cm[:, sel]
            row[f"n_{name}"] = int(cs.sum())
            row[f"pos_rate_{name}"] = round(float(pred[:, sel][cs].mean()), 4) if cs.any() else None
        if nts is not None:
            row["asserted_only_not_to_scale"] = int((c["asserted_only"][:, sel] & nts[:, None]).sum())
        err_a = (pred != a)[:, sel] & m[:, sel]
        err_g = (pred != g)[:, sel] & m[:, sel]
        row["errors_vs_asserted"] = int(err_a.sum())
        row["errors_vs_asserted_siding_with_verifier"] = int((err_a & (pred == g)[:, sel]).sum())
        row["share_asserted_errors_that_are_label_errors"] = round(row["errors_vs_asserted_siding_with_verifier"] / row["errors_vs_asserted"], 4) if row["errors_vs_asserted"] else None
        row["errors_vs_geometric"] = int(err_g.sum())
        row["errors_vs_geometric_siding_with_mark"] = int((err_g & (pred == a)[:, sel]).sum())
        row["share_geometric_errors_following_the_mark"] = round(row["errors_vs_geometric_siding_with_mark"] / row["errors_vs_geometric"], 4) if row["errors_vs_geometric"] else None
        out.append(row)
    return out


def side_by_side(pred: np.ndarray, geo: np.ndarray, asr: np.ndarray, mask: np.ndarray, vocab: list[str]) -> dict:
    pg, pa = M.per_relation(pred, geo, mask, vocab), M.per_relation(pred, asr, mask, vocab)
    row = {"macro_f1_geometric": round(M.macro_f1(pg), 4), "macro_f1_asserted": round(M.macro_f1(pa), 4)}
    for r in vocab:
        row.update({f"f1_geometric_{r}": round(pg[r]["f1"], 4), f"f1_asserted_{r}": round(pa[r]["f1"], 4),
                    f"support_geometric_{r}": pg[r]["support"], f"support_asserted_{r}": pa[r]["support"]})
    return row


def pooled(runs: Path, members: list[str], unit: str, qid: np.ndarray, val_qid: np.ndarray, vocab: list[str]
           ) -> tuple[dict | None, list[str]]:
    parts, used = defaultdict(list), []
    for run in members:
        p = load_predictions(runs, run, unit, qid)
        if p is None:
            continue
        th = run_thresholds(runs, run, vocab, val_qid)
        parts["pred"].append(p["probs"] >= th)
        parts["probs"].append(p["probs"])
        for k in ("geo", "asr", "mask"):
            parts[k].append(p[k])
        used.append(run)
    if not used:
        return None, used
    return {k: np.concatenate(v) for k, v in parts.items()}, used


def analyse(runs: Path, latest: dict, experiments: list[str], units: dict, val_qid: np.ndarray, vocab: list[str]
            ) -> tuple[list[dict], list[dict], list[str]]:
    """units: unit -> {"qid", "nts" (per query bool)}."""
    table, disagreement, problems = [], [], []
    for exp in experiments:
        members = sorted(r for r in latest if split_name(r)[0] == exp)
        if not members:
            problems.append(f"{exp}: no runs")
            continue
        for u, info in units.items():
            try:
                p, used = pooled(runs, members, u, info["qid"], val_qid, vocab)
            except (ValueError, FileNotFoundError) as ex:
                problems.append(str(ex))
                continue
            if p is None:
                continue
            base = {"experiment": exp, "unit": u, "seeds": " ".join(str(split_name(r)[1]) for r in used)}
            table.append({**base, **side_by_side(p["pred"], p["geo"], p["asr"], p["mask"], vocab)})
            nts = np.concatenate([info["nts"]] * len(used)) if info.get("nts") is not None else None
            disagreement += [{**base, **row} for row in disagreement_rows(p["pred"], p["geo"], p["asr"], p["mask"], vocab, nts)]
    return table, disagreement, problems


def pick_examples(rows: list[dict], p: dict, thresholds: dict[str, float], vocab: list[str], n: int = 4
                  ) -> dict[str, list[dict]]:
    """The clearest disagreements: asserted-only with the largest margin / tolerance (most out of
    tolerance), geometric-only with the smallest; one relation at a time."""
    c = cells(p["geo"], p["asr"], p["mask"])
    margins = np.array([[math.nan if m is None else m for m in r["margins"]] for r in rows], float)
    ratio = margins / np.array([thresholds[v] for v in vocab])
    out = {}
    for name, sign in (("asserted_only", -1), ("geometric_only", 1)):
        per_rel = []
        for j in range(len(vocab)):
            idx = np.flatnonzero(c[name][:, j] & np.isfinite(ratio[:, j]))
            per_rel.append(list(idx[np.argsort(sign * ratio[idx, j], kind="stable")]))
        chosen = []
        while len(chosen) < n and any(per_rel):
            for j in range(len(vocab)):
                if per_rel[j] and len(chosen) < n:
                    i = per_rel[j].pop(0)
                    chosen.append({"index": int(i), "relation": vocab[j], "ratio": float(ratio[i, j]),
                                   "prob": float(p["probs"][i, j]), "qid": rows[i]["qid"], "diagram_id": rows[i]["diagram_id"],
                                   "refs": rows[i]["refs"]})
        out[name] = chosen
    return out


def examples_figure(examples: dict[str, list[dict]], diagrams: dict, raw: Path, out: Path, unit: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.visualize import draw_query
    n = max(len(v) for v in examples.values()) or 1
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.6), squeeze=False)
    titles = {"asserted_only": "asserted, not true in the drawing", "geometric_only": "true in the drawing, not asserted"}
    for r, (name, items) in enumerate(examples.items()):
        for k in range(n):
            ax = axes[r][k]
            ax.axis("off")
            if k < len(items):
                it = items[k]
                draw_query(ax, diagrams[it["diagram_id"]], it["refs"], raw,
                           title=f"{titles[name] if k == 0 else ''}\n{it['relation']}: margin {it['ratio']:.1f}x tolerance, model p {it['prob']:.2f}")
                ax.title.set_fontsize(8)
    fig.suptitle(f"{unit}: where the two label sets disagree, and what the model says", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def markdown(table: list[dict], disagreement: list[dict], vocab: list[str], experiment: str) -> str:
    units = list(dict.fromkeys(r["unit"] for r in table if r["experiment"] == experiment))
    lines = [f"## {experiment}: geometric vs asserted labels (seeds pooled, val-tuned thresholds)", "",
             "### Macro-F1 and per-relation F1 against each label set", "",
             "| unit | macro geometric | macro asserted | " + " | ".join(f"{v} g / a" for v in vocab) + " |",
             "|---|---|---|" + "---|" * len(vocab)]
    for r in table:
        if r["experiment"] == experiment:
            lines.append(f"| {r['unit']} | {r['macro_f1_geometric']:.3f} | {r['macro_f1_asserted']:.3f} | " + " | ".join(
                f"{r[f'f1_geometric_{v}']:.3f} / {r[f'f1_asserted_{v}']:.3f}" for v in vocab) + " |")
    lines += ["", "### Label cells and the model's positive rate in each (asserted-only = marked but drawn out of tolerance; "
              "geometric-only = true in the drawing, unmarked)", ""]
    for u in units:
        lines += [f"{u}:", "", "| relation | n | agree + (model +) | agree - (model +) | asserted-only (model +) | geometric-only (model +) | "
                  "errors vs asserted: label's | errors vs geometric: followed the mark |", "|---|---|---|---|---|---|---|---|"]
        for r in disagreement:
            if r["experiment"] == experiment and r["unit"] == u:
                def cell(name):
                    pr = r[f"pos_rate_{name}"]
                    return f"{r[f'n_{name}']} ({'-' if pr is None else f'{pr:.2f}'})"
                sa = r["share_asserted_errors_that_are_label_errors"]
                sg = r["share_geometric_errors_following_the_mark"]
                lines.append(f"| {r['relation']} | {r['n']} | {cell('agree_pos')} | {cell('agree_neg')} | {cell('asserted_only')} | {cell('geometric_only')} | "
                             f"{r['errors_vs_asserted_siding_with_verifier']} of {r['errors_vs_asserted']}" + (f" ({sa:.0%})" if sa is not None else "") + " | "
                             f"{r['errors_vs_geometric_siding_with_mark']} of {r['errors_vs_geometric']}" + (f" ({sg:.0%})" if sg is not None else "") + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS), help="comma-separated; the first one gets the figure")
    ap.add_argument("--examples-unit", default="geometry3k_test")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    runs = Path(args.runs) if args.runs else REPO_ROOT / cfg["paths"]["runs"]
    tables, figs = REPO_ROOT / cfg["paths"]["tables"], REPO_ROOT / cfg["paths"]["figures"]
    vocab = vocabulary(cfg)
    experiments = [e for e in args.experiments.split(",") if e]
    latest = latest_runs(runs)
    units, rows_of, diagrams_of = {}, {}, {}
    for u in TEST_UNITS:
        rows, diagrams = eval_rows(cfg, u)
        rows_of[u], diagrams_of[u] = rows, diagrams
        units[u] = {"qid": np.array([r["qid"] for r in rows]),
                    "nts": np.array([bool(diagrams[r["diagram_id"]].flags.get("not_to_scale")) for r in rows])}
    val_qid = np.array([r["qid"] for r in eval_rows(cfg, "val")[0]])
    table, disagreement, problems = analyse(runs, latest, experiments, units, val_qid, vocab)
    write_csv(table, tables / "asserted_vs_geometric.csv")
    write_csv(disagreement, tables / "label_disagreement.csv")
    md = markdown(table, disagreement, vocab, experiments[0])
    (tables / "asserted_vs_geometric.md").write_text(md)
    print(md)
    # examples from the first experiment on the chosen unit
    members = sorted(r for r in latest if split_name(r)[0] == experiments[0])
    u = args.examples_unit
    p, used = pooled(runs, members[:1], u, units[u]["qid"], val_qid, vocab)
    if p is not None:
        thresholds = {r: margin_threshold(r, cfg["tolerances"]) for r in vocab}
        ex = pick_examples(rows_of[u], p, thresholds, vocab)
        raw = REPO_ROOT / cfg["paths"]["raw"] if "raw" in cfg["paths"] else None
        from data.dataset import QueryDataset
        raw = raw or QueryDataset(cfg, u, train=False).raw
        examples_figure(ex, diagrams_of[u], raw, figs / "audit_examples_model.png", u)
        for name, items in ex.items():
            print(name + ":", "; ".join(f"{it['relation']} on {it['diagram_id']} ({it['ratio']:.1f}x, p {it['prob']:.2f})" for it in items))
    for pr in problems:
        print("WARNING:", pr)
    print(f"-> {tables}/asserted_vs_geometric.{{csv,md}}, label_disagreement.csv, {figs}/audit_examples_model.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
