"""Every measured quantity in the technical report (plan §9), collected into LaTeX macros.

Reads the committed tables in reports/tables/, configs/base.yaml and
reports/tables/methodology_inputs.json, and writes reports/report/numbers.tex (one \\newcommand per
quantity). The report's text uses the macros, never typed digits, so a regenerated table changes
the report. Modelled on scripts/methodology_numbers.py; the macros it shares with the methodology
report are collected by that module so the two cannot drift apart.

    python scripts/report_numbers.py            # writes numbers.tex
    python scripts/report_numbers.py --check    # exit 1 if numbers.tex would change
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

from scripts import methodology_numbers as mn

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "report" / "numbers.tex"

RELATIONS = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
UNITS = {"geometry3k_test": "Id", "pgdp5k_heldout": "Ho", "synth_extrap": "Ex"}
# report-set experiments (plan §0.2 #10 note): square-padded where a checkpoint exists
EXPERIMENTS = {
    "sq_probe": "Probe", "sq_finetune_colab": "Ft", "sq_scaling_1k": "SynOneK",
    "sq_finetune_synth10k": "SynTenK", "sq_scaling_50k": "SynFiftyK", "sq_ablation_dinov2": "Dinov",
    "sq_ablation_nooverlay": "NoOverlay", "sq_selftrain_confidence": "StConf",
    "sq_selftrain_consistency": "StCons", "sq_selftrain_control": "StControl",
    "finetune": "FtBatch", "ablation_res224": "ResLow", "ablation_noaug": "NoAug",
}
POINT_GROUPS = {"<=4": "UpToFour", "5-6": "FiveSix", "7-8": "SevenEight", "9-12": "NineTwelve",
                "13-16": "ThirteenSixteen", "17-20": "SeventeenTwenty"}
MARGIN_BINS = {"[0, 0.25)": "A", "[0.25, 0.5)": "B", "[0.5, 1)": "C", "[1, 1.5)": "D",
               "[1.5, 2)": "E", "[2, 4)": "F", "[4, inf)": "G"}

rows, by, macro_name, fmt_int, fmt = mn.rows, mn.by, mn.macro_name, mn.fmt_int, mn.fmt


def pct(x, nd: int = 0) -> str:
    return fmt(100 * float(x), nd)


def collect() -> dict[str, str]:
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    m: dict[str, str] = dict(mn.collect())          # dataset, verifier, config macros

    # ---- Phase 8 summary: macro-F1 (tuned and 0.5), CIs, sd, ECE, per-relation F1
    p8 = rows("phase8_summary")
    for r in p8:
        e = EXPERIMENTS.get(r["experiment"])
        if e is None or r["unit"] not in UNITS:
            continue
        u = UNITS[r["unit"]]
        lab = {"geometric": "", "asserted": "Ass"}[r["label_set"]]
        op = {"tuned": "", "0.5": "Half"}[r["operating"]]
        key = f"f{e}{u}{lab}{op}"
        m[key] = fmt(r["mean"], 3)
        m[key + "Lo"], m[key + "Hi"] = fmt(r["ci_lo"], 3), fmt(r["ci_hi"], 3)
        if r["sd"]:
            m[key + "Sd"] = fmt(r["sd"], 3)
        if lab == "" and op == "":
            m[f"ci{e}{u}"] = f"{m[key]} [{m[key + 'Lo']}--{m[key + 'Hi']}]"
            m[f"ece{e}{u}"] = fmt(r["ece_mean"], 3)
            m[f"nDiagrams{u}"] = fmt_int(r["n_diagrams"])
            m[f"nSeeds{e}"] = r["n_seeds"]
            for rel in RELATIONS:
                if r[f"f1_{rel}"]:
                    m[f"f{e}{u}{macro_name(rel)}"] = fmt(r[f"f1_{rel}"], 3)
    m["thresholdFixed"] = next(r["operating"] for r in p8 if r["operating"] != "tuned")   # Phase 7's operating point
    # derived differences the text states
    m["gapAssertedId"] = fmt(float(m["fFtId"]) - float(m["fFtIdAss"]), 3)
    m["gapAssertedHo"] = fmt(float(m["fFtHo"]) - float(m["fFtHoAss"]), 3)
    m["gainControlOverConsHo"] = fmt(float(m["fStControlHo"]) - float(m["fStConsHo"]), 3)
    m["gainControlOverConsId"] = fmt(float(m["fStControlId"]) - float(m["fStConsId"]), 3)
    m["gainConsOverConfHo"] = fmt(float(m["fStConsHo"]) - float(m["fStConfHo"]), 3)
    m["gainConsOverConfId"] = fmt(float(m["fStConsId"]) - float(m["fStConfId"]), 3)
    m["gainSynTenKEx"] = fmt(float(m["fSynTenKEx"]) - float(m["fFtEx"]), 3)
    m["gainSynTenKHo"] = fmt(float(m["fSynTenKHo"]) - float(m["fFtHo"]), 3)
    m["gainSynOneKEx"] = fmt(float(m["fSynOneKEx"]) - float(m["fFtEx"]), 3)
    m["gainDinovId"] = fmt(float(m["fDinovId"]) - float(m["fFtId"]), 3)
    m["gainResId"] = fmt(float(m["fFtBatchIdHalf"]) - float(m["fResLowIdHalf"]), 3)
    m["gainResEx"] = fmt(float(m["fFtBatchExHalf"]) - float(m["fResLowExHalf"]), 3)
    m["gainAugEx"] = fmt(float(m["fFtBatchExHalf"]) - float(m["fNoAugExHalf"]), 3)

    # ---- padding effect (appendix): decisions changed and F1 change per experiment
    pe = rows("padding_effect")
    for exp, e in EXPERIMENTS.items():
        base = exp.removeprefix("sq_")
        for unit, u in UNITS.items():
            sel = [r for r in pe if r["experiment"] == base and r["unit"] == unit]
            if not sel:
                continue
            m[f"padChanged{e}{u}"] = pct(sum(float(r["share_changed"]) for r in sel) / len(sel), 1)
            m[f"padDelta{e}{u}"] = f"{sum(float(r['geometric_change']) for r in sel) / len(sel):+.3f}"
    real = [r for r in pe if r["unit"] in ("geometry3k_test", "pgdp5k_heldout") and r["experiment"] != "probe"]
    m["padChangedMinPct"] = pct(min(float(r["share_changed"]) for r in real), 1)
    m["padChangedMaxPct"] = pct(max(float(r["share_changed"]) for r in real), 1)
    m["padGainMin"] = f"{min(float(r['geometric_change']) for r in real if r['experiment'] != 'ablation_nooverlay'):+.3f}"
    m["padGainMax"] = f"{max(float(r['geometric_change']) for r in real):+.3f}"

    # ---- headline table (8.2)
    for r in rows("headline_table"):
        if r["seed"] != "mean" or r["unit"] not in UNITS:
            continue
        u = UNITS[r["unit"]]
        tag = "" if r["level"] == "query" else macro_name(r["relation"])
        for cell in ("both", "probe_only", "finetune_only", "neither"):
            m[f"hl{macro_name(cell)}{u}{tag}"] = pct(r[f"share_{cell}"], 1)
        if r["level"] == "query":
            m[f"nQueries{u}"] = fmt_int(r["n"])

    # ---- asserted vs geometric (8.3): label cells and error attribution for 7.2
    for r in rows("label_disagreement"):
        if r["experiment"] != "sq_finetune_colab" or r["unit"] not in UNITS:
            continue
        u, rel = UNITS[r["unit"]], macro_name(r["relation"]) if r["relation"] != "all" else ""
        m[f"nGeomOnly{u}{rel}"] = fmt_int(r["n_geometric_only"])
        m[f"nAssOnly{u}{rel}"] = fmt_int(r["n_asserted_only"])
        if r["pos_rate_geometric_only"]:
            m[f"posRateGeomOnly{u}{rel}"] = pct(r["pos_rate_geometric_only"])
        m[f"shareAssErrorsLabels{u}{rel}"] = pct(r["share_asserted_errors_that_are_label_errors"])
        m[f"shareGeomErrorsMark{u}{rel}"] = pct(r["share_geometric_errors_following_the_mark"])
        m[f"nErrorsGeom{u}{rel}"] = fmt_int(r["errors_vs_geometric"])
        m[f"nErrorsGeomMark{u}{rel}"] = fmt_int(r["errors_vs_geometric_siding_with_mark"])
        if r["relation"] == "all":
            m[f"pctGeomOnlyOfApplicable{u}"] = pct(int(r["n_geometric_only"]) / int(r["n"]))

    # ---- error analysis (8.4): segment-length and point breakdowns, margin bins, for 7.2
    for r in rows("error_analysis"):
        if r["experiment"] != "sq_finetune_colab" or r["unit"] not in UNITS:
            continue
        u = UNITS[r["unit"]]
        if r["breakdown"] == "segment" and r["group"] != "no segment":
            g = {"<0.10": "Short", "0.10-0.20": "Mid", "0.20-0.35": "Long", ">=0.35": "Longest"}[r["group"]]
            m[f"fSeg{g}{u}"] = fmt(r["macro_f1"], 2)
            for rel in ("collinear", "equal_length"):
                if r[f"f1_{rel}"]:
                    m[f"fSeg{g}{u}{macro_name(rel)}"] = fmt(r[f"f1_{rel}"], 2)
        if r["breakdown"] == "not_to_scale":
            m[f"fNts{macro_name(r['group'].replace(' ', '_'))}{u}"] = fmt(r["macro_f1"], 3)
            m[f"nNts{macro_name(r['group'].replace(' ', '_'))}{u}"] = fmt_int(r["n_queries"])
        if r["breakdown"] == "points":
            m[f"fPts{POINT_GROUPS[r['group']]}{u}"] = fmt(r["macro_f1"], 2)
    for r in rows("error_margin"):
        if r["experiment"] != "sq_finetune_colab" or r["unit"] not in UNITS or not r["accuracy"]:
            continue
        m[f"acc{macro_name(r['relation'])}{UNITS[r['unit']]}{MARGIN_BINS[r['bin']]}"] = fmt(r["accuracy"], 2)
    nts = by(rows("not_to_scale_eval"), "run")
    m["nNtsTestDiagrams"] = fmt_int(nts["finetune_colab_s0"]["n_diagrams"])
    m["nNtsTestQueries"] = fmt_int(nts["finetune_colab_s0"]["n_queries"])

    # ---- attention (8.5)
    att = {(r["map"], r["kind"]): r for r in rows("attention_summary")}
    a = att[("last_layer", "all")]
    m["attHighlightPct"] = pct(a["share_highlight"])
    m["attHighlightAreaPct"] = pct(a["area_highlight"])
    m["attHighlightRatio"] = fmt(a["ratio_highlight"], 0)
    m["attInkPct"] = pct(a["share_ink"])
    m["attBlankPct"] = pct(a["share_blank"])
    m["attPaddingPct"] = pct(a["share_padding"])
    m["attPaddingAreaPct"] = pct(a["area_padding"])
    m["attCorrectPct"] = pct(att[("last_layer", "correct")]["share_highlight"])
    m["attWrongPct"] = pct(att[("last_layer", "wrong")]["share_highlight"])
    m["attRolloutHighlightPct"] = pct(att[("rollout", "all")]["share_highlight"])
    m["nAttentionQueries"] = a["n"]

    # ---- self-training pseudo-labels (7.11)
    st = {(r["kept_by"], r["relation"]): r for r in rows("selftrain_pseudo_labels")}
    m["stPrecisionConf"] = fmt(st[("confidence", "all")]["pos_precision"], 3)
    m["stPrecisionCons"] = fmt(st[("consistency", "all")]["pos_precision"], 3)
    m["stPrecisionRemoved"] = fmt(st[("removed_by_consistency", "all")]["pos_precision"], 3)
    m["stPrecisionConfEqualLength"] = fmt(st[("confidence", "equal_length")]["pos_precision"], 2)
    m["stPoolDiagrams"] = fmt_int(st[("confidence", "all")]["pool_diagrams"])
    m["stContradictionDiagrams"] = fmt_int(st[("confidence", "all")]["contradiction_diagrams"])
    m["stTruthContradictionPct"] = pct(st[("confidence", "all")]["truth_contradiction_rate"], 1)
    m["stPseudoPosConf"] = fmt_int(st[("confidence", "all")]["pseudo_pos"])

    # ---- relation census for the audit table
    census = by(rows("relation_census"), "relation")
    m["nTestFiguresCollinearTrain"] = fmt_int(census["collinear"]["geometry3k_figures_train"])
    return m


def render(m: dict[str, str]) -> str:
    lines = ["% GENERATED by scripts/report_numbers.py from reports/tables, configs/base.yaml and",
             "% reports/tables/methodology_inputs.json. Do not edit; rerun the script.", ""]
    for k in sorted(m):
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if numbers.tex would change")
    args = ap.parse_args(argv)
    text = render(collect())
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print("numbers.tex is", "up to date" if same else "STALE")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({text.count(chr(10)) - 3} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
