"""Every measured quantity in the methodology report, collected into LaTeX macros.

Reads the committed tables in reports/tables/, configs/base.yaml and the split manifests, and
writes reports/methodology/numbers.tex (one \\newcommand per quantity). The report's text uses
the macros, never typed digits, so a regenerated table changes the report (METHODOLOGY_PLAN.md §4).

Two inputs live under gitignored directories (runs/benchmark_lab/benchmark.json, the query index
and figure clusters under data/processed/). When they are present, their relevant numbers are
copied to reports/tables/methodology_inputs.json; otherwise that committed copy is read, so the
macros regenerate identically from a fresh clone.

    python scripts/methodology_numbers.py            # writes numbers.tex (+ refreshes inputs)
    python scripts/methodology_numbers.py --check    # exit 1 if numbers.tex would change
"""

from __future__ import annotations

import argparse
import csv
import re
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "methodology" / "numbers.tex"
INPUTS = TABLES / "methodology_inputs.json"

RELATIONS = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
EXCLUDED = ["midpoint", "equal_angle", "perp_bisector"]
EVAL_UNITS = ["geometry3k_test", "pgdp5k_heldout", "synth_extrap", "pgps9k_pool"]


def rows(name: str) -> list[dict]:
    with open(TABLES / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def by(table: list[dict], key: str) -> dict[str, dict]:
    return {r[key]: r for r in table}


def macro_name(s: str) -> str:
    """'equal_length' -> 'EqualLength', 'geometry3k_test' -> 'GeometryTest'."""
    words = re.sub(r"\d+k?", "", s.replace("-", "_")).split("_")      # macro names allow letters only
    return "".join(w[:1].upper() + w[1:] for w in words)


def fmt_int(x) -> str:
    return f"{int(round(float(x))):,}"


def fmt(x, nd: int) -> str:
    return f"{float(x):.{nd}f}"


# ----------------------------------------------------------------------------- gitignored inputs

def refresh_inputs() -> dict:
    """Copy the numbers that come from gitignored files into reports/tables/methodology_inputs.json."""
    data = json.loads(INPUTS.read_text()) if INPUTS.exists() else {}
    bench = ROOT / "runs" / "benchmark_lab" / "benchmark.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        jobs = {j["run"]: j for j in b["jobs"]}
        data["benchmark"] = {"gpu": b["gpu"], "n_gpus": b["n_gpus"], "rates": b["rates"],
                             "train_queries": jobs["finetune_s0"]["train_queries"]}
    index = ROOT / "data" / "processed" / "index"
    if index.exists():
        sizes = {}
        for unit in EVAL_UNITS:
            p = index / f"{unit}.jsonl"
            if p.exists():
                with open(p) as f:
                    sizes[unit] = sum(1 for _ in f)
        # merge, never drop: some indexes (the Part B pool) are built only by an experiment, so a
        # fresh clone lacks them and must keep the committed sizes (fresh-clone test, 2026-10-25)
        data["index_sizes"] = {**data.get("index_sizes", {}), **sizes}
    clusters = ROOT / "data" / "processed" / "figure_clusters.csv"
    if clusters.exists():
        with open(clusters, newline="") as f:
            cl = list(csv.DictReader(f))
        g3k = [r for r in cl if r["source"] == "geometry3k"]
        in_trainval = {r["cluster"] for r in g3k if r["split"] in ("train", "val")}
        test = [r for r in g3k if r["split"] == "test"]
        data["geometry3k_test_overlap"] = {"test_images": len(test),
                                           "share_figure_with_trainval": sum(r["cluster"] in in_trainval for r in test)}
    if data:
        INPUTS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


# ----------------------------------------------------------------------------- collection

def collect() -> dict[str, str]:
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    inputs = refresh_inputs()
    m: dict[str, str] = {}

    # Sources and splits (relation census, figure-level columns)
    census = by(rows("relation_census"), "relation")
    d = census["_diagrams"]
    m["nGeometryProblems"] = fmt_int(d["geometry3k_total"])
    m["nGeometryFigures"] = fmt_int(d["geometry3k_figures_total"])
    m["nPgdpDiagrams"] = fmt_int(d["pgdp5k_total"])
    m["nPgpsDiagrams"] = fmt_int(d["pgps9k_all"])
    m["nGeometryTrainProblems"] = fmt_int(d["geometry3k_train"])
    m["nGeometryValProblems"] = fmt_int(d["geometry3k_val"])
    m["nGeometryTestProblems"] = fmt_int(d["geometry3k_test"])
    m["nPgdpTrain"], m["nPgdpVal"], m["nPgdpTest"] = (fmt_int(d[k]) for k in ("pgdp5k_train", "pgdp5k_val", "pgdp5k_test"))
    for split in ("train", "val", "test"):
        m[f"nFigures{macro_name(split)}"] = fmt_int(d[f"geometry3k_figures_{split}"])
    m["nHeldoutFigures"] = fmt_int(d["pgdp5k_heldout_test"])
    m["nPoolFigures"] = fmt_int(d["pgps9k_pool_all"])
    for r in RELATIONS:
        m[f"nTrain{macro_name(r)}"] = fmt_int(census[r]["geometry3k_figures_train"])
    for r in EXCLUDED:
        m[f"nTrain{macro_name(r)}"] = fmt_int(census[r]["geometry3k_figures_train"])
    m["nTangentAssertedAll"] = fmt_int(sum(int(census["tangent"][k]) for k in
                                           ("geometry3k_total", "pgdp5k_total", "pgps9k_all")))
    m["tierTwoMinPositives"] = fmt_int(cfg["relations"]["tier2_min_positives"])
    for split, n in (("train", 70), ("val", 10), ("test", 20)):
        m[f"splitPct{macro_name(split)}"] = str(int(cfg["data"]["split_fractions"][["train", "val", "test"].index(split)] * 100))

    # Overlap
    ov = {(r["source"], r["split"], r["duplicated_in_source"], r["duplicated_in_split"]): r for r in rows("source_overlap")}
    m["nGeometryTestImages"] = fmt_int(ov[("geometry3k", "test", "geometry3k", "train")]["n_images"])
    m["nGeometryTestDupTrain"] = fmt_int(ov[("geometry3k", "test", "geometry3k", "train")]["n_duplicated"])
    m["pctGeometryTestDupTrain"] = fmt(ov[("geometry3k", "test", "geometry3k", "train")]["pct"], 0)
    m["nPgpsImagesGeometryTest"] = fmt_int(ov[("pgps9k", "all", "geometry3k", "test")]["n_duplicated"])
    m["nPgpsImagesPgdpTest"] = fmt_int(ov[("pgps9k", "all", "pgdp5k", "test")]["n_duplicated"])
    tv = inputs.get("geometry3k_test_overlap")
    if tv:
        m["nGeometryTestShareFigure"] = fmt_int(tv["share_figure_with_trainval"])
        m["pctGeometryTestShareFigure"] = fmt(100 * tv["share_figure_with_trainval"] / tv["test_images"], 0)

    # Audit (curated Geometry3K figures and the held-out source)
    for name, tag in (("audit_geometry3k_figures", "G"), ("audit_pgdp5k", "P")):
        a = by(rows(name), "relation")
        for r in RELATIONS:
            m[f"pctConfirmed{tag}{macro_name(r)}"] = fmt(a[r]["pct_asserted_confirmed"], 1)
            m[f"pctUnmarked{tag}{macro_name(r)}"] = fmt(a[r]["pct_geometric_unmarked"], 1)
            m[f"nUnmarked{tag}{macro_name(r)}"] = fmt_int(a[r]["geometric_not_asserted"])

    # Coordinates
    cs = by(rows("coordinate_sanity"), "source")
    m["pctGeometryPointsOutside"] = fmt(cs["geometry3k"]["pct_points_outside"], 2)
    pairs = rows("g3k_vs_pgdp5k_coords")
    med = sorted(float(r["median_px"]) for r in pairs)
    m["nCoordPairs"] = fmt_int(len(pairs))
    m["medianCoordErrorPx"] = fmt(med[len(med) // 2], 1)
    m["pctCoordPairsOverTen"] = fmt(100 * sum(float(r["max_px"]) > 10 for r in pairs) / len(pairs), 0)

    # Tolerances (fixed values) and calibration (where they came from)
    tol = cfg["tolerances"]
    m["tolParallelDeg"] = fmt(tol["parallel_deg"], 1)
    m["tolPerpendicularDeg"] = fmt(tol["perpendicular_deg"], 1)
    m["tolEqualLengthPct"] = fmt(100 * tol["equal_length_rel"], 0)
    m["tolOnCirclePct"] = fmt(100 * tol["on_circle_rel"], 2)
    m["tolTangentPct"] = fmt(100 * tol["tangent_rel"], 2)
    m["tolCollinearPct"] = fmt(100 * tol["collinear_rel"], 2)
    m["ntsAngleDeg"] = fmt(cfg["not_to_scale"]["angle_deg"], 0)
    m["ntsLengthPct"] = fmt(100 * cfg["not_to_scale"]["rel_length"], 0)
    cal = by(rows("tolerance_calibration"), "relation")
    m["valleyParallelDeg"] = fmt(cal["parallel"]["tolerance"], 2)
    m["valleyPerpendicularDeg"] = fmt(cal["perpendicular"]["tolerance"], 2)
    m["kneeEqualLengthPct"] = fmt(100 * float(cal["equal_length"]["tolerance"]), 2)

    # Gold set
    g = {(r["evaluation"], r["relation"]): r for r in rows("verifier_audit")}
    m["nGoldRoundOne"] = fmt_int(g[("round1", "ALL")]["n"])
    m["nGoldRoundTwo"] = fmt_int(g[("round2", "ALL")]["n"])
    m["pctGoldRoundOneFP"] = fmt(100 * float(g[("round1", "ALL")]["fp_rate"]), 1)
    m["pctGoldRoundOneFN"] = fmt(100 * float(g[("round1", "ALL")]["fn_rate"]), 1)
    m["pctGoldRoundTwoWeightedFP"] = fmt(100 * float(g[("round2_weighted", "ALL")]["fp_rate"]), 1)
    m["pctGoldRoundTwoWeightedFN"] = fmt(100 * float(g[("round2_weighted", "ALL")]["fn_rate"]), 1)
    adj = g.get(("round2_adjudicated_weighted", "ALL"))
    if adj:
        m["pctGoldRoundTwoAdjFP"] = fmt(100 * float(adj["fp_rate"]), 1)
        m["pctGoldRoundTwoAdjFN"] = fmt(100 * float(adj["fn_rate"]), 1)
    m["pctGoldRoundTwoAdjStressFP"] = fmt(100 * float(g[("round2_adjudicated", "ALL")]["fp_rate"]), 1)
    m["pctGoldRoundTwoAdjStressFN"] = fmt(100 * float(g[("round2_adjudicated", "ALL")]["fn_rate"]), 1)
    m["goldCeilingPct"] = "3"
    with open(ROOT / "verifier" / "gold" / "adjudication.csv", newline="") as f:
        m["nGoldAdjudicated"] = fmt_int(len({r["gold_id"] for r in csv.DictReader(f) if r["gold_id"].startswith("h")}))

    # Not to scale
    nts = by(rows("not_to_scale"), "set")
    for s in ("geometry3k", "geometry3k_figures", "pgdp5k", "pgps9k", "pgdp5k_heldout"):
        m[f"pctNts{macro_name(s)}"] = fmt(nts[s]["pct"], 1)

    # Synthetic
    syn = cfg["synthetic"]
    m["synthPointsMin"], m["synthPointsMax"] = (str(x) for x in syn["n_points"])
    m["synthSegmentsMin"], m["synthSegmentsMax"] = (str(x) for x in syn["n_segments"])
    m["synthCirclesMax"] = str(syn["n_circles"][1])
    m["synthCanvasPx"] = str(syn["canvas_px"])
    m["synthInjectRate"] = fmt(syn["inject_rate"], 1)
    m["synthJitterPx"] = fmt(syn["jitter_px"], 1)
    m["synthRotationDeg"] = str(syn["render"]["rotation_deg"])
    m["synthMinSeparationPx"] = str(syn["min_separation_px"])
    m["synthExtrapPointsMin"], m["synthExtrapPointsMax"] = (str(x) for x in syn["extrap_points"])
    m["synthExtrapSize"] = fmt_int(syn["extrap_size"])
    m["synthUnlabelledSize"] = fmt_int(syn["unlabelled_size"])
    m["synthPoolMax"] = fmt_int(syn["pool_sizes"][-1])
    m["scalingSizes"] = ", ".join(fmt_int(x) for x in cfg["scaling"]["sizes"])
    chk = [r for r in rows("synthetic_verifier_check") if r["pool"] == "synth_train"]
    m["synthInjectedPassMinPct"] = fmt(100 * min(float(r["injected_pass"]) for r in chk), 1)
    m["synthRandomPassMaxPct"] = fmt(100 * max(float(r["random_pass"]) for r in chk if r["relation"] != "equal_length"), 1)
    m["synthRandomPassEqualLengthPct"] = fmt(100 * float(by(chk, "relation")["equal_length"]["random_pass"]), 1)

    # Queries, overlay, negatives, augmentation
    m["queryCap"] = str(cfg["data"]["max_queries_per_diagram"])
    m["noneRatio"] = fmt(cfg["data"]["train_none_ratio"], 0)
    ovl = cfg["overlay"]
    m["overlayRGB"] = ", ".join(str(c) for c in ovl["color"])
    m["overlayWidthPx"] = str(ovl["width_px"])
    m["overlayHaloPx"] = str(ovl["halo_width_px"])
    col = by(rows("highlight_colour_check"), "test")
    m["nColourImages"] = fmt_int(col["hue_green"]["images"])
    m["nColourNearHighlight"] = fmt_int(col["near_highlight_r80"]["images_with"])
    m["pctColourGreenFamily"] = fmt(col["hue_green"]["pct"], 1)
    m["pctColourRed"] = fmt(col["hue_red"]["pct"], 0)
    m["pctColourBlue"] = fmt(col["hue_blue"]["pct"], 0)
    m["inputLongSide"] = str(cfg["input"]["long_side"])
    m["inputPadMultiple"] = str(cfg["input"]["pad_to_multiple"])
    aug = cfg["augment"]
    m["augRotationDeg"] = str(aug["rotation_deg"])
    m["augScaleMin"], m["augScaleMax"] = (fmt(x, 1) for x in aug["scale"])
    m["augThickenProb"] = fmt(aug["line_thickness_prob"], 1)
    m["augNoiseStdMax"] = fmt(aug["gaussian_noise_std"][1], 2)
    m["augJpegMin"], m["augJpegMax"] = (str(x) for x in aug["jpeg_quality"])
    m["augContrastMin"], m["augContrastMax"] = (fmt(x, 1) for x in aug["contrast"])
    m["posWeightCap"] = "50"

    # Model and training
    m["backboneId"] = cfg["model"]["backbone"].replace("_", r"\_")
    m["fallbackId"] = cfg["model"]["backbone_fallback"].replace("_", r"\_")
    ft = cfg["finetune"]
    m["ftBatch"] = str(ft["batch_size"])
    m["ftHeadLR"] = f"{ft['head_lr']:.0e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
    m["ftBackboneLR"] = f"{ft['backbone_lr']:.0e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
    m["ftLLRD"] = fmt(ft["llrd"], 1)
    m["ftWeightDecay"] = fmt(ft["weight_decay"], 2)
    m["ftEpochs"] = str(ft["epochs"])
    m["ftEpochImages"] = fmt_int(ft["epoch_images"])
    m["ftTotalImages"] = fmt_int(ft["epochs"] * ft["epoch_images"])
    m["ftRealFractionPct"] = fmt(100 * ft["real_fraction"], 0)
    m["ftWarmupEpochs"] = fmt(ft["warmup_epochs"], 1)
    m["ftLPQueries"] = fmt_int(ft["lp_queries"])
    m["ftLPEpochs"] = str(ft["lp_epochs"])
    m["ftEMA"] = fmt(ft["ema_decay"], 3)
    m["ftAmpDtype"] = ft["amp_dtype"]
    pr = cfg["probe"]
    m["probeLR"] = f"{pr['lr']:.0e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
    m["probeEpochs"] = str(pr["epochs"])
    m["probeBatch"] = str(pr["batch_size"])
    m["probePatience"] = str(pr["early_stopping_patience"])
    m["nSeeds"] = str(len(cfg["seeds"]))
    m["seedList"] = ", ".join(str(s) for s in cfg["seeds"])
    ev = cfg["eval"]
    m["eceBins"] = str(ev["ece_bins"])
    m["bootstrapResamples"] = fmt_int(ev["bootstrap_resamples"])
    m["ciLevelPct"] = fmt(100 * ev["ci_level"], 0)
    m["extrapEvalCap"] = fmt_int(ev["max_queries"]["synth_extrap"])
    st = cfg["selftrain"]
    m["selftrainTau"] = fmt(st["tau"], 1)

    # Gitignored-derived: benchmark, index sizes
    b = inputs.get("benchmark")
    if b:
        m["benchGPU"] = b["gpu"]
        m["benchGPUs"] = str(b["n_gpus"])
        m["benchTrainImgPerS"] = fmt(b["rates"]["448"]["train_img_per_s"], 1)
        m["benchEvalImgPerS"] = fmt(b["rates"]["448"]["eval_img_per_s"], 1)
        m["benchPeakMemGB"] = fmt(b["rates"]["448"]["peak_mem_gb"], 1)
        m["benchTrainImgPerSLow"] = fmt(b["rates"]["224"]["train_img_per_s"], 1)
        m["nTrainQueries"] = fmt_int(b["train_queries"])
        hours = ft["epochs"] * ft["epoch_images"] / b["rates"]["448"]["train_img_per_s"] / 3600
        m["ftHoursPerRun"] = fmt(hours, 1)
    sizes = inputs.get("index_sizes", {})
    for unit, n in sizes.items():
        m[f"nQueries{macro_name(unit)}"] = fmt_int(n)
    return m


def render(m: dict[str, str]) -> str:
    lines = ["% GENERATED by scripts/methodology_numbers.py from reports/tables, configs/base.yaml and",
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
