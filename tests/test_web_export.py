"""web/export.py on fake runs, tables and split manifests (no GPU, no data, no hub)."""

import csv
import json

import pytest
import yaml

from web import export
from experiments.run import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")
VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
UNITS = ["val", "geometry3k_test", "pgdp5k_heldout", "synth_extrap"]


def _metrics(score, n=10):
    res = {"geometric": {"macro_f1": score, "f1": {r: score for r in VOCAB}},
           "asserted": {"macro_f1": score / 2, "f1": {r: score / 2 for r in VOCAB}}, "n": n}
    return {"status": "ok", "backbone_loaded": "fake/bb", "results": {u: res for u in UNITS}}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp_path):
    runs = tmp_path / "runs"
    log = []
    for run, score in (("probe_s0", 0.6), ("finetune_s0", 0.8), ("finetune_s1", 0.9), ("scaling_1k_s0", 0.88),
                       ("ablation_nooverlay_s0", 0.43), ("smoke_finetune_s0", 0.1), ("selftrain_control_s0", 0.91)):
        d = runs / run
        d.mkdir(parents=True)
        m = _metrics(score)
        if run.startswith("selftrain"):
            stats = {"entries": 5, "pos": 2, "pos_precision": 0.5, "accuracy": 0.8,
                     **{r: {"pos": 1, "pos_precision": 1.0, "neg": 1, "neg_accuracy": 1.0} for r in VOCAB}}
            m.update(teacher="runs/finetune_synth10k_s2/checkpoints/best.pt", pseudo_labels={
                "tau": 0.9, "consistency_negatives": True, "pool_diagrams": 10, "contradiction_diagrams": 1,
                "truth_contradiction_rate": 0.1, "confidence": stats, "consistency": stats,
                "removed_by_consistency": stats})
        (d / "metrics.json").write_text(json.dumps(m))
        log.append({"timestamp": "t", "name": run, "mode": "finetune", "seed": run[-1], "config_hash": "h",
                    "status": "ok", "best_epoch": 2, "duration_s": 1, "metrics": "{}"})
    _write_csv(runs / "log.csv", log)
    (runs / "benchmark_lab").mkdir()
    (runs / "benchmark_lab" / "benchmark.json").write_text(json.dumps(
        {"gpu": "G", "n_gpus": 4, "rates": {"448": {"train_img_per_s": 22.5}}, "total_gpu_hours": 1420.6,
         "wall_hours": 452.8}))
    tables = tmp_path / "tables"
    _write_csv(tables / "not_to_scale_eval.csv", [
        {"run": "finetune_s0", "experiment": "finetune", "seed": 0, "label_set": "geometric", "macro_f1": 0.8,
         "n_queries": 353, "n_diagrams": 7},
        {"run": "finetune_s1", "experiment": "finetune", "seed": 1, "label_set": "geometric", "macro_f1": 0.9,
         "n_queries": 353, "n_diagrams": 7}])
    _write_csv(tables / "audit_geometry3k_figures.csv", [
        {"relation": "parallel", "candidates": 100, "pct_geometric_unmarked": 89.0}])
    _write_csv(tables / "highlight_colour_check.csv", [
        {"test": "near_highlight_r80", "images": 600, "images_with": 0, "pct": 0.0, "min_pixels": 20}])
    _write_csv(tables / "verifier_audit.csv", [
        {"evaluation": "round2_adjudicated_weighted", "relation": "ALL", "n": 60, "fp_rate": "0.000", "fn_rate": "0.000"},
        {"evaluation": "round2", "relation": "parallel", "n": 5, "fp_rate": "0.2", "fn_rate": "0"}])
    processed = tmp_path / "processed"
    (processed / "splits").mkdir(parents=True)
    for name, n in (("train", 13), ("val", 2), ("test", 4), ("pgdp5k_heldout", 6)):
        (processed / "splits" / f"{name}.txt").write_text("".join(f"x{i}\n" for i in range(n)))
    return runs, tables, processed


def _fake_backbone(name, L):
    return {"name": name, "available": True, "patch": 16, "tokens": (L // 16) ** 2 + 5, "params_M": 1.0}


def test_export_from_fake_runs(tmp_path):
    runs, tables, processed = _fixture(tmp_path)
    site = yaml.safe_load(export.SITE_YAML.read_text())
    out = tmp_path / "data"
    docs = export.export(out, runs, tables, processed, CFG, site, _fake_backbone, lambda: {"hero": {"x": 1}})
    assert {p.name for p in out.iterdir()} == {"results.json", "scaling.json", "pseudo_labels.json", "facts.json",
                                              "figures.json"}
    for p in out.iterdir():
        json.loads(p.read_text())                                          # valid JSON, no NaN

    r = docs["results"]
    assert "smoke_finetune" not in r["experiments"] and r["runs"][0] == "ablation_nooverlay_s0"
    ft = r["experiments"]["finetune"]["units"]["geometry3k_test"]["geometric"]
    assert ft["n_seeds"] == 2 and ft["mean"] == pytest.approx(0.85) and ft["sd"] == pytest.approx(0.0707, abs=1e-4)
    assert ft["values"] == [0.8, 0.9] and ft["per_relation"]["parallel"]["mean"] == pytest.approx(0.85)
    assert r["experiments"]["probe"]["units"]["val"]["asserted"]["mean"] == pytest.approx(0.3)
    assert r["experiments"]["ablation_nooverlay"]["units"]["val"]["geometric"]["sd"] is None   # one seed
    assert list(r["experiments"])[:2] == ["probe", "finetune"]                                  # plan order
    assert r["not_to_scale"]["finetune"]["geometric"]["n_diagrams"] == 7
    assert r["units"]["synth_extrap"]["n_queries"] == 10

    s = docs["scaling"]
    pts = {(p["unit"], p["synthetic"]): p for p in s["points"]}
    assert pts[("pgdp5k_heldout", 0)]["n_seeds"] == 2 and pts[("pgdp5k_heldout", 1000)]["mean"] == pytest.approx(0.88)
    assert s["budget"]["total_images"] == CFG["finetune"]["epochs"] * CFG["finetune"]["epoch_images"]

    pl = docs["pseudo_labels"]
    assert pl["available"] and pl["teacher"] == "finetune_synth10k_s2" and pl["filters"]["consistency"]["all"]["accuracy"] == 0.8

    assert docs["figures"] == {"hero": {"x": 1}}
    f = docs["facts"]
    assert f["meta"]["author"] == site["author"] and f["data"]["geometry3k_train_figures"] == 13
    assert f["data"]["synth_extrap_diagrams"] is None                       # missing file -> null, not a crash
    assert f["tolerances"]["parallel_deg"] == CFG["tolerances"]["parallel_deg"]
    assert f["tolerances"]["equal_length_pct"] == pytest.approx(100 * CFG["tolerances"]["equal_length_rel"])
    assert f["overlay"]["color"] == CFG["overlay"]["color"]
    assert f["compute"]["planned_gpu_hours"] == 1420.6 and f["gold"]["round2_adjudicated_weighted"]["fn_rate"] == 0
    assert f["colour_check"]["near_highlight_r80"]["images_with"] == 0 and f["colour_check"]["near_highlight_r80"]["radius"] == 80
    assert f["audit"]["parallel"]["candidates"] == 100
    assert f["backbones"]["main"]["name"] == CFG["model"]["backbone"]
    assert f["backbones"]["demo"]["name"] == "facebook/dinov2-large"        # demo config from site.yaml


def test_missing_backbone_is_reported_not_raised():
    b = export.hf_backbone_facts("nonexistent-org/nonexistent-model", 448, online=False)
    assert b["available"] is False and b["reason"]
