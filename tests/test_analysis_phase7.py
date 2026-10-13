"""analysis/phase7.py on fake run records (no GPU): tables, not-to-scale alignment, scaling figure."""

import csv
import json

import numpy as np
import pytest

from analysis import phase7
from experiments.run import REPO_ROOT, load_config

needs_data = pytest.mark.skipif(not (REPO_ROOT / "data/processed/index/geometry3k_test.jsonl").exists(),
                                reason="query indexes not built")
VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]


def _metrics(score):
    res = {"geometric": {"macro_f1": score, "f1": {r: score for r in VOCAB}},
           "asserted": {"macro_f1": score / 2, "f1": {r: score / 2 for r in VOCAB}}, "n": 10}
    return {"status": "ok", "backbone_loaded": "x", "results": {u: res for u in phase7.UNITS}}


def _runs(tmp_path, cfg=None):
    runs = tmp_path / "runs"
    runs.mkdir()
    rows = []
    for run, score in (("finetune_s0", 0.8), ("finetune_s1", 0.9), ("scaling_50k_s0", 0.95), ("smoke_finetune_s0", 0.1),
                       ("selftrain_consistency_s0", 0.85)):
        d = runs / run
        d.mkdir()
        m = _metrics(score)
        if run.startswith("selftrain"):
            stats = {"entries": 5, "pos": 2, "pos_precision": 0.5, "accuracy": 0.8,
                     **{r: {"pos": 1, "pos_precision": 1.0, "neg": 1, "neg_accuracy": 1.0} for r in VOCAB}}
            m.update(teacher="runs/finetune_synth10k_s2/checkpoints/best.pt", pseudo_labels={
                "tau": 0.9, "pool_diagrams": 10, "contradiction_diagrams": 1, "truth_contradiction_rate": 0.0,
                "confidence": stats, "consistency": stats, "removed_by_consistency": stats})
        (d / "metrics.json").write_text(json.dumps(m))
        if cfg is not None and run == "finetune_s0":
            (d / "predictions").mkdir()
            for u in phase7.NTS_UNITS:
                r, _ = phase7.eval_rows(cfg, u)
                geo = np.array([x["geo"] for x in r], float)
                np.savez(d / "predictions" / f"{u}.npz", qid=np.array([x["qid"] for x in r]), probs=geo, geo=geo,
                         asr=np.array([x["asr"] for x in r], float), mask=np.array([x["mask"] for x in r], float))
        rows.append({"timestamp": "t", "name": run, "mode": "finetune", "seed": run[-1], "config_hash": "h",
                     "status": "ok", "best_epoch": 2, "duration_s": 1, "metrics": "{}"})
    with open(runs / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return runs


def test_tables_and_figure(tmp_path):
    runs = _runs(tmp_path)
    latest = phase7.latest_runs(runs)
    assert "smoke_finetune_s0" not in latest and len(latest) == 4
    rows = phase7.per_run_table(runs, latest, VOCAB)
    summary = phase7.summarise(rows)
    ft = next(r for r in summary if r["experiment"] == "finetune" and r["unit"] == "pgdp5k_heldout"
              and r["label_set"] == "geometric")
    assert ft["n_seeds"] == 2 and ft["mean"] == pytest.approx(0.85) and ft["sd"] == pytest.approx(0.0707, abs=1e-3)
    md = phase7.markdown(summary)
    assert "| finetune |" in md and "0.850 ± 0.071" in md
    pts = phase7.scaling_figure(rows, tmp_path / "scaling.png")
    assert (tmp_path / "scaling.png").exists() and {p["synthetic"] for p in pts} == {0, 50000}
    pl = phase7.pseudo_label_table(runs, latest, VOCAB)
    assert len(pl) == 3 * (len(VOCAB) + 1) and pl[0]["teacher"] == "finetune_synth10k_s2"


@needs_data
def test_not_to_scale_alignment(tmp_path):
    cfg = load_config(REPO_ROOT / "configs/base.yaml")
    runs = _runs(tmp_path, cfg)
    nts, problems = phase7.not_to_scale(runs, {"finetune_s0": {}}, cfg, VOCAB)
    assert not problems and len(nts) == 2 and nts[0]["n_diagrams"] == 7
    assert nts[0]["macro_f1"] == pytest.approx(1.0)                           # probs = geometric labels
    np.savez(runs / "finetune_s0/predictions/pgdp5k_heldout.npz", qid=np.array(["x"]), probs=np.zeros((1, 6)),
             geo=np.zeros((1, 6)), asr=np.zeros((1, 6)), mask=np.ones((1, 6)))
    nts, problems = phase7.not_to_scale(runs, {"finetune_s0": {}}, cfg, VOCAB)
    assert not nts and problems                                                 # stale predictions reported
