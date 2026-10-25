"""Phase 8.3 (asserted vs geometric) and 8.6 (figures) on synthetic data; a couple of seconds."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import asserted_vs_geometric as AG, figures as FG

VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
R = len(VOCAB)


def test_cells_and_disagreement_rows():
    geo = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], float).T                   # 4 entries x 2 relations
    asr = np.array([[1, 1, 0, 0], [0, 1, 0, 0]], float).T
    mask = np.ones((4, 2))
    mask[3, 1] = 0
    c = AG.cells(geo, asr, mask)
    assert c["agree_pos"].sum() == 2 and c["asserted_only"].sum() == 1 and c["geometric_only"].sum() == 2
    assert c["agree_neg"].sum() == 2                                        # (0, 1) and (3, 0); (3, 1) is masked
    pred = geo.astype(bool)                                                 # the model follows the verifier exactly
    rows = AG.disagreement_rows(pred, geo, asr, mask, ["a", "b"], nts=np.array([False, True, False, False]))
    assert [r["relation"] for r in rows] == ["a", "b", "all"]
    all_ = rows[-1]
    assert all_["n"] == 7 and all_["n_agree_neg"] == 2 and all_["n_asserted_only"] == 1 and all_["pos_rate_asserted_only"] == 0.0
    assert all_["pos_rate_geometric_only"] == 1.0 and all_["errors_vs_geometric"] == 0
    assert all_["errors_vs_asserted"] == 3 and all_["share_asserted_errors_that_are_label_errors"] == 1.0
    assert rows[0]["asserted_only_not_to_scale"] == 1
    pred2 = asr.astype(bool)                                                # the model follows the marks exactly
    r2 = AG.disagreement_rows(pred2, geo, asr, mask, ["a", "b"])[-1]
    assert r2["errors_vs_asserted"] == 0 and r2["share_geometric_errors_following_the_mark"] == 1.0
    s = AG.side_by_side(pred, geo, asr, mask, ["a", "b"])
    assert s["macro_f1_geometric"] == 1.0 and s["macro_f1_asserted"] < 1.0


def _rows(n_diag=4, per=5):
    rows, diagrams = [], {}
    for d in range(n_diag):
        did = f"g:{d}"
        diagrams[did] = SimpleNamespace(points={"A": (0, 0), "B": (10, 0), "C": (10, 10)}, circles=[], flags={"not_to_scale": d == 0},
                                        meta={}, image_path="x.png")
        for i in range(per):
            rows.append({"qid": f"q{d}_{i}", "diagram_id": did, "kind": "seg_pair", "refs": [["seg", "A", "B"], ["seg", "B", "C"]],
                         "margins": [0.01 * (i + 1), 0.5, 0.02 * (i + 1), None, None, None]})
    return rows, diagrams


def test_pick_examples_order():
    rows, _ = _rows()
    n = len(rows)
    rng = np.random.default_rng(0)
    p = {"probs": rng.random((n, R)).astype(np.float32), "geo": np.zeros((n, R), np.float32), "asr": np.zeros((n, R), np.float32),
         "mask": np.zeros((n, R), np.float32)}
    p["mask"][:, :3] = 1
    p["asr"][:4, 0] = 1                                                      # asserted-only parallel, margins 0.01..0.04
    p["geo"][5:9, 2] = 1                                                     # geometric-only equal_length, margins 0.02..0.08
    thr = {v: 0.05 for v in VOCAB}
    ex = AG.pick_examples(rows, p, thr, VOCAB, n=2)
    assert [e["relation"] for e in ex["asserted_only"]] == ["parallel", "parallel"]
    assert ex["asserted_only"][0]["index"] == 3                             # the largest margin first (most out of tolerance)
    assert [e["index"] for e in ex["geometric_only"]] == [5, 6]              # the smallest margin first (clearest truth)
    assert ex["geometric_only"][0]["ratio"] == pytest.approx(0.4)


def _runs(tmp_path, rows, quality):
    rng = np.random.default_rng(1)
    runs = tmp_path / "runs"
    log = []
    n = len(rows)
    geo = (rng.random((n, R)) < 0.3).astype(np.float32)
    asr = geo.copy()
    asr[rng.random((n, R)) < 0.2] = 0                                        # some true relations unmarked
    mask = np.ones((n, R), np.float32)
    for exp, q in (("sq_finetune_colab", 0.95), ("sq_probe", 0.6)):
        for s in (0, 1):
            run = f"{exp}_s{s}"
            (runs / run / "predictions").mkdir(parents=True)
            for u in ("val", "geometry3k_test"):
                right = rng.random((n, R)) < q
                probs = np.where(right == (geo > 0), 0.9, 0.1).astype(np.float32)
                np.savez(runs / run / "predictions" / f"{u}.npz", probs=probs, geo=geo, asr=asr, mask=mask,
                         qid=np.array([r["qid"] for r in rows]))
            (runs / run / "metrics.json").write_text(json.dumps({"status": "ok"}))
            log.append({"timestamp": "t", "name": run, "mode": "x", "seed": s, "config_hash": "h", "status": "ok",
                        "best_epoch": 1, "duration_s": 1, "metrics": "{}"})
    with open(runs / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    return runs


def test_analyse_and_markdown(tmp_path):
    rows, diagrams = _rows(n_diag=8, per=6)
    runs = _runs(tmp_path, rows, 0.9)
    latest = AG.latest_runs(runs)
    qid = np.array([r["qid"] for r in rows])
    units = {"geometry3k_test": {"qid": qid, "nts": np.array([diagrams[r["diagram_id"]].flags["not_to_scale"] for r in rows])}}
    table, dis, problems = AG.analyse(runs, latest, ["sq_finetune_colab", "sq_probe", "none"], units, qid, VOCAB)
    assert problems == ["none: no runs"] and len(table) == 2 and len(dis) == 2 * (R + 1)
    ft = next(r for r in table if r["experiment"] == "sq_finetune_colab")
    assert ft["seeds"] == "0 1" and ft["macro_f1_geometric"] > ft["macro_f1_asserted"]      # unmarked truths count against asserted
    all_ = next(r for r in dis if r["experiment"] == "sq_finetune_colab" and r["relation"] == "all")
    assert all_["n"] == 2 * len(rows) * R and all_["n_geometric_only"] > 0
    pr = next(r for r in dis if r["experiment"] == "sq_probe" and r["relation"] == "all")
    assert all_["share_asserted_errors_that_are_label_errors"] > pr["share_asserted_errors_that_are_label_errors"]   # the better model sides with the verifier more
    md = AG.markdown(table, dis, VOCAB, "sq_finetune_colab")
    assert "| geometry3k_test |" in md and "| all |" in md


def test_figures_and_manifest(tmp_path):
    summary = []
    for e in ("sq_probe", "sq_finetune_colab"):
        for u, _ in FG.UNITS:
            row = {"experiment": e, "unit": u, "label_set": "geometric", "operating": "tuned"}
            for v in VOCAB:
                f = 0.5 if e == "sq_probe" else 0.9
                row.update({f"f1_{v}": f, f"f1_lo_{v}": f - 0.05, f"f1_hi_{v}": f + 0.05})
            summary.append(row)
    summary[0][f"f1_{VOCAB[4]}"] = ""                                       # a relation without support: blank cells
    summary[0][f"f1_lo_{VOCAB[4]}"] = summary[0][f"f1_hi_{VOCAB[4]}"] = ""
    FG.per_relation_figure(summary, VOCAB, tmp_path / "f.png")
    headline = [{"unit": u, "probe": "sq_probe", "finetune": "sq_finetune_colab", "seed": "mean", "level": "query", "relation": "all",
                 "n": "100", "share_both": "0.6", "share_finetune_only": "0.25", "share_probe_only": "0.05", "share_neither": "0.1"}
                for u, _ in FG.UNITS]
    FG.headline_figure(headline, tmp_path / "h.png")
    assert (tmp_path / "f.png").exists() and (tmp_path / "h.png").exists()
    (tmp_path / "tolerance_all.png").write_bytes(b"")
    text = FG.manifest(tmp_path)
    assert "| tolerance histograms (combined) | `tolerance_all.png` | yes |" in text and "MISSING" in text
    assert all(any(f in text for f in files) for _, files, _ in FG.MANIFEST)
