"""Phase 8.4 on synthetic diagrams and predictions (no data, no GPU; ~1 s)."""

import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import error_analysis as EA

VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
R = len(VOCAB)
THR = {"parallel": math.sin(math.radians(3.7)), "perpendicular": math.sin(math.radians(3.7)), "equal_length": 0.1,
       "on_circle": 0.0794, "tangent": 0.0794, "collinear": 0.0119}


def _diagram(n_points, size=100.0, nts=False, annotation="pgdp5k"):
    pts = {chr(65 + i): (size * i / max(n_points - 1, 1), 0.0 if i % 2 else size * 0.3) for i in range(n_points)}
    return SimpleNamespace(points=pts, circles=[], flags={"not_to_scale": nts}, meta={"annotation": annotation},
                           source="geometry3k")


def test_queried_segments_and_bins():
    d = _diagram(4)                        # A(0,30) B(33.3,0) C(66.7,30) D(100,0)
    seg = EA.queried_segments(d, [["seg", "A", "D"], ["seg", "A", "B"]])
    assert len(seg) == 2 and seg[0] == pytest.approx(math.hypot(100, 30))
    assert EA.queried_segments(d, [["pt", "C"], ["pt", "A"], ["pt", "D"]]) == [pytest.approx(math.hypot(100, 30))]
    assert EA.queried_segments(d, [["pt", "A"], ["circle", 0]]) == []
    assert [EA._bin(n, EA.POINT_BINS) for n in (2, 4, 5, 6, 7, 9, 12, 13, 20)] == \
        ["<=4", "<=4", "5-6", "5-6", "7-8", "9-12", "9-12", "13-16", "17-20"]
    assert [EA._bin(x, EA.SEGMENT_BINS) for x in (0.0, 0.09, 0.1, 0.34, 0.35, 2.0)] == \
        ["<0.10", "<0.10", "0.10-0.20", "0.20-0.35", ">=0.35", ">=0.35"]


def _rows(diagrams, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for did, d in diagrams.items():
        names = list(d.points)
        for i in range(per):
            a, b, c = rng.choice(names, 3, replace=False)
            kind = "seg_pair" if i % 2 else "pt_line"
            refs = [["seg", a, b], ["seg", b, c]] if kind == "seg_pair" else [["pt", a], ["pt", b], ["pt", c]]
            margins = [float(rng.uniform(0, 0.3)), float(rng.uniform(0, 0.3)), float(rng.uniform(0, 0.5)), None, None, None] \
                if kind == "seg_pair" else [None] * 5 + [float(rng.uniform(0, 0.05))]
            rows.append({"qid": f"{did}_{i}", "diagram_id": did, "kind": kind, "refs": refs, "margins": margins})
    return rows


def test_query_features():
    diagrams = {"g:0": _diagram(3), "g:1": _diagram(8, nts=True, annotation="geometry3k"), "s:0": _diagram(15, size=500)}
    rows = _rows(diagrams)
    f = EA.query_features(rows, diagrams)
    assert len(f["points"]) == 18 and set(f["points"]) == {"<=4", "7-8", "13-16"}
    assert set(f["source"]) == {"pgdp5k", "geometry3k"} and (f["not_to_scale"] == "not to scale").sum() == 6
    assert f["margins"].shape == (18, R) and np.isnan(f["margins"][0, 3])
    assert all(s in {b[2] for b in EA.SEGMENT_BINS} for s in f["segment"])   # every query here has a segment


def _preds(rows, quality, rng):
    n = len(rows)
    mask = np.array([[1, 1, 1, 0, 0, 0] if r["kind"] == "seg_pair" else [0, 0, 0, 0, 0, 1] for r in rows], np.float32)
    margins = np.array([[np.nan if m is None else m for m in r["margins"]] for r in rows])
    thr = np.array([THR[v] for v in VOCAB])
    geo = (np.nan_to_num(margins, nan=9.0) <= thr).astype(np.float32)
    right = rng.random((n, R)) < quality
    probs = np.where(right == (geo > 0), 0.9, 0.1).astype(np.float32)
    return probs, geo, mask


def _runs(tmp_path, diagrams, rows_of):
    rng = np.random.default_rng(1)
    runs = tmp_path / "runs"
    log = []
    for exp, q in (("sq_finetune_colab", 0.9), ("sq_probe", 0.6)):
        for s in (0, 1):
            run = f"{exp}_s{s}"
            (runs / run / "predictions").mkdir(parents=True)
            for u, rows in rows_of.items():
                probs, geo, mask = _preds(rows, q, rng)
                np.savez(runs / run / "predictions" / f"{u}.npz", probs=probs, geo=geo, asr=geo, mask=mask,
                         qid=np.array([r["qid"] for r in rows]))
            (runs / run / "metrics.json").write_text(json.dumps({"status": "ok"}))
            log.append({"timestamp": "t", "name": run, "mode": "x", "seed": s, "config_hash": "h", "status": "ok",
                        "best_epoch": 1, "duration_s": 1, "metrics": "{}"})
    with open(runs / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    return runs


def test_margin_table_perfect_and_breakdown_sums():
    diagrams = {f"g:{i}": _diagram(5 + i) for i in range(6)}
    rows = _rows(diagrams, per=10)
    probs, geo, mask = _preds(rows, 1.0, np.random.default_rng(0))       # perfect predictions
    f = EA.query_features(rows, diagrams)
    mt = EA.margin_table(probs >= 0.5, geo, mask, f["margins"], THR, VOCAB)
    assert len(mt) == R * len(EA.MARGIN_BINS)
    assert all(r["accuracy"] == 1.0 for r in mt if r["n"])
    assert sum(r["n"] for r in mt if r["relation"] == "parallel") == int(mask[:, 0].sum())
    assert all(r["positives"] == r["n"] for r in mt if r["bin_lo"] < 1 and r["n"])     # below the tolerance: positives
    assert all(r["positives"] == 0 for r in mt if r["bin_lo"] >= 1)
    bd = EA.breakdown(probs >= 0.5, geo, mask, f["points"], VOCAB, "points")
    assert sum(r["n_queries"] for r in bd) == len(rows) and all(r["macro_f1"] == 1.0 for r in bd)
    assert [r["group"] for r in bd] == [g for g in ("5-6", "7-8", "9-12") if g in set(f["points"])]
    rel = EA.relation_rows(probs >= 0.5, geo, mask, VOCAB)
    assert [r["group"] for r in rel] == VOCAB and rel[0]["f1"] == 1.0


def test_analyse_tables_figures_and_problems(tmp_path):
    diagrams = {"g:0": _diagram(3), "g:1": _diagram(8, nts=True), "g:2": _diagram(12)}
    units = {}
    for u in ("geometry3k_test", "pgdp5k_heldout"):
        rows = _rows(diagrams, per=8, seed=hash(u) % 100)
        units[u] = {"rows": rows, "features": EA.query_features(rows, diagrams)}
    rows_of = {u: v["rows"] for u, v in units.items()}
    rows_of["val"] = _rows(diagrams, per=4, seed=7)
    runs = _runs(tmp_path, diagrams, rows_of)
    latest = EA.latest_runs(runs)
    val_qid = np.array([r["qid"] for r in rows_of["val"]])
    table, margin, problems = EA.analyse(runs, latest, ["sq_finetune_colab", "sq_probe", "missing"], units, val_qid, VOCAB, THR)
    assert problems == ["missing: no runs"]
    assert {r["breakdown"] for r in table} == set(EA.BREAKDOWNS)
    pts = [r for r in table if r["experiment"] == "sq_finetune_colab" and r["unit"] == "geometry3k_test" and r["breakdown"] == "points"]
    assert sum(r["n_queries"] for r in pts) == 2 * 24 and pts[0]["seeds"] == "0 1"       # two seeds pooled
    ft = np.mean([r["macro_f1"] for r in table if r["experiment"] == "sq_finetune_colab" and r["breakdown"] == "source"])
    pr = np.mean([r["macro_f1"] for r in table if r["experiment"] == "sq_probe" and r["breakdown"] == "source"])
    assert ft > pr
    md = EA.markdown(table, margin, VOCAB, "sq_finetune_colab")
    assert "### Macro-F1 by points" in md and "| not to scale |" in md and "| collinear |" in md
    paths = EA.figures(table, margin, VOCAB, "sq_finetune_colab", tmp_path / "figs")
    assert [p.name for p in paths] == ["error_margin.png", "error_breakdowns.png"] and all(p.exists() for p in paths)
    # a stale validation file is reported, not silently used
    z = dict(np.load(runs / "sq_probe_s0/predictions/val.npz"))
    z["qid"] = z["qid"][::-1]
    np.savez(runs / "sq_probe_s0/predictions/val.npz", **z)
    _, _, problems = EA.analyse(runs, latest, ["sq_probe"], units, val_qid, VOCAB, THR)
    assert problems and "sq_probe_s0/val" in problems[0]
