"""analysis/walkthrough.py (plan 10.3) without a checkpoint or the datasets: run selection, diagram
choice, the live verifier verdict and the decision table, plus the figure on a tiny model."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from analysis import walkthrough as W
from data.schema import Circle, Diagram

TOL = {"parallel_deg": 3.7, "perpendicular_deg": 3.7, "equal_length_rel": 0.10, "on_circle_rel": 0.0794,
       "tangent_rel": 0.0794, "collinear_rel": 0.0119}


def _run(root, name, f1, backbone, status="ok", ckpt=True):
    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    if ckpt:
        (d / "checkpoints" / "best.pt").write_bytes(b"x")
    (d / "metrics.json").write_text(json.dumps({"status": status, "val_macro_f1": f1, "backbone_loaded": backbone}))


def test_best_checkpoint_by_validation_and_backbone(tmp_path):
    v3 = W.config()["model"]["backbone"]
    _run(tmp_path, "a_s0", 0.86, v3)
    _run(tmp_path, "a_s2", 0.88, v3)
    _run(tmp_path, "dinov2_s0", 0.92, "facebook/dinov2-large")
    _run(tmp_path, "smoke_finetune_s0", 0.99, v3)
    _run(tmp_path, "failed_s0", 0.99, v3, status="failed")
    _run(tmp_path, "nockpt_s0", 0.99, v3, ckpt=False)
    assert W.best_checkpoint(tmp_path) == "a_s2"
    assert W.best_checkpoint(tmp_path, backbone=None) == "dinov2_s0"
    with pytest.raises(FileNotFoundError):
        W.best_checkpoint(tmp_path / "empty")


def _diagram():
    pts = {"A": (0.0, 0.0), "B": (100.0, 0.0), "C": (0.0, 100.0), "D": (100.0, 100.0), "E": (50.0, 0.0)}
    return Diagram(id="synthetic:w", source="synthetic", split="test", image_path="", width=120, height=120,
                   points=pts, segments=[("A", "B"), ("C", "D"), ("A", "C")],
                   circles=[Circle(center=None, center_xy=(50.0, 50.0), radius=50.0)])


def test_verdict_matches_geometry():
    d = _diagram()
    ok, m, r = W.verdict(d, "parallel", [["seg", "A", "B"], ["seg", "C", "D"]], TOL)
    assert ok is True and r == pytest.approx(0.0, abs=1e-9)
    ok, m, r = W.verdict(d, "perpendicular", [["seg", "A", "B"], ["seg", "C", "D"]], TOL)
    assert ok is False and r > 1
    ok, m, r = W.verdict(d, "perpendicular", [["seg", "A", "B"], ["seg", "A", "C"]], TOL)
    assert ok is True and r <= 1
    ok, m, r = W.verdict(d, "tangent", [["seg", "A", "B"], ["circle", 0]], TOL)
    assert ok is True and r <= 1
    ok, m, r = W.verdict(d, "collinear", [["pt", "E"], ["pt", "A"], ["pt", "B"]], TOL)
    assert ok is True


class _DS:
    def __init__(self, rows, diagrams):
        self.rows, self.diagrams = rows, diagrams


def test_pick_diagram_prefers_small_lettered_and_rich():
    lettered = _diagram()
    machine = Diagram(id="pgdp5k:m", source="pgdp5k", split="test", image_path="", width=1, height=1,
                      points={"_p0": (0, 0), "_p1": (1, 1)})
    small_poor = Diagram(id="geometry3k:p", source="geometry3k", split="test", image_path="", width=1, height=1,
                         points={"A": (0, 0)})
    def rows(did, n, rels):
        out = []
        for i in range(n):
            geo = [1 if j in rels and i == 0 else 0 for j in range(6)]
            out.append({"diagram_id": did, "geo": geo, "asr": [0] * 6})
        return out
    ds = _DS(rows("synthetic:w", 8, {0, 1, 2}) + rows("pgdp5k:m", 8, {0, 1, 2, 3, 4}) + rows("geometry3k:p", 8, {0}),
             {"synthetic:w": lettered, "pgdp5k:m": machine, "geometry3k:p": small_poor})
    assert W.pick_diagram(ds) == "synthetic:w"


def test_decision_table_columns_and_agreement():
    res = [{"row": {"refs": [["seg", "A", "B"], ["seg", "C", "D"]]}, "stats_last": {"share_highlight": 0.7},
            "decisions": [{"relation": "parallel", "p": 0.9, "threshold": 0.5, "model": True, "verifier": True,
                           "margin_over_tol": 0.0, "marked": False, "p_eval": 0.9},
                          {"relation": "perpendicular", "p": 0.6, "threshold": 0.5, "model": True, "verifier": False,
                           "margin_over_tol": 15.5, "marked": False, "p_eval": None}]}]
    t = W.decision_table(res)
    assert list(t.primitives.unique()) == ["AB CD"]
    assert t.agrees_with_verifier.tolist() == [True, False]
    assert set(t.columns) >= {"relation", "p", "model", "verifier", "marked", "p_eval", "highlight_attention"}


def test_show_query_figure(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    L = 64
    view = Image.new("RGB", (48, 32), (255, 255, 255))
    res = {"view": view, "diagram": _diagram(), "maps": {"last_layer": np.full((4, 4), 1 / 16)},
           "stats_last": {"share_highlight": 0.5},
           "decisions": [{"relation": "parallel", "p": 0.9, "model": True, "verifier": True,
                          "margin_over_tol": 0.0, "marked": False}]}
    monkeypatch.setattr(W, "original_image", lambda ds, d: view)
    fig = W.show_query(res, type("D", (), {"L": L})(), "t")
    assert len(fig.axes) == 3
