"""web/server on a tiny DINOv2 (no download): API behaviour, the Ruler, limits, warm-up, and the
parity rule (WEBSITE_PLAN.md §4.1): the server's tensor equals QueryDataset's for real validation
queries. Probability parity with the real checkpoint runs with WEB_PARITY=1 (loads 0.6 GB)."""

import base64
import io
import os

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from data.queries import vocabulary
from experiments.run import REPO_ROOT, load_config
from models.backbone import tiny_backbone
from models.finetune import build_classifier
from web.server import core

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from web.server.app import create_app  # noqa: E402

CFG = load_config(REPO_ROOT / "configs" / "ablation_dinov2.yaml")
SQUARE = {"points": {"A": [0, 0], "B": [100, 0], "C": [100, 100], "D": [0, 100], "O": [50, 50]},
          "segments": [["A", "B"], ["C", "D"], ["B", "C"], ["A", "D"]],
          "circles": [{"center": "O", "through": "A"}]}
needs_data = pytest.mark.skipif(not (REPO_ROOT / "data/processed/index/val.jsonl").exists()
                                and not (REPO_ROOT / "data/processed/geometry3k_figures.jsonl").exists(),
                                reason="processed data not built")


def tiny_engine() -> core.Engine:
    torch.manual_seed(0)
    model = build_classifier(CFG, tiny_backbone("dinov2"), len(vocabulary(CFG)))
    return core.Engine(CFG, model, torch.device("cpu"), "tiny / test")


@pytest.fixture(scope="module")
def engine():
    return tiny_engine()


@pytest.fixture()
def client(engine, monkeypatch):
    monkeypatch.setenv("RATE_PER_MIN", "1000")
    return TestClient(create_app(loader=lambda: engine, background=False))


def _png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health_and_ask_construction(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["ready"] and h["model"] == "tiny / test" and h["device"] == "cpu"
    q = {"kind": "seg_pair", "refs": [["seg", "A", "B"], ["seg", "C", "D"]]}
    r = client.post("/api/ask", json={"construction": SQUARE, "queries": [q, {"kind": "pt_circle",
                                                                             "refs": [["pt", "C"], ["circle", 0]]}]})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    pair, onc = res
    assert pair["applicable"] == ["parallel", "perpendicular", "equal_length"]
    assert set(pair["probs"]) == set(pair["applicable"]) and all(0 <= v <= 1 for v in pair["probs"].values())
    assert pair["ruler"]["basis"] == "construction"
    assert pair["ruler"]["parallel"]["holds"] and pair["ruler"]["parallel"]["measured"] == pytest.approx(0, abs=1e-6)
    assert pair["ruler"]["parallel"]["tolerance"] == CFG["tolerances"]["parallel_deg"]
    assert not pair["ruler"]["perpendicular"]["holds"] and pair["ruler"]["equal_length"]["holds"]
    assert onc["ruler"]["on_circle"]["holds"]                       # C is on the circle through A
    im = Image.open(io.BytesIO(base64.b64decode(pair["model_input"])))
    assert im.size == (CFG["input"]["long_side"],) * 2                  # the exact square the network sees
    a = np.asarray(im.convert("RGB")).astype(int)
    assert (np.abs(a - np.array(CFG["overlay"]["color"])).sum(-1) < 30).sum() > 200   # the green highlight is drawn


def test_read_whole_figure(client):
    r = client.post("/api/read", json={"construction": SQUARE})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["queries_asked"] == out["queries_total"] == len(out["results"]) <= core.MAX_QUERIES
    par = [x for x in out["ruler_relations"] if x["relation"] == "parallel"]
    assert {(tuple(x["refs"][0]), tuple(x["refs"][1])) for x in par} >= {(("seg", "A", "B"), ("seg", "C", "D"))}
    assert isinstance(out["contradictions"], list) and "model_input" not in out["results"][0]


def test_upload_and_errors(client):
    im = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(im).line([(20, 20), (280, 20)], fill="black", width=2)
    ImageDraw.Draw(im).line([(20, 180), (280, 170)], fill="black", width=2)
    prims = {"points": {"A": [20, 20], "B": [280, 20], "C": [20, 180], "D": [280, 170]},
             "segments": [["A", "B"], ["C", "D"]]}
    q = [{"kind": "seg_pair", "refs": [["seg", "C", "D"], ["seg", "A", "B"]]}]   # any order
    r = client.post("/api/ask", json={"image": _png_b64(im), "primitives": prims, "queries": q})
    assert r.status_code == 200, r.text
    ruler = r.json()["results"][0]["ruler"]
    assert ruler["basis"] == "clicks" and 2.0 < ruler["parallel"]["measured"] < 2.4      # atan(10/260) = 2.2°
    bad = [({"construction": SQUARE, "image": _png_b64(im), "queries": q}, "either"),
           ({"image": _png_b64(im), "queries": q}, "primitives"),
           ({"image": "bm90IGFuIGltYWdl", "primitives": prims, "queries": q}, "readable"),
           ({"construction": SQUARE, "queries": [{"kind": "seg_pair", "refs": [["seg", "A", "B"], ["seg", "X", "Y"]]}]},
            "cannot be asked"),
           ({"construction": {**SQUARE, "segments": [["A", "Z"]]}, "queries": q}, "known points"),
           ({"construction": SQUARE, "queries": q * (core.MAX_QUERIES + 1)}, "at most")]
    for body, msg in bad:
        r = client.post("/api/ask", json=body)
        assert r.status_code == 422 and msg in r.json()["error"], (body.keys(), r.text)
    assert client.post("/api/ask", json={**{"construction": SQUARE}, "queries": q, "extra": 1}).status_code == 422


def test_warming_up_and_rate_limit(engine, monkeypatch):
    def failing():
        raise RuntimeError("no checkpoint")
    c = TestClient(create_app(loader=failing, background=False))
    h = c.get("/api/health").json()
    assert not h["ready"] and h["error"] == "the model could not be loaded"      # no paths or details leak
    assert c.post("/api/read", json={"construction": SQUARE}).status_code == 503
    monkeypatch.setenv("RATE_PER_MIN", "2")
    c = TestClient(create_app(loader=lambda: engine, background=False))
    codes = [c.post("/api/read", json={"construction": SQUARE}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_construction_is_rendered_by_the_project_renderer():
    p = core.from_construction(SQUARE, CFG)
    S, M = CFG["synthetic"]["canvas_px"], CFG["synthetic"]["margin_px"]
    assert p.diagram.source == "user" and p.image.size == (S, S)
    c = p.diagram.circles[0]
    left = min([x for x, _ in p.diagram.points.values()] + [c.center_xy[0] - c.radius])
    right = max([x for x, _ in p.diagram.points.values()] + [c.center_xy[0] + c.radius])
    assert left == pytest.approx(M) and right == pytest.approx(S - M)   # fitted into the synthetic canvas
    st = core.clean_style(CFG)
    assert st["scan_noise"] == 0 and st["jpeg_quality"] == 0


@needs_data
def test_parity_with_query_dataset(engine):
    """The server's tensor for a real validation query, sent as an upload, equals QueryDataset's."""
    from data.dataset import QueryDataset
    ds = QueryDataset(CFG, "val", train=False)
    raw = REPO_ROOT / CFG["paths"]["raw"]
    for i in (0, len(ds) // 2, len(ds) - 1):
        row = ds.rows[i]
        d = ds.diagrams[row["diagram_id"]]
        b64 = base64.b64encode((raw / d.image_path).read_bytes()).decode()
        prims = {"points": {k: list(v) for k, v in d.points.items()}, "segments": [list(s) for s in d.segments],
                 "circles": [{"center": None, "center_xy": list(c.center_xy), "radius": c.radius} if c.center_xy else
                             {"center": None, "center_xy": [0, 0], "radius": 1.0} for c in d.circles]}
        p = core.from_upload(b64, prims)
        engine.pad = "patch"
        x, _ = engine.model_input(p, row["refs"])
        assert torch.equal(x, ds[i][0]), row["qid"]
        engine.pad = "square"                                          # the server's default adds white only
        x, view = engine.model_input(p, row["refs"])
        assert torch.equal(x, core.pad_square(ds[i][0], engine.L, engine.white)) and view.size == (engine.L, engine.L)


CKPT = REPO_ROOT / "runs" / "ablation_dinov2_s0" / "checkpoints" / "best.pt"


@needs_data
@pytest.mark.skipif(not (os.environ.get("WEB_PARITY") and CKPT.exists()), reason="set WEB_PARITY=1 with the checkpoint")
def test_probability_parity_with_saved_predictions():
    """Asked as evaluation asked (one original batch of eval.batch_size, padded to its largest image),
    the server reproduces the run's saved val predictions: no decision differs, |dp| within bf16 noise
    (0.04 measured on MPS against CUDA, web.server.parity, 2026-10-13)."""
    from data.dataset import QueryDataset
    e = core.Engine.from_checkpoint(CFG, CKPT, os.environ.get("DEVICE", "auto"), "ablation_dinov2_s0")
    e.pad, e.batch = "patch", CFG["eval"]["batch_size"]
    ds = QueryDataset(CFG, "val", train=False)
    z = np.load(REPO_ROOT / "runs" / "ablation_dinov2_s0" / "predictions" / "val.npz")
    sel = np.arange(0, e.batch)                                       # the first evaluation batch: wide and tall images
    assert all(z["qid"][i] == ds.rows[i]["qid"] for i in sel)
    probs = e.predict([ds[i][0] for i in sel])
    m = z["mask"][sel].astype(bool)
    assert np.abs(probs - z["probs"][sel])[m].max() <= 0.05
    assert ((probs > 0.5) == (z["probs"][sel] > 0.5))[m].all()
