"""data/queries.py index functions and data/dataset.py (plan §5.1-5.2, 5.4, 5.6-5.8)."""

import copy

import numpy as np
import pytest

from data import dataset as DS
from data.queries import UNITS, cap_rows, downsample_none, label_rows, vocabulary
from data.schema import Circle, Diagram, Relation, pt, seg
from experiments.run import REPO_ROOT, load_config

SMOKE = load_config(REPO_ROOT / "configs" / "smoke.yaml")
VOCAB = vocabulary(SMOKE)
needs_data = pytest.mark.skipif(not (REPO_ROOT / "data/processed/geometry3k_figures.jsonl").exists()
                                or not (REPO_ROOT / "data/processed/synth_train.jsonl").exists(),
                                reason="processed data not built")


def _row(kind, pos):
    return {"kind": kind, "geo": [int(pos)] + [0] * 5, "asr": [0] * 6}


def test_vocabulary_and_labels():
    assert VOCAB == ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
    d = Diagram(id="synthetic:x", source="synthetic", split="train", image_path="", width=200, height=200,
                points={"A": (0.0, 0.0), "B": (100.0, 0.0), "C": (0.0, 50.0), "D": (100.0, 50.0), "P": (50.0, 100.0)},
                segments=[("A", "B"), ("C", "D")], circles=[Circle(None, (50.0, 140.0), 40.0, [])],
                asserted=[Relation("parallel", [seg("A", "B"), seg("C", "D")])])
    rows = label_rows(d, VOCAB, SMOKE["tolerances"], ("collinear",), "x")
    pair = next(r for r in rows if r["kind"] == "seg_pair")
    assert pair["geo"][:3] == [1, 0, 1] and pair["asr"][:3] == [1, 0, 0] and pair["mask"] == [1, 1, 1, 0, 0, 0]
    pc = next(r for r in rows if r["kind"] == "pt_circle" and r["refs"][0] == pt("P"))
    assert pc["geo"][3] == 1 and pc["mask"] == [0, 0, 0, 1, 0, 0]


def test_cap_and_downsample():
    rng = np.random.default_rng(0)
    rows = [_row("seg_pair", i < 5) for i in range(100)]
    kept = cap_rows(rows, 20, rng, keep_positives=True)
    assert len(kept) == 20 and sum(any(r["geo"]) for r in kept) == 5
    many_pos = [_row("seg_pair", True) for _ in range(30)]
    assert len(cap_rows(many_pos, 20, rng, keep_positives=True)) == 30                 # all positives kept
    assert len(cap_rows(rows, 20, rng, keep_positives=False)) == 20
    ds = downsample_none(rows + [_row("pt_circle", i < 2) for i in range(50)], 3.0, rng)
    for kind, p in (("seg_pair", 5), ("pt_circle", 2)):
        rk = [r for r in ds if r["kind"] == kind]
        assert sum(any(r["geo"]) for r in rk) == p and len(rk) - p == 3 * p


@needs_data
def test_smoke_dataset_batches(tmp_path):
    cfg = copy.deepcopy(SMOKE)
    cfg["data"]["max_diagrams"] = 3
    cfg["data"]["synthetic_train_size"] = 3
    cfg["paths"]["cache"] = str(tmp_path / "cache")
    for unit in UNITS:
        ds = DS.QueryDataset(cfg, unit, build=True)
        assert len(ds) > 0
        x, geo, asr, meta = next(iter(DS.loader(ds, 4, shuffle=False)))
        B = min(4, len(ds))
        assert x.shape[:2] == (B, 3) and x.shape[2] % 16 == 0 and x.shape[3] % 16 == 0 and max(x.shape[2:]) <= 448
        assert geo.shape == asr.shape == meta["mask"].shape == (B, len(VOCAB))
        assert ((geo <= meta["mask"]) | (meta["mask"] == 1)).all() and not (geo * (1 - meta["mask"])).any()
    assert any((tmp_path / "cache").rglob("*.png"))                                   # overlays cached
    DS.check_disjoint(cfg)
