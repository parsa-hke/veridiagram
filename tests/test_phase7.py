"""Phase 7 pieces: verifier/consistency.py (7.10), the no-overlay ablation (7.8: data/query_tokens.py,
QueryAttention) and Part B self-training (7.11: experiments/selftrain.py), on a tiny random ViT, CPU."""

import copy
import json

import numpy as np
import pytest
import torch

from data.query_tokens import CIRCLE, KIND_OFFSET, N_SLOTS, encode, name_token
from experiments.run import REPO_ROOT, load_config
from models.backbone import tiny_backbone
from models.finetune import Classifier, param_groups
from verifier.consistency import closure

SMOKE = load_config(REPO_ROOT / "configs" / "smoke.yaml")
needs_data = pytest.mark.skipif(not (REPO_ROOT / "data/processed/pgps9k_pool.jsonl").exists(),
                                reason="processed data not built")


# ----------------------------------------------------------------------------- 7.10 consistency

def test_closure_rules():
    c, facts = closure([("parallel", "a", "b"), ("parallel", "b", "c")])
    assert not c and ("parallel", "a", "c") in facts                                   # R1
    c, facts = closure([("perpendicular", "a", "b"), ("parallel", "b", "c")])
    assert not c and ("perpendicular", "a", "c") in facts                              # R2
    c, facts = closure([("perpendicular", "a", "c"), ("perpendicular", "b", "c")])
    assert not c and ("parallel", "a", "b") in facts                                   # R3
    c, facts = closure([("equal_length", "x", "y"), ("equal_length", "z", "y")])
    assert not c and ("equal_length", "x", "z") in facts                               # R4
    c, facts = closure([("on_circle", "p", "c0")])
    assert not c and facts == {("on_circle", "c0", "p")}


def test_closure_contradictions():
    c, _ = closure([("parallel", "a", "b"), ("perpendicular", "a", "b")])
    assert c and c[0]["kind"] == "parallel_and_perpendicular"                          # C1
    c, _ = closure([("parallel", "a", "b"), ("parallel", "b", "c"), ("perpendicular", "c", "a")])
    assert c                                                                           # C1 after R1
    c, _ = closure([("perpendicular", "a", "c"), ("perpendicular", "b", "c"), ("perpendicular", "a", "b")])
    assert c                                                                           # R3 then C1
    c, _ = closure([("parallel", "a", "b"), ("parallel", "b", "c")], negatives=[("parallel", "c", "a")])
    assert c and c[0]["kind"] == "derived_negated"                                     # C2
    c, _ = closure([("perpendicular", "a", "b")], same_line=[("a", "b")])
    assert c                                                                           # same line is parallel
    c, _ = closure([("perpendicular", "a", "b"), ("parallel", "c", "d")], negatives=[("parallel", "a", "c")])
    assert not c


# ----------------------------------------------------------------------------- 7.8 query tokens

def test_query_tokens_encoding():
    t = encode("seg_pair", [["seg", "A", "B"], ["seg", "C", "D"]])
    assert t.shape == (N_SLOTS,) and t[0] == KIND_OFFSET
    assert t[1:5].tolist() == [name_token(n) for n in "ABCD"] and t[5:].tolist() == [0, 0]
    t = encode("pt_circle", [["pt", "P"], ["circle", 0]], {0: "O"})
    assert t[1:5].tolist() == [name_token("P"), 0, name_token("O"), 0]
    assert encode("seg_circle", [["seg", "A", "B"], ["circle", 1]], {1: None})[3] == CIRCLE
    assert name_token("A") == encode("pt_circle", [["pt", "A"], ["circle", 0]])[1]     # stable across calls


def test_query_attention_classifier():
    model = Classifier(tiny_backbone("dinov3"), 6, query_tokens=True)
    x = torch.randn(2, 3, 64, 64)
    q = torch.stack([encode("seg_pair", [["seg", "A", "B"], ["seg", "C", "D"]]),
                     encode("seg_pair", [["seg", "A", "C"], ["seg", "B", "D"]])])
    out = model(x, qtok=q)
    assert out.shape == (2, 6) and not torch.allclose(out[0], out[1])                   # the query matters
    groups = param_groups(model, {"backbone_lr": 1e-3, "llrd": 0.8, "head_lr": 1e-2, "weight_decay": 0.05})
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))      # query encoder trained


def _tiny_cfg(tmp_path):
    cfg = copy.deepcopy(SMOKE)
    cfg["device"] = "cpu"
    cfg["data"].update(max_diagrams=3, synthetic_train_size=0, max_queries_per_unit=12)
    cfg["paths"]["cache"] = str(tmp_path / "cache")
    cfg["finetune"].update(epochs=1, batch_size=4, save_last=True, amp=False, max_steps_per_epoch=2)
    cfg["eval"]["batch_size"] = 8
    return cfg


@needs_data
def test_no_overlay_dataset_and_finetune(tmp_path):
    from data.dataset import QueryDataset, loader
    from models import finetune

    cfg = _tiny_cfg(tmp_path)
    cfg["overlay"]["enabled"] = False
    cfg["model"]["query_input"] = "tokens"
    ds = QueryDataset(cfg, "val", train=False)
    same = [i for i, r in enumerate(ds.rows) if r["diagram_id"] == ds.rows[0]["diagram_id"]][:2]
    a, b = ds[same[0]], ds[same[1]]
    assert torch.equal(a[0], b[0]) and not torch.equal(a[3]["qtok"], b[3]["qtok"])     # no highlight
    x, _, _, meta = next(iter(loader(ds, 4, shuffle=False)))
    assert meta["qtok"].shape == (4, N_SLOTS)
    out = finetune.run(cfg, tmp_path / "noov", backbone=tiny_backbone("dinov3"))
    assert "lp_queries" not in out and set(out["results"]) >= {"val", "geometry3k_test"}


# ----------------------------------------------------------------------------- 7.11 self-training

@needs_data
def test_pool_unit_excludes_training_figures(tmp_path):
    from data.dataset import QueryDataset
    from data.schema import read_jsonl

    cfg = _tiny_cfg(tmp_path)
    ds = QueryDataset(cfg, "pgps9k_pool", train=False)
    in_train = {d.id for d in read_jsonl(REPO_ROOT / "data/processed/pgps9k_pool.jsonl") if d.meta.get("in_train")}
    assert ds.rows and not ({r["diagram_id"] for r in ds.rows} & in_train)


@needs_data
def test_pseudo_label_filters(tmp_path):
    from data.dataset import QueryDataset
    from experiments.selftrain import pseudo_label

    cfg = _tiny_cfg(tmp_path)
    cfg["data"]["max_queries_per_unit"] = None
    cfg["data"]["train_none_ratio"] = 1000.0
    cfg["selftrain"]["consistency_negatives"] = False                   # plan rule C1 only
    ds = QueryDataset(cfg, "pgps9k_pool", train=False)
    rows = ds.rows
    mask = np.array([r["mask"] for r in rows], bool)
    probs = np.where(np.array([r["geo"] for r in rows]) == 1, 0.97, 0.03) * mask
    # make one segment pair confidently parallel AND perpendicular: its diagram must be dropped by (b)
    i = next(k for k, r in enumerate(rows) if r["kind"] == "seg_pair" and r["mask"][0] and r["mask"][1])
    probs[i, 0] = probs[i, 1] = 0.99
    probs[i + 1 if i + 1 < len(rows) else i - 1] = np.where(mask[i + 1 if i + 1 < len(rows) else i - 1], 0.6, 0)
    bad = rows[i]["diagram_id"]
    out = {}
    for flt in ("confidence", "consistency", "none"):
        cfg["selftrain"]["filter"] = flt
        out[flt], rep = pseudo_label(cfg, rows, probs, ds.diagrams)
    assert out["none"] == [] and rep["contradiction_diagrams"] >= 1
    assert any(r["diagram_id"] == bad for r in out["confidence"])
    assert not any(r["diagram_id"] == bad for r in out["consistency"])
    assert {r["qid"] + r["diagram_id"] for r in out["consistency"]} <= {r["qid"] + r["diagram_id"] for r in out["confidence"]}
    for r in out["confidence"]:
        assert r["asr"] == [0] * 6 and all(m <= a for m, a in zip(r["mask"], next(
            x["mask"] for x in rows if x["qid"] == r["qid"] and x["diagram_id"] == r["diagram_id"])))
    assert rep["confidence"]["entries"] > rep["consistency"]["entries"]
    assert rep["truth_contradiction_rate"] == 0 and rep["removed_by_consistency"]["entries"] > 0   # C1 never fires on the verifier labels


@needs_data
def test_selftrain_end_to_end(tmp_path):
    from experiments import selftrain
    from models import finetune

    cfg = _tiny_cfg(tmp_path)
    runs = tmp_path / "runs"
    teacher = finetune.build_classifier(cfg, tiny_backbone("dinov3"), 6)
    (runs / "teacher_s0" / "checkpoints").mkdir(parents=True)
    torch.save({"model": teacher.state_dict(), "backbone": "tiny-dinov3"}, runs / "teacher_s0/checkpoints/best.pt")
    cfg["selftrain"].update(teacher_run="teacher_s0", filter="consistency", tau=0.0)
    out = selftrain.run(cfg, runs / "st_s0", backbone=tiny_backbone("dinov3"))
    assert out["filter"] == "consistency" and out["pseudo_labels"]["pool_queries"] > 0
    assert (runs / "selftrain_teacher/teacher_s0_smoke/pool_predictions.npz").exists()
    rows = [json.loads(line) for line in open(runs / "st_s0/pseudo_index.jsonl")]
    assert rows and all(r["set"] == "pgps9k_pool" and r["pseudo"] for r in rows)
    assert out["n_train"] > len(rows) and set(out["results"]) >= {"val", "pgdp5k_heldout"}
    cfg["selftrain"]["filter"] = "none"                                  # control reuses the cached teacher
    out2 = selftrain.run(cfg, runs / "ctl_s0", backbone=tiny_backbone("dinov3"))
    assert not (runs / "ctl_s0/pseudo_index.jsonl").exists() and out2["n_train"] < out["n_train"]


@needs_data
def test_consistency_negatives_flag_more(tmp_path):
    from data.dataset import QueryDataset
    from experiments.selftrain import pseudo_label

    cfg = _tiny_cfg(tmp_path)
    cfg["data"]["max_diagrams"] = 40
    ds = QueryDataset(cfg, "pgps9k_pool", train=False)
    true = np.array([r["geo"] for r in ds.rows])
    rates = {}
    for neg in (False, True):
        cfg["selftrain"]["consistency_negatives"] = neg
        _, rep = pseudo_label(cfg, ds.rows, np.where(true == 1, 0.97, 0.03), ds.diagrams)
        rates[neg] = rep["truth_contradiction_rate"]
    assert rates[False] == 0 and rates[True] >= rates[False]
