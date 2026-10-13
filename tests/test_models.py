"""models/{backbone,head,common,probe,finetune}.py and analysis/metrics.py on a tiny random ViT (no
download), CPU only (plan §6.1-6.5)."""

import copy
import json

import numpy as np
import pytest
import torch

from analysis.metrics import evaluate, macro_f1, per_relation
from models import finetune, probe
from models.backbone import tiny_backbone
from models.common import EMA, cosine_with_warmup
from models.finetune import Classifier, param_groups
from models.head import Head, masked_bce, pos_weights
from experiments.run import REPO_ROOT, load_config

SMOKE = load_config(REPO_ROOT / "configs" / "smoke.yaml")
needs_data = pytest.mark.skipif(not (REPO_ROOT / "data/processed/geometry3k_figures.jsonl").exists(),
                                reason="processed data not built")


@pytest.mark.parametrize("kind", ["dinov3", "dinov2"])
def test_backbone_outputs_variable_size(kind):
    bb = tiny_backbone(kind).eval()
    for h, w in ((64, 96), (112, 48)):
        out = bb(torch.randn(2, 3, h, w))
        n = (h // bb.patch_size) * (w // bb.patch_size)
        assert out["cls"].shape == (2, bb.dim) and out["patches"].shape == (2, n, bb.dim)
        assert out["pooled"].shape == (2, bb.out_dim) and torch.allclose(out["pooled"][:, : bb.dim], out["cls"])


def test_layer_depth_and_llrd():
    bb = tiny_backbone("dinov3", layers=3)
    model = Classifier(bb, 6)
    f = {"backbone_lr": 1e-3, "llrd": 0.5, "head_lr": 1e-2, "weight_decay": 0.05}
    groups = param_groups(model, f)
    lr_of = {}
    for g in groups:
        for p in g["params"]:
            lr_of[id(p)] = g["lr"]
    named = dict(model.named_parameters())
    assert lr_of[id(named["backbone.model.model.layer.2.mlp.up_proj.weight"])] == pytest.approx(1e-3)       # last block
    assert lr_of[id(named["backbone.model.model.layer.0.mlp.up_proj.weight"])] == pytest.approx(1e-3 * 0.25)
    assert lr_of[id(named["backbone.model.embeddings.patch_embeddings.weight"])] == pytest.approx(1e-3 * 0.125)
    assert lr_of[id(named["backbone.model.norm.weight"])] == pytest.approx(1e-3)
    assert lr_of[id(named["head.fc.weight"])] == pytest.approx(1e-2)
    nodecay = [g for g in groups if g["weight_decay"] == 0]
    assert any(p is named["head.fc.bias"] for g in nodecay for p in g["params"])
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_head_loss_mask_and_weights():
    logits = torch.tensor([[5.0, -5.0], [0.0, 100.0]])
    y = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    m = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    l1 = masked_bce(logits, y, m)
    l2 = masked_bce(logits, y, torch.tensor([[1.0, 1.0], [1.0, 0.0]]) * torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    assert torch.isclose(l1, l2) and l1 < 1.0                  # the masked 100-logit wrong entry is ignored
    w = pos_weights(np.array([[1, 0], [0, 0], [0, 1], [0, 0]]), np.array([[1, 1], [1, 1], [1, 1], [0, 0]]))
    assert w.tolist() == [2.0, 2.0]
    assert Head(8, 3)(torch.randn(4, 8)).shape == (4, 3)


def test_metrics():
    vocab = ["a", "b"]
    probs = np.array([[0.9, 0.2], [0.8, 0.9], [0.1, 0.6], [0.2, 0.1]])
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    m = np.ones_like(y)
    per = per_relation(probs >= 0.5, y, m, vocab)
    assert (per["a"]["tp"], per["a"]["fp"], per["a"]["fn"]) == (1, 1, 1) and per["a"]["f1"] == pytest.approx(0.5)
    assert per["b"]["f1"] == pytest.approx(2 / 3)
    assert macro_f1(per) == pytest.approx((0.5 + 2 / 3) / 2)
    m2 = m.copy()
    m2[:, 1] = 0
    assert evaluate(probs, y, m2, vocab)["macro_f1"] == pytest.approx(0.5)      # b has no applicable entries


def test_ema_and_schedule():
    net = torch.nn.Linear(2, 1)
    ema = EMA(net, 0.999)
    with torch.no_grad():
        net.weight.add_(1.0)
    ema.update(net)                                              # warm-up: decay (1+1)/(10+1)
    d = 2 / 11
    assert torch.allclose(ema.model.weight, d * (net.weight - 1.0) + (1 - d) * net.weight)
    f = cosine_with_warmup(10, 110)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0) and f(110) == pytest.approx(0.01)
    assert f(60) == pytest.approx(0.01 + 0.99 * 0.5, rel=1e-6)


def _tiny_cfg(tmp_path):
    cfg = copy.deepcopy(SMOKE)
    cfg["device"] = "cpu"
    cfg["data"].update(max_diagrams=3, synthetic_train_size=0, max_queries_per_unit=12)
    cfg["paths"]["cache"] = str(tmp_path / "cache")
    cfg["finetune"].update(epochs=2, batch_size=4, save_last=True, amp=False)
    cfg["probe"].update(epochs=3, batch_size=8)
    cfg["eval"]["batch_size"] = 8
    return cfg


@needs_data
def test_finetune_end_to_end_and_resume(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg["finetune"]["keep_last"] = True                                 # keep the resume state to test resuming
    torch.manual_seed(0)
    out = finetune.run(cfg, tmp_path / "ft", backbone=tiny_backbone("dinov3"))
    assert (tmp_path / "ft/checkpoints/best.pt").exists() and (tmp_path / "ft/checkpoints/last.pt").exists()
    best = torch.load(tmp_path / "ft/checkpoints/best.pt", weights_only=False)["model"]
    assert all(v.dtype == torch.bfloat16 for v in best.values() if v.is_floating_point())
    assert set(out["results"]) == {"val", "geometry3k_test", "pgdp5k_heldout", "synth_extrap"}
    hist = [json.loads(line) for line in open(tmp_path / "ft/history.jsonl")]
    assert [h["epoch"] for h in hist] == [1, 2] and all("val_f1" in h and "lr" in h for h in hist)
    assert (tmp_path / "ft/predictions/pgdp5k_heldout.npz").exists()
    cfg["finetune"]["epochs"] = 3                                       # resume: only epoch 3 runs
    finetune.run(cfg, tmp_path / "ft", backbone=tiny_backbone("dinov3"))
    hist = [json.loads(line) for line in open(tmp_path / "ft/history.jsonl")]
    assert [h["epoch"] for h in hist] == [1, 2, 3]
    cfg["finetune"].update(keep_last=False, epochs=1)
    finetune.run(cfg, tmp_path / "ft2", backbone=tiny_backbone("dinov3"))
    assert not (tmp_path / "ft2/checkpoints/last.pt").exists()            # deleted once finished (default)


@needs_data
def test_probe_end_to_end(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    out = probe.run(cfg, tmp_path / "pr", backbone=tiny_backbone("dinov3"))
    assert 1 <= out["best_epoch"] <= 3 and "pgdp5k_heldout" in out["results"]
    feats = list((tmp_path / "cache" / "features").rglob("*.npz"))
    assert len(feats) == 5                                              # one per unit, cached
    again = probe.run(cfg, tmp_path / "pr2", backbone=tiny_backbone("dinov3"))
    assert again["results"].keys() == out["results"].keys()


def test_describe_tiny_backbone():
    from models.backbone import describe
    info = describe(tiny_backbone("dinov3"), size=(64, 96))
    assert info["register_tokens"] == 4 and info["patch_tokens"] == info["expected_patch_tokens"] == 24
    assert info["pooled_dim"] == 2 * info["hidden"]


def test_train_sampler_mix_and_budget():
    from models.finetune import train_sampler

    class DS:
        rows = [{"set": "geometry3k_figures"}] * 30 + [{"set": "synth_train"}] * 270

        def __len__(self):
            return len(self.rows)

        def auxiliary(self):
            return np.array([r["set"] != "geometry3k_figures" for r in self.rows])

    gen = torch.Generator().manual_seed(0)
    s = train_sampler(DS(), {"epoch_images": 4000, "real_fraction": 0.5}, gen)
    idx = list(iter(s))
    assert len(idx) == 4000 and abs(sum(i < 30 for i in idx) / 4000 - 0.5) < 0.04
    assert train_sampler(DS(), {"epoch_images": None, "real_fraction": None}, gen) is None
