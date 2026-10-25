"""Phase 8.5 with a tiny random ViT (no checkpoint, no data; a few seconds)."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from analysis import attention_maps as AM
from models.backbone import tiny_backbone

VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]
R = len(VOCAB)


def test_rollout_and_patch_map():
    T = 7
    eye = [torch.eye(T).repeat(2, 1, 1) for _ in range(3)]                  # identity attention, 2 heads
    r = AM.rollout(eye)
    assert r.shape == (T,) and r[0] == pytest.approx(1.0) and r[1:].sum() == pytest.approx(0.0)
    uni = [torch.full((2, T, T), 1.0 / T) for _ in range(2)]                 # uniform attention
    r = AM.rollout(uni)
    assert r.sum() == pytest.approx(1.0) and r[1:].std() < 1e-6 and r[0] > r[1]   # residual keeps CLS mass
    last = AM.last_layer(uni)
    assert last == pytest.approx(np.full(T, 1.0 / T))
    m = AM.patch_map(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]), prefix=3, grid=(2, 2))
    assert m.shape == (2, 2) and m.sum() == pytest.approx(1.0) and m[0, 0] == pytest.approx(1 / 3)


def test_patch_classes_and_shares():
    L, patch = 64, 16
    img = Image.new("RGB", (48, 32), (255, 255, 255))                       # 3 x 2 patches of image, rest padding
    ImageDraw.Draw(img).line([(0, 8), (47, 8)], fill=(0, 0, 0), width=2)    # ink across the top row
    hl = np.zeros((32, 48), bool)
    hl[20:28, 0:8] = True                                                   # highlight in patch (1, 0)
    cls = AM.patch_classes(img, hl, L, patch)
    assert cls.shape == (4, 4)
    names = np.array(AM.CLASSES)[cls]
    assert names[1, 0] == "highlight" and (names[0, :3] == "ink").all() and names[1, 1] == "blank"
    assert (names[2:, :] == "padding").all() and (names[:, 3] == "padding").all()
    amap = np.zeros((4, 4))
    amap[1, 0] = 1.0                                                        # all attention on the highlight
    s = AM.attention_shares(amap, cls)
    assert s["share_highlight"] == 1.0 and s["area_highlight"] == pytest.approx(1 / 16)
    assert s["ratio_highlight"] == pytest.approx(16.0) and s["share_padding"] == 0.0 and s["ratio_padding"] == 0.0


def test_select_queries_round_robin_and_confidence():
    rng = np.random.default_rng(0)
    n = 60
    probs = rng.random((n, R)).astype(np.float32)
    geo = (rng.random((n, R)) < 0.5).astype(np.float32)
    mask = np.ones((n, R), np.float32)
    mask[:, 4] = 0                                                          # tangent never applicable
    probs[0, 0], geo[0, 0] = 1.0, 1                                         # the clearest right parallel
    probs[1, 1], geo[1, 1] = 0.0, 1                                         # the clearest wrong perpendicular
    mask[0, 1:], mask[1, [0, 2, 3, 5]] = 0, 0                               # ... and candidates for nothing else
    th = np.full(R, 0.5)
    preds = {"probs": probs, "geo": geo, "mask": mask, "qid": np.array([f"q{i}" for i in range(n)])}
    sel = AM.select_queries(preds, th, VOCAB, 5, 5)
    assert [s["kind"] for s in sel] == ["correct"] * 5 + ["wrong"] * 5
    assert [s["relation"] for s in sel[:5]] == ["parallel", "perpendicular", "equal_length", "on_circle", "collinear"]
    assert sel[0]["index"] == 0 and sel[0]["label"] == 1 and sel[0]["prob"] == pytest.approx(1.0)
    assert sel[6]["index"] == 1 and sel[6]["relation"] == "perpendicular" and sel[6]["label"] == 1
    assert all((s["prob"] >= 0.5) == bool(s["label"]) for s in sel[:5])
    assert all((s["prob"] >= 0.5) != bool(s["label"]) for s in sel[5:])
    assert len({s["index"] for s in sel}) == 10                             # no query twice


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tiny_backbone("dinov3", eager=True)
        self.head = torch.nn.Linear(self.backbone.out_dim, R)


def test_forward_with_attention_and_figures(tmp_path):
    torch.manual_seed(0)
    model = _Tiny().eval()
    L = 64
    x = torch.randn(1, 3, L, L)
    probs, att = AM.forward_with_attention(model, x)
    assert probs.shape == (R,) and len(att) == model.backbone.num_layers
    T = model.backbone.num_prefix + (L // 16) ** 2
    assert att[0].shape == (2, T, T) and torch.allclose(att[0].sum(-1), torch.ones(2, T), atol=1e-4)
    grid = (L // 16, L // 16)
    maps = {"last_layer": AM.patch_map(AM.last_layer(att), model.backbone.num_prefix, grid),
            "rollout": AM.patch_map(AM.rollout(att), model.backbone.num_prefix, grid)}
    assert all(m.shape == grid and m.sum() == pytest.approx(1.0) for m in maps.values())
    view = Image.new("RGB", (48, 32), (255, 255, 255))
    AM.query_figure(view, maps, L, "t", tmp_path / "q.png")
    classes = AM.patch_classes(view, np.zeros((32, 48), bool), L, 16)
    item = {"kind": "correct", "relation": "parallel", "label": 1, "prob": 0.9, "view": view, "maps": maps,
            "stats_last": AM.attention_shares(maps["last_layer"], classes),
            "stats_rollout": AM.attention_shares(maps["rollout"], classes)}
    AM.grid_figure([item, {**item, "kind": "wrong"}], L, tmp_path / "g.png", "t")
    assert (tmp_path / "q.png").exists() and (tmp_path / "g.png").exists()
    stats = [{"map": m, "kind": k, **AM.attention_shares(maps[m], classes)} for m in maps for k in ("correct", "sample")]
    summ = AM.summarise(stats)
    assert {(s["map"], s["kind"]) for s in summ} == {(m, k) for m in maps for k in ("all", "correct", "sample")}
    assert all(s["share_padding"] + s["share_blank"] + s["share_ink"] + s["share_highlight"] == pytest.approx(1.0, abs=1e-3) for s in summ)
