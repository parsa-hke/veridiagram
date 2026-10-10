"""Shared pieces for the probe and fine-tune trainers.

- `autocast(device, cfg)`: bfloat16 autocast on CUDA and MPS when `finetune.amp` is on, else a no-op.
- `EMA`: exponential moving average of a model's parameters and buffers with the usual warm-up,
  decay_t = min(decay, (1 + t) / (10 + t)), so a short run is not dominated by the initial weights.
- `cosine_with_warmup(step)`: linear warm-up, then cosine decay to `min_ratio` of the peak LR.
- `evaluate_units`: predict every evaluation unit, compute metrics against both label sets and save
  the predictions (qid, probs, labels, mask) to <run_dir>/predictions/<unit>.npz for Phase 8.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analysis.metrics import evaluate
from data.dataset import QueryDataset, loader
from data.queries import UNITS, vocabulary
from experiments.run import get_device

EVAL_UNITS = [u for u in UNITS if u != "train"]


def device_of(cfg: dict) -> torch.device:
    return get_device(cfg.get("device", "auto"))


def autocast(device: torch.device, cfg: dict):
    f = cfg.get("finetune", {})
    if not f.get("amp", True) or device.type not in ("cuda", "mps"):
        return contextlib.nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[f.get("amp_dtype", "bfloat16")]
    return torch.autocast(device_type=device.type, dtype=dtype)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.steps = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(m.detach(), alpha=1 - d)
        for e, m in zip(self.model.buffers(), model.buffers()):
            e.copy_(m)

    def state_dict(self) -> dict:
        return {"steps": self.steps, "model": self.model.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.steps = sd["steps"]
        self.model.load_state_dict(sd["model"])


def cosine_with_warmup(warmup: int, total: int, min_ratio: float = 0.01):
    def f(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


@torch.no_grad()
def predict(forward, dl, device: torch.device, cfg: dict) -> dict:
    """Run `forward(images) -> logits` over a loader; returns numpy arrays."""
    probs, geo, asr, mask, qids = [], [], [], [], []
    for x, g, a, meta in dl:
        with autocast(device, cfg):
            logits = forward(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        geo.append(g.numpy())
        asr.append(a.numpy())
        mask.append(meta["mask"].numpy())
        qids += meta["qid"]
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, 0))  # noqa: E731
    return {"probs": cat(probs), "geo": cat(geo), "asr": cat(asr), "mask": cat(mask), "qid": np.array(qids)}


def metrics_for(pred: dict, vocab: list[str]) -> dict:
    out = {}
    for label_set, key in (("geometric", "geo"), ("asserted", "asr")):
        m = evaluate(pred["probs"], pred[key], pred["mask"], vocab)
        out[label_set] = {"macro_f1": round(m["macro_f1"], 4),
                          "f1": {r: round(v["f1"], 4) for r, v in m["per_relation"].items()},
                          "support": {r: v["support"] for r, v in m["per_relation"].items()}}
    out["n"] = int(len(pred["probs"]))
    return out


def save_predictions(pred: dict, run_dir: Path, unit: str) -> None:
    d = Path(run_dir) / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{unit}.npz", **pred)


def eval_loader(cfg: dict, unit: str):
    ds = QueryDataset(cfg, unit, train=False)
    return ds, loader(ds, cfg["eval"].get("batch_size", cfg["finetune"]["batch_size"] * 2), shuffle=False,
                      num_workers=cfg["data"]["num_workers"])


def evaluate_units(forward, cfg: dict, device: torch.device, run_dir: Path, units=None) -> dict:
    vocab = vocabulary(cfg)
    res = {}
    for unit in units or EVAL_UNITS:
        t0 = time.time()
        _, dl = eval_loader(cfg, unit)
        pred = predict(forward, dl, device, cfg)
        save_predictions(pred, run_dir, unit)
        res[unit] = metrics_for(pred, vocab)
        print(f"  eval {unit:16s} n={res[unit]['n']:6d}  macro-F1 geometric {res[unit]['geometric']['macro_f1']:.3f}"
              f"  asserted {res[unit]['asserted']['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return res


def log_epoch(run_dir: Path, record: dict) -> None:
    with open(Path(run_dir) / "history.jsonl", "a") as f:
        f.write(json.dumps(record, default=float) + "\n")
