"""Linear probe on cached frozen features (plan §6.3). Dispatched by experiments.run (mode: probe).

1. Frozen backbone, no augmentation: pooled features (CLS ++ mean patch) for every query of every
   unit, cached at <cache>/features/<backbone>/<overlay-settings hash>/<index file>.npz together
   with their query ids (a cache whose ids do not match the current index is recomputed).
2. Head only (LayerNorm -> Linear), AdamW (probe.lr, probe.weight_decay), masked BCE with census
   positive weights, up to probe.epochs epochs, early stopping on validation macro-F1 (geometric
   labels, threshold 0.5) with probe.early_stopping_patience.
3. The best head is evaluated on val and the test units against both label sets; predictions are
   saved to <run_dir>/predictions/.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from analysis.metrics import evaluate
from data.dataset import QueryDataset, loader
from data.queries import UNITS, index_path, vocabulary
from experiments.run import REPO_ROOT
from models.backbone import load_backbone
from models.common import autocast, device_of, log_epoch, metrics_for, save_predictions
from models.head import Head, masked_bce, pos_weights


def feature_path(cfg: dict, backbone_name: str, ds: QueryDataset) -> Path:
    tag = backbone_name.replace("/", "__")
    return (REPO_ROOT / cfg["paths"]["cache"] / "features" / tag / ds.cache_tag
            / (index_path(cfg, ds.unit).stem + ".npz"))


@torch.no_grad()
def extract(backbone, ds: QueryDataset, cfg: dict, device: torch.device) -> np.ndarray:
    path = feature_path(cfg, backbone.name, ds)
    qids = np.array([r["qid"] for r in ds.rows])
    if path.exists():
        z = np.load(path)
        if len(z["qid"]) == len(qids) and (z["qid"] == qids).all():
            return z["feats"].astype(np.float32)
    dl = loader(ds, cfg["probe"].get("extract_batch_size", 32), shuffle=False, num_workers=cfg["data"]["num_workers"])
    feats, t0 = [], time.time()
    for x, *_ in dl:
        with autocast(device, cfg):
            feats.append(backbone(x.to(device))["pooled"].float().cpu().numpy())
    feats = np.concatenate(feats) if feats else np.zeros((0, backbone.out_dim), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")      # atomic: parallel runs share the cache
    np.savez(tmp, feats=feats.astype(np.float16), qid=qids)
    os.replace(tmp, path)
    print(f"  features {ds.unit:16s} {feats.shape} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return feats


def labels_of(ds: QueryDataset) -> dict[str, np.ndarray]:
    return {"geo": np.array([r["geo"] for r in ds.rows], np.float32),
            "asr": np.array([r["asr"] for r in ds.rows], np.float32),
            "mask": np.array([r["mask"] for r in ds.rows], np.float32),
            "qid": np.array([r["qid"] for r in ds.rows])}


def train_head(feats: dict, labels: dict, cfg: dict, device: torch.device, run_dir: Path, vocab: list[str]):
    p = cfg["probe"]
    torch.manual_seed(cfg["seed"])
    X = torch.tensor(feats["train"], device=device)
    Y = torch.tensor(labels["train"]["geo"], device=device)
    M = torch.tensor(labels["train"]["mask"], device=device)
    Xv = torch.tensor(feats["val"], device=device)
    head = Head(X.shape[1], len(vocab), cfg["model"].get("head_dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    pw = pos_weights(labels["train"]["geo"], labels["train"]["mask"]).to(device)
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, p["epochs"] + 1):
        head.train()
        perm = torch.randperm(len(X), generator=g)
        total = 0.0
        for i in range(0, len(X), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]].to(device)
            loss = masked_bce(head(X[idx]), Y[idx], M[idx], pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head(Xv)).cpu().numpy()
        m = evaluate(pv, labels["val"]["geo"], labels["val"]["mask"], vocab)
        log_epoch(run_dir, {"epoch": epoch, "train_loss": total / max(len(X), 1), "val_macro_f1": m["macro_f1"],
                            "val_f1": {r: v["f1"] for r, v in m["per_relation"].items()}, "lr": p["lr"]})
        if m["macro_f1"] > best:
            best, best_epoch, bad = m["macro_f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= p["early_stopping_patience"]:
                break
    head.load_state_dict(best_state)
    return head, best, best_epoch, epoch


def run(cfg: dict, run_dir: Path, backbone=None) -> dict:
    t_start = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device_of(cfg)
    vocab = vocabulary(cfg)
    backbone = (backbone or load_backbone(cfg)).to(device).eval()
    print(f"probe: backbone {backbone.name} on {device}; normalisation {getattr(backbone, 'load_info', {}).get('normalisation')}", flush=True)
    loaded, load_info = backbone.name, getattr(backbone, "load_info", {})
    feats, labels = {}, {}
    for unit in UNITS:
        ds = QueryDataset(cfg, unit, train=False)
        feats[unit] = extract(backbone, ds, cfg, device)
        labels[unit] = labels_of(ds)
    del backbone
    head, best, best_epoch, last_epoch = train_head(feats, labels, cfg, device, run_dir, vocab)
    torch.save({"head": head.state_dict(), "vocab": vocab, "in_dim": feats["train"].shape[1]},
               Path(run_dir) / "probe_head.pt")
    results = {}
    head.eval()
    for unit in [u for u in UNITS if u != "train"]:
        with torch.no_grad():
            probs = torch.sigmoid(head(torch.tensor(feats[unit], device=device))).cpu().numpy()
        pred = {"probs": probs, **labels[unit]}
        save_predictions(pred, run_dir, unit)
        results[unit] = metrics_for(pred, vocab)
        print(f"  {unit:16s} macro-F1 geometric {results[unit]['geometric']['macro_f1']:.3f}  "
              f"asserted {results[unit]['asserted']['macro_f1']:.3f}", flush=True)
    return {"backbone": cfg["model"]["backbone"], "backbone_loaded": loaded, "backbone_info": load_info,
            "best_epoch": best_epoch, "epochs_run": last_epoch, "val_macro_f1": round(best, 4),
            "n_train": int(len(feats["train"])), "results": results, "wall_s": round(time.time() - t_start, 1)}
