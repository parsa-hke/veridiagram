"""Full fine-tune with layer-wise LR decay, cosine schedule, AMP and EMA (plan §6.4).

Dispatched by experiments.run (mode: finetune, also used by scaling and self-training).

Optimiser: AdamW; head LR `finetune.head_lr` (1e-3); backbone LR `finetune.backbone_lr` (2e-5) for
the last transformer block and the final norm, multiplied by `finetune.llrd` (0.8) per block going
down (block L-1-k gets 2e-5 * 0.8^k, the embeddings 2e-5 * 0.8^L); weight decay
`finetune.weight_decay` (0.05) except on biases, norms and tokens. Cosine schedule with
`warmup_epochs` of linear warm-up (per optimiser step). Batch `batch_size` with
`grad_accum_steps` accumulation, gradient clipping at 1.0, bfloat16 autocast (float16 with a
gradient scaler on GPUs without bf16). EMA of the weights
(`ema_decay`, with warm-up) is evaluated on val after every epoch; the best EMA weights by val
macro-F1 (geometric labels) are saved to <run_dir>/checkpoints/best.pt (in `save_dtype`, bfloat16 by
default: ~0.6 GB for ViT-L). With `finetune.save_last`,
<run_dir>/checkpoints/last.pt holds everything needed to resume (plan §0.6: Colab disconnects);
a run started on a run dir that has it resumes from the next epoch; it is deleted when the run
finishes unless `finetune.keep_last`. Per epoch, history.jsonl gets
the train loss, val macro-F1, per-class F1 and learning rates (end of epoch and peak).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analysis.metrics import evaluate
from data.dataset import QueryDataset, loader
from data.queries import vocabulary
from models.backbone import Backbone, load_backbone
from models.common import EMA, autocast, cosine_with_warmup, device_of, evaluate_units, log_epoch, predict
from models.head import Head, masked_bce, pos_weights


class Classifier(nn.Module):
    def __init__(self, backbone: Backbone, n_out: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.head = Head(backbone.out_dim, n_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)["pooled"])


def param_groups(model: Classifier, f: dict) -> list[dict]:
    """LLRD groups: (lr, weight decay) per backbone depth, plus the head."""
    L = model.backbone.num_layers
    groups: dict[tuple, dict] = {}
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        depth = model.backbone.layer_depth(name)
        lr = f["backbone_lr"] * f["llrd"] ** max(0, L - depth)
        wd = 0.0 if p.ndim <= 1 or "token" in name else f["weight_decay"]
        g = groups.setdefault((round(lr, 15), wd), {"params": [], "lr": lr, "weight_decay": wd, "depth": depth})
        g["params"].append(p)
    for name, p in model.head.named_parameters():
        wd = 0.0 if p.ndim <= 1 else f["weight_decay"]
        g = groups.setdefault(("head", wd), {"params": [], "lr": f["head_lr"], "weight_decay": wd, "depth": "head"})
        g["params"].append(p)
    return list(groups.values())


def _lrs(opt) -> dict:
    head = [g["lr"] for g in opt.param_groups if g.get("depth") == "head"]
    bb = [g["lr"] for g in opt.param_groups if g.get("depth") != "head"]
    return {"head": head[0] if head else None, "backbone_top": max(bb) if bb else None,
            "backbone_bottom": min(bb) if bb else None}


def run(cfg: dict, run_dir: Path, backbone: Backbone | None = None) -> dict:
    t_start = time.time()
    f = cfg["finetune"]
    device = device_of(cfg)
    vocab = vocabulary(cfg)
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone = backbone or load_backbone(cfg)
    loaded = backbone.name
    model = Classifier(backbone, len(vocab), cfg["model"].get("head_dropout", 0.0)).to(device)
    if f.get("init_checkpoint"):
        sd = torch.load(f["init_checkpoint"], map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=True)

    ds = QueryDataset(cfg, "train")
    gen = torch.Generator().manual_seed(cfg["seed"])
    dl = loader(ds, f["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], generator=gen)
    val_ds = QueryDataset(cfg, "val", train=False)
    val_dl = loader(val_ds, cfg["eval"].get("batch_size", f["batch_size"] * 2), shuffle=False,
                    num_workers=cfg["data"]["num_workers"])
    pw = pos_weights(np.array([r["geo"] for r in ds.rows]), np.array([r["mask"] for r in ds.rows])).to(device)

    accum = max(1, f.get("grad_accum_steps", 1))
    max_steps = f.get("max_steps_per_epoch")
    batches = len(dl) if not max_steps else min(len(dl), max_steps * accum)
    steps_per_epoch = math.ceil(batches / accum)
    opt = torch.optim.AdamW(param_groups(model, f), betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(int(f["warmup_epochs"] * steps_per_epoch), f["epochs"] * steps_per_epoch))
    ema = EMA(model, f["ema_decay"])
    # float16 autocast needs loss scaling (bfloat16 does not): GPUs without bf16 (V100, T4) use it
    fp16 = f.get("amp", True) and f.get("amp_dtype") == "float16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    start, best, best_epoch = 1, -1.0, 0
    last = ckpt_dir / "last.pt"
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, best_epoch = st["epoch"] + 1, st["best"], st["best_epoch"]
        gen.set_state(st["gen"])
        print(f"finetune: resumed from epoch {st['epoch']} (best {best:.3f} at {best_epoch})", flush=True)
    print(f"finetune: {loaded} on {device}; {len(ds)} train queries, {steps_per_epoch} steps/epoch x "
          f"{f['epochs']} epochs; {len(opt.param_groups)} param groups", flush=True)

    for epoch in range(start, f["epochs"] + 1):
        t0 = time.time()
        ds.set_epoch(epoch)
        model.train()
        total, n = 0.0, 0
        peak = _lrs(opt)
        opt.zero_grad(set_to_none=True)
        for i, (x, g, _, meta) in enumerate(dl):
            if i >= batches:
                break
            x, g, m = x.to(device, non_blocking=True), g.to(device), meta["mask"].to(device)
            with autocast(device, cfg):
                logits = model(x)
            loss = masked_bce(logits, g, m, pw)
            scaler.scale(loss / accum).backward()
            total += float(loss.detach()) * len(x)
            n += len(x)
            if (i + 1) % accum == 0 or i + 1 == batches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                cur = _lrs(opt)
                peak = {k: max(v, cur[k]) if v is not None else cur[k] for k, v in peak.items()}
                opt.zero_grad(set_to_none=True)
                ema.update(model)
        pred = predict(lambda x: ema.model(x), val_dl, device, cfg)
        mval = evaluate(pred["probs"], pred["geo"], pred["mask"], vocab)
        rec = {"epoch": epoch, "train_loss": total / max(n, 1), "val_macro_f1": mval["macro_f1"],
               "val_f1": {r: v["f1"] for r, v in mval["per_relation"].items()}, "lr": _lrs(opt), "lr_peak": peak,
               "seconds": round(time.time() - t0, 1)}
        log_epoch(run_dir, rec)
        print(f"  epoch {epoch}/{f['epochs']}  loss {rec['train_loss']:.4f}  val macro-F1 {mval['macro_f1']:.3f}  "
              f"peak lr head {peak['head']:.2e} top {peak['backbone_top']:.2e}  ({rec['seconds']:.0f}s)", flush=True)
        if mval["macro_f1"] > best:
            best, best_epoch = mval["macro_f1"], epoch
            dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[f.get("save_dtype", "bfloat16")]
            sd = {k: v.to(dt) if v.is_floating_point() else v for k, v in ema.model.state_dict().items()}
            torch.save({"model": sd, "epoch": epoch, "val_macro_f1": best, "backbone": loaded,
                        "vocab": vocab}, ckpt_dir / "best.pt")
        if f.get("save_last", True):
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                        "gen": gen.get_state()}, last)

    if last.exists() and not f.get("keep_last", False):
        last.unlink()                                      # finished: the resume state (~5 GB for ViT-L) is not needed
    ema.model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    ema.model.to(device).eval()
    results = evaluate_units(lambda x: ema.model(x), cfg, device, run_dir)
    return {"backbone": cfg["model"]["backbone"], "backbone_loaded": loaded,
            "backbone_info": getattr(backbone, "load_info", {}), "best_epoch": best_epoch,
            "val_macro_f1": round(best, 4), "n_train": len(ds), "steps_per_epoch": steps_per_epoch,
            "results": results, "wall_s": round(time.time() - t_start, 1)}
