"""Measure training and inference throughput on this machine and estimate the queue's run time.

    python -m experiments.benchmark [--queue configs/queue_phase7.yaml] [--steps 30] [--gpus N]

For each input resolution used by the queue (448 and 224), on one GPU with the primary backbone:
  train  images/s over `--steps` optimiser steps of the real fine-tune loop (data loading,
         augmentation, overlay cache, bf16 autocast, gradient checkpointing, AdamW with LLRD)
  eval   images/s of the forward pass on the validation unit
  peak GPU memory
Then, for every job of the queue: train queries per epoch (from its query index if built, else
estimated from the real index plus queries per synthetic diagram), epochs, and the estimated hours
= epochs x (train / train rate + val / eval rate) + final evaluation of all units / eval rate. With
--gpus N the queue's wall time is estimated by assigning jobs in order to the first free GPU.
Writes runs/benchmark/benchmark.json. Timings include a cold overlay cache, so they are
conservative for later epochs. The DINOv2 ablation is estimated as 1.35x the 448 cost (1024 vs 784
patch tokens), not measured.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from data.dataset import QueryDataset, loader
from data.queries import UNITS, index_path, vocabulary
from experiments.queue import available_gpus, expand
from experiments.run import REPO_ROOT, load_config, apply_overrides
from models.backbone import load_backbone
from models.common import autocast, device_of
from models.finetune import Classifier, param_groups
from models.head import masked_bce

SYNTH_QUERIES_PER_DIAGRAM = 18.37      # from the 10k index (221,519 - 37,846) / 10,000; refined if built
DINOV2_FACTOR = 1.35


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(cfg: dict, backbone, steps: int) -> dict:
    device = device_of(cfg)
    model = Classifier(backbone, len(vocabulary(cfg))).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["finetune"]))
    ds = QueryDataset(cfg, "train")
    dl = loader(ds, cfg["finetune"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    n, t0, warm = 0, None, 3
    for i, (x, g, _, meta) in enumerate(dl):
        if i == warm:
            _sync(device)
            t0, n = time.time(), 0
        if i >= warm + steps:
            break
        x, g, m = x.to(device), g.to(device), meta["mask"].to(device)
        with autocast(device, cfg):
            loss = masked_bce(model(x), g, m)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        n += len(x)
    _sync(device)
    train_ips = n / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    vds = QueryDataset(cfg, "val", train=False)
    vdl = loader(vds, cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    model.eval()
    n, t0 = 0, None
    with torch.no_grad():
        for i, (x, *_rest) in enumerate(vdl):
            if i == 2:
                _sync(device)
                t0, n = time.time(), 0
            if i >= 2 + steps:
                break
            with autocast(device, cfg):
                model(x.to(device))
            n += len(x)
    _sync(device)
    eval_ips = n / (time.time() - t0)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"train_img_per_s": round(train_ips, 1), "eval_img_per_s": round(eval_ips, 1),
            "peak_mem_gb": None if mem is None else round(mem, 1)}


def _count(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path) as f:
        return sum(1 for _ in f)


def unit_sizes(cfg: dict) -> dict:
    out = {}
    for u in UNITS:
        n = _count(index_path(cfg, u))
        out[u] = n
    return out


def estimate_job(job: dict, rates: dict, real_train: int, eval_sizes: dict) -> dict:
    cfg = apply_overrides(load_config(job["config"]), job["set"])
    L = cfg["input"]["long_side"]
    r = dict(rates[L])
    if "dinov2" in cfg["model"]["backbone"]:
        r = {k: v / DINOV2_FACTOR for k, v in r.items() if k.endswith("_s")}
    synth = cfg["data"].get("synthetic_train_size", 0)
    built = _count(index_path(cfg, "train"))
    n_train = built if built else int(real_train + synth * SYNTH_QUERIES_PER_DIAGRAM)
    n_eval = sum(v for k, v in eval_sizes.items() if k != "train")
    if job["mode"] == "probe":
        hours = (n_train + n_eval) / r["eval_img_per_s"] / 3600 if job["seed"] == 0 else 0.05
        epochs = cfg["probe"]["epochs"]
    else:
        epochs = cfg["finetune"]["epochs"]
        per_epoch = n_train / r["train_img_per_s"] + eval_sizes["val"] / r["eval_img_per_s"]
        hours = (epochs * per_epoch + n_eval / r["eval_img_per_s"] + 120) / 3600
    return {"run": job["run"], "mode": job["mode"], "resolution": L, "train_queries": n_train,
            "train_queries_source": "index" if built else "estimate", "epochs": epochs, "hours": round(hours, 2)}


def makespan(hours: list[float], n_gpus: int) -> float:
    free = [0.0] * max(1, n_gpus)
    for h in hours:
        i = free.index(min(free))
        free[i] += h
    return max(free)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=str(REPO_ROOT / "configs" / "queue_phase7.yaml"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gpus", type=int, default=None, help="GPUs to plan for (default: all visible)")
    args = ap.parse_args(argv)
    base = load_config(REPO_ROOT / "configs" / "base.yaml")
    device = device_of(base)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    n_gpus = args.gpus or len([g for g in available_gpus(None) if g != "cpu"]) or 1
    print(f"benchmark on {gpu_name} ({device}); planning for {n_gpus} GPU(s)", flush=True)
    backbone = load_backbone(base)
    print(f"backbone: {backbone.name}", flush=True)

    jobs = expand(args.queue)
    resolutions = sorted({apply_overrides(load_config(j["config"]), j["set"])["input"]["long_side"] for j in jobs},
                         reverse=True)
    rates = {}
    for L in resolutions:
        cfg = copy.deepcopy(base)
        cfg["input"]["long_side"] = L
        t0 = time.time()
        rates[L] = measure(cfg, backbone, args.steps)
        print(f"  {L}px: train {rates[L]['train_img_per_s']} img/s, eval {rates[L]['eval_img_per_s']} img/s, "
              f"peak memory {rates[L]['peak_mem_gb']} GB ({time.time() - t0:.0f}s)", flush=True)

    sizes = unit_sizes(base)
    missing = [u for u, n in sizes.items() if n is None]
    if missing:
        raise SystemExit(f"query indexes missing for {missing}: run python -m data.queries first")
    est = [estimate_job(j, rates, sizes["train"], sizes) for j in jobs]
    print(f"\n{'run':32s} {'mode':9s} {'px':>4s} {'train q/epoch':>14s} {'epochs':>6s} {'hours':>7s}")
    for e in est:
        flag = "" if e["train_queries_source"] == "index" else " (est.)"
        print(f"{e['run']:32s} {e['mode']:9s} {e['resolution']:4d} {e['train_queries']:14,d}{flag:7s} "
              f"{e['epochs']:6d} {e['hours']:7.1f}")
    total = sum(e["hours"] for e in est)
    wall = makespan([e["hours"] for e in est], n_gpus)
    print(f"\ntotal GPU-hours {total:.1f}; wall time on {n_gpus} GPU(s) ~{wall:.1f} h ({wall / 24:.1f} days)")
    whatif = {}
    for ep in (30, 20, 15, 10):
        e2 = [estimate_job({**j, "set": j["set"] + [f"finetune.epochs={ep}"]}, rates, sizes["train"], sizes) for j in jobs]
        whatif[ep] = round(makespan([e["hours"] for e in e2], n_gpus), 1)
    print("wall time by fine-tune epochs: " + ", ".join(f"{ep} epochs ~{h} h" for ep, h in whatif.items())
          + "   (run with: python -m experiments.queue <queue> --set finetune.epochs=N)")
    out = REPO_ROOT / "runs" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(
        {"gpu": gpu_name, "n_gpus": n_gpus, "backbone": backbone.name, "rates": rates, "jobs": est,
         "total_gpu_hours": round(total, 1), "wall_hours": round(wall, 1), "wall_hours_by_epochs": whatif}, indent=2))
    print(f"-> {out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
