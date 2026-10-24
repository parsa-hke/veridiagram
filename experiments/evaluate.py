"""Evaluate a saved checkpoint on every evaluation unit against both label sets (plan §7.4, §0.2 #10).

Dispatched by experiments.run (mode: evaluate). Predictions go to <run_dir>/predictions/<unit>.npz
(qid, probs, geo, asr, mask); the not-to-scale subset and other slices are computed from them.

Two ways to name the model:
- `eval.checkpoint`: a best.pt written by models.finetune, evaluated under this config.
- `eval.source_config` (configs/eval_square.yaml, plan §0.2 #10 note): the config of a finished
  experiment, e.g. configs/finetune_synth10k.yaml. The run evaluated is <runs>/<source name>_s<seed>
  (`eval.source_run` overrides; with --smoke the source is the smoke run of that config). Its model,
  input, overlay, relation, tolerance and data settings come from the source config, so the
  evaluation rows are the source run's own; this config adds only the evaluation settings
  (`eval.pad`, `eval.batch_size`), paths, device and data loading. The checkpoint is
  <source run>/checkpoints/best.pt, or probe_head.pt for a probe (its features are re-extracted).
  When the source run dir holds its config.yaml, the settings that define the evaluation rows and
  the model are compared with it and a difference is an error.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

from data.queries import vocabulary
from experiments.run import REPO_ROOT, SMOKE_CONFIG, apply_overlay, load_config
from models.backbone import load_backbone
from models.common import device_of, evaluate_units

# settings of the source run that its predictions depend on (checked against its saved config.yaml)
MODEL_KEYS = {"model": ("backbone", "pooling", "query_input", "head_dropout"), "input": None, "overlay": None,
              "relations": None, "tolerances": None,
              "data": ("train_sources", "val_sources", "test_sets", "split_seed", "split_fractions",
                       "max_diagrams", "max_queries_per_diagram", "train_none_ratio", "index_seed",
                       "max_queries_per_unit", "synthetic_train_size"),
              "eval": ("max_queries",)}
LOCAL_KEYS = ("paths", "device", "deterministic")              # taken from this config, not the source


def source_config(cfg: dict) -> dict:
    """The effective config of a source-run evaluation: the source experiment's settings plus this
    config's evaluation settings, paths, device and data loading; name, seed and mode stay this run's."""
    src = load_config(REPO_ROOT / cfg["eval"]["source_config"])
    if cfg["name"].startswith("smoke"):
        name = src.get("name", "run")
        src = apply_overlay(src, SMOKE_CONFIG)
        src["name"] = name if name.startswith("smoke") else f"smoke_{name}"
    eff = copy.deepcopy(src)
    eff["source_mode"], eff["source_name"] = src["mode"], src["name"]
    for k in LOCAL_KEYS:
        if k in cfg:
            eff[k] = copy.deepcopy(cfg[k])
    eff["data"]["num_workers"] = cfg["data"]["num_workers"]
    eff["data"]["cache_overlays"] = cfg["data"].get("cache_overlays", True)
    eff["eval"]["pad"] = cfg["eval"].get("pad", "batch")
    eff["eval"]["batch_size"] = cfg["eval"]["batch_size"]
    for k in ("name", "seed", "mode", "run_dir", "device_resolved"):
        if k in cfg:
            eff[k] = cfg[k]
    eff["eval"]["source_config"] = cfg["eval"]["source_config"]
    eff["eval"]["source_run"] = cfg["eval"].get("source_run") or f"{src['name']}_s{cfg['seed']}"
    return eff


def source_mismatches(eff: dict, saved: dict) -> list[str]:
    """Settings in MODEL_KEYS where the effective config differs from the source run's saved config. A
    key the saved config lacks was added after that run (e.g. model.query_input, overlay.enabled for
    the lab runs); it matches when the value here is base.yaml's default, which kept the old behaviour."""
    base = load_config(REPO_ROOT / "configs" / "base.yaml", env_overlays=False)
    out = []
    for sec, keys in MODEL_KEYS.items():
        a, b, d = eff.get(sec, {}), saved.get(sec, {}), base.get(sec, {})
        for k in keys or sorted(set(a) | set(b)):
            ref = b[k] if k in b else d.get(k)
            if a.get(k) != ref:
                out.append(f"{sec}.{k}: {a.get(k)!r} here, {b.get(k, '(absent)')!r} in the source run")
    return out


def runs_root(cfg: dict) -> Path:
    p = Path(cfg["paths"]["runs"])
    return p if p.is_absolute() else REPO_ROOT / p


def checkpoint_of(eff: dict) -> Path:
    if eff["eval"].get("checkpoint"):
        return Path(eff["eval"]["checkpoint"])
    d = runs_root(eff) / eff["eval"]["source_run"]
    return d / ("probe_head.pt" if eff["source_mode"] == "probe" else "checkpoints/best.pt")


def evaluate_finetune(cfg: dict, ckpt: Path, run_dir: Path, backbone=None) -> dict:
    from models.finetune import build_classifier
    device = device_of(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    backbone = backbone or load_backbone(cfg)
    if sd.get("backbone") and sd["backbone"] != backbone.name:
        raise ValueError(f"checkpoint was trained on {sd['backbone']}, loaded {backbone.name}")
    model = build_classifier(cfg, backbone, len(vocabulary(cfg)))
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    results = evaluate_units(model, cfg, device, Path(run_dir))
    return {"checkpoint_epoch": sd.get("epoch"), "backbone_loaded": backbone.name,
            "backbone_info": getattr(backbone, "load_info", {}), "results": results}


def run(cfg: dict, run_dir: Path, backbone=None) -> dict:
    run_dir = Path(run_dir)
    if not cfg["eval"].get("source_config"):
        ckpt = cfg["eval"].get("checkpoint")
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"eval.checkpoint not found: {ckpt}")
        return {"checkpoint": str(ckpt), "pad": cfg["eval"].get("pad", "batch"),
                **evaluate_finetune(cfg, Path(ckpt), run_dir, backbone)}

    eff = source_config(cfg)
    ckpt = checkpoint_of(eff)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint of {eff['eval']['source_run']} not found: {ckpt}")
    saved = runs_root(eff) / eff["eval"]["source_run"] / "config.yaml"
    checked = saved.exists()
    if checked:
        bad = source_mismatches(eff, yaml.safe_load(saved.read_text()))
        if bad:
            raise ValueError(f"{eff['source_name']} config differs from the run {saved.parent.name}: " + "; ".join(bad))
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "effective_config.yaml", "w") as f:
        yaml.safe_dump(eff, f, sort_keys=False)
    print(f"evaluate: {eff['eval']['source_run']} ({eff['source_mode']}) from {ckpt}, pad {eff['eval']['pad']}; "
          f"source config {'checked against ' + str(saved) if checked else 'not checked (no saved config.yaml)'}",
          flush=True)
    if eff["source_mode"] == "probe":
        from models.probe import evaluate_head
        out = evaluate_head(eff, run_dir, ckpt, backbone)
    else:
        out = evaluate_finetune(eff, ckpt, run_dir, backbone)
    return {"source_run": eff["eval"]["source_run"], "source_config": eff["eval"]["source_config"],
            "source_mode": eff["source_mode"], "source_config_checked": checked, "checkpoint": str(ckpt),
            "pad": eff["eval"]["pad"], **out}
