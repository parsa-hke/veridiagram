"""Evaluate a fine-tune checkpoint on every evaluation unit against both label sets (plan §7.4).

Dispatched by experiments.run (mode: evaluate) with `eval.checkpoint` pointing at a best.pt written
by models.finetune. Predictions go to <run_dir>/predictions/<unit>.npz (qid, probs, geo, asr, mask);
the not-to-scale subset and other slices are computed from them in Phase 8 (flags live in the
processed diagrams). Every fine-tune already runs this at its end; this mode re-evaluates a saved
checkpoint, e.g. after an evaluation unit changes or for the Part B teacher.
"""

from __future__ import annotations

from pathlib import Path

import torch

from data.queries import vocabulary
from models.backbone import load_backbone
from models.common import device_of, evaluate_units
from models.finetune import build_classifier


def run(cfg: dict, run_dir: Path) -> dict:
    ckpt = cfg["eval"].get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"eval.checkpoint not found: {ckpt}")
    device = device_of(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    backbone = load_backbone(cfg)
    if sd.get("backbone") and sd["backbone"] != backbone.name:
        raise ValueError(f"checkpoint was trained on {sd['backbone']}, loaded {backbone.name}")
    model = build_classifier(cfg, backbone, len(vocabulary(cfg)))
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    results = evaluate_units(model, cfg, device, Path(run_dir))
    return {"checkpoint": str(ckpt), "checkpoint_epoch": sd.get("epoch"), "backbone_loaded": backbone.name,
            "results": results}
