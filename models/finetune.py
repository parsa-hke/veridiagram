"""Full fine-tune with layer-wise LR decay, cosine schedule, AMP and EMA (plan §6.4).

Dispatched by experiments.run (mode: finetune).
"""

from __future__ import annotations

from pathlib import Path


def run(cfg: dict, run_dir: Path) -> dict:
    raise NotImplementedError("Phase 6.4")
