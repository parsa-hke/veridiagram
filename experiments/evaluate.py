"""Evaluate a checkpoint on every test set and the not-to-scale subset, against both
label sets (plan §7.4). Dispatched by experiments.run (mode: evaluate)."""

from __future__ import annotations

from pathlib import Path


def run(cfg: dict, run_dir: Path) -> dict:
    raise NotImplementedError("Phase 7.4")
