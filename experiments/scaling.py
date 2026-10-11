"""Part A: fine-tune on Geometry3K-figure train + the first `data.synthetic_train_size` synth_train
diagrams (plan §7.9). Dispatched by experiments.run (mode: scaling).

The training unit already includes the synthetic prefix (data.queries.unit_sources), so a scaling
run is a fine-tune with that config; this mode only checks the size and records it.
"""

from __future__ import annotations

from pathlib import Path

from models import finetune


def run(cfg: dict, run_dir: Path) -> dict:
    size = int(cfg["data"].get("synthetic_train_size", 0))
    allowed = {0, *cfg["scaling"]["sizes"]}
    if size not in allowed:
        raise ValueError(f"synthetic_train_size {size} not in scaling.sizes {sorted(allowed)}")
    out = finetune.run(cfg, run_dir)
    out["synthetic_train_size"] = size
    return out
