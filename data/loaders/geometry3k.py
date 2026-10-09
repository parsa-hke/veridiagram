"""Loader: Geometry3K (Lu et al., 2021; Inter-GPS) raw files -> `Diagram` (plan §2.4).

Field names are documented in data/README.md after inspection (task 2.2).
"""

from __future__ import annotations

from pathlib import Path


def load(raw_dir: str | Path, split: str | None = None) -> list:
    """Return a list of `Diagram` objects for `split` (or all splits)."""
    raise NotImplementedError("Phase 2.4")
