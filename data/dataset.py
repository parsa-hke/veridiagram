"""torch Dataset over (image, query) -> (image, label_geometric, label_asserted, meta) (plan §5.6)."""

from __future__ import annotations


class QueryDataset:
    def __init__(self, cfg: dict, split: str):
        raise NotImplementedError("Phase 5.6")
