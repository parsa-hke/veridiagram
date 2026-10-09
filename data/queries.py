"""Enumerate candidate n-ary queries per diagram (plan §5.1-5.2).

Segment pairs, point-circle pairs, segment-circle pairs (+ Tier 2 tuples), capped
per diagram with all positives kept; multi-label vectors for both label sets.
"""

from __future__ import annotations


def enumerate_queries(diagram, cfg: dict, seed: int) -> list:
    raise NotImplementedError("Phase 5.1")
