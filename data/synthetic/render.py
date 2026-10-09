"""Matplotlib textbook-style renderer for synthetic diagrams (plan §4.2).

Draws point labels, right-angle markers and tick marks for injected relations,
so synthetic data has an `asserted` label set too.
"""

from __future__ import annotations


def render(diagram, out_path, cfg: dict, seed: int) -> None:
    """Render `diagram` to `out_path` with style knobs from `synthetic.render`."""
    raise NotImplementedError("Phase 4.2")
