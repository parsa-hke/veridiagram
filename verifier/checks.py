"""One function per relation, each returning (is_true, margin) (plan §3.2).

`margin` is the raw quantity before thresholding, so its distribution can be plotted.
Tolerances come from the `tolerances` config section, fixed in Phase 3.4.
"""

from __future__ import annotations
