"""Loader: PGPS9K (Zhang et al., 2023) raw files -> `Diagram` (plan §2.4).

Layout under <raw>/pgps9k/PGPS9K/: diagram_annotation.json (4,000 diagrams keyed "img_<n>", PGDP
primitive format), Diagram/img_<n>.png (clean), Diagram_Visual/img_<n>.png (annotation overlay),
PGPS9K/{train,test}.json and Geometry3K/{train,test}.json (problems: text, structural and semantic
clauses, solution program; keyed "prob_<n>", each pointing at a diagram). Problems, not diagrams,
are split, and 701 diagrams are used by both train and test problems, so diagrams get split "all";
the problem splits and source books are kept in `meta`. In this project PGPS9K is the unlabelled
pool for Part B.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from data.loaders.pgdp_format import parse_record
from data.schema import Diagram


def load(raw_dir: str | Path, split: str | None = None, ids: Iterable[str] | None = None,
         limit: int | None = None) -> list[Diagram]:
    root = Path(raw_dir) / "pgps9k" / "PGPS9K"
    if split not in (None, "all"):
        return []
    usage = defaultdict(lambda: {"problem_splits": set(), "books": set(), "n_problems": 0})
    for sp in ("train", "test"):
        for prob in json.loads((root / "PGPS9K" / f"{sp}.json").read_text()).values():
            u = usage[prob["diagram"]]
            u["problem_splits"].add(sp)
            u["books"].add(prob.get("book", ""))
            u["n_problems"] += 1
    recs = json.loads((root / "diagram_annotation.json").read_text())
    want = set(map(str, ids)) if ids is not None else None
    out = []
    for key, rec in recs.items():
        if want is not None and key not in want:
            continue
        d = parse_record(rec, diagram_id=f"pgps9k:{key}", source="pgps9k", split="all",
                         image_path=f"pgps9k/PGPS9K/Diagram/{rec['file_name']}")
        u = usage.get(rec["file_name"])
        d.meta.update({"problem_splits": sorted(u["problem_splits"]) if u else [],
                       "books": sorted(u["books"]) if u else [], "n_problems": u["n_problems"] if u else 0})
        out.append(d)
        if limit and len(out) >= limit:
            break
    return out
