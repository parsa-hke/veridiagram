"""Loader: PGDP5K (Zhang et al., 2022) raw files -> `Diagram` (plan §2.4).

Layout under <raw>/pgdp5k/PGDP5K/: annotations/{train,val,test}.json (dict keyed by image id,
PGDP primitive format, see data/loaders/pgdp_format.py), images {train,val,test}/<id>.png, and
our_diagram_logic_forms_annot.json (the authors' derived logic forms, used only to validate the
parser). Note the shipped readme.txt swaps the val/test sizes: val has 500, test has 1000.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from data.loaders.pgdp_format import parse_record
from data.schema import Diagram

SPLITS = ("train", "val", "test")


def load(raw_dir: str | Path, split: str | None = None, ids: Iterable[str] | None = None,
         limit: int | None = None) -> list[Diagram]:
    root = Path(raw_dir) / "pgdp5k" / "PGDP5K"
    want = set(map(str, ids)) if ids is not None else None
    out = []
    for sp in ([split] if split else SPLITS):
        recs = json.loads((root / "annotations" / f"{sp}.json").read_text())
        for key, rec in recs.items():
            if want is not None and key not in want:
                continue
            out.append(parse_record(rec, diagram_id=f"pgdp5k:{key}", source="pgdp5k", split=sp,
                                    image_path=f"pgdp5k/PGDP5K/{sp}/{rec['file_name']}"))
            if limit and len(out) >= limit:
                return out
    return out
