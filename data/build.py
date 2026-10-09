"""CLI: raw -> data/processed/<source>.jsonl, with per-source counts (plan §2.5).

    python -m data.build --config configs/base.yaml [--sources geometry3k pgdp5k] [--limit 50]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from data.loaders import geometry3k, pgdp5k, pgps9k
from data.schema import RELATION_TYPES, Diagram, write_jsonl
from experiments.run import REPO_ROOT, load_config

LOADERS = {"geometry3k": geometry3k.load, "pgdp5k": pgdp5k.load, "pgps9k": pgps9k.load}


def counts(diagrams: list[Diagram]) -> dict:
    rel = Counter(r.type for d in diagrams for r in d.asserted)
    return {
        "diagrams": len(diagrams),
        "points": sum(len(d.points) for d in diagrams),
        "segments": sum(len(d.segments) for d in diagrams),
        "lines": sum(len(d.lines) for d in diagrams),
        "circles": sum(len(d.circles) for d in diagrams),
        **{f"asserted_{t}": rel.get(t, 0) for t in RELATION_TYPES},
        "splits": dict(Counter(d.split for d in diagrams)),
        "flags": dict(Counter(k for d in diagrams for k in d.flags)),
    }


def build(cfg: dict, sources: list[str], limit: int | None = None) -> dict[str, dict]:
    raw = REPO_ROOT / cfg["paths"]["raw"]
    out_dir = REPO_ROOT / cfg["paths"]["processed"]
    report = {}
    for src in sources:
        t0 = time.time()
        diagrams = LOADERS[src](raw, limit=limit)
        write_jsonl(diagrams, out_dir / f"{src}.jsonl")
        report[src] = counts(diagrams)
        print(f"\n== {src}: {len(diagrams)} diagrams -> {out_dir / f'{src}.jsonl'} ({time.time() - t0:.1f}s)")
        for k, v in report[src].items():
            print(f"  {k:24s} {v}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--sources", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--limit", type=int, default=None, help="max diagrams per source (smoke)")
    args = ap.parse_args(argv)
    build(load_config(args.config), args.sources, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
