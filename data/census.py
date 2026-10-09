"""Relation census: asserted positives per relation type per source and split (plan §2.6).

Writes reports/tables/relation_census.csv. Columns per source/split give raw counts (every image)
and, where data/processed/figure_clusters.csv exists, counts over unique figures (duplicate images
counted once), because Geometry3K reuses figures across problems. The Tier 2 rule is applied to
the training source named in §0.2 #9 (Geometry3K train), unique figures.

    python -m data.census
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

from data.schema import RELATION_TYPES, read_jsonl
from experiments.run import REPO_ROOT, load_config

SOURCES = ("geometry3k", "pgdp5k", "pgps9k")


def census(proc, sources=SOURCES) -> tuple[list[str], dict[str, Counter]]:
    cluster = {}
    path = proc / "figure_clusters.csv"
    if path.exists():
        with open(path) as f:
            cluster = {r["id"]: r["cluster"] for r in csv.DictReader(f)}
    cols: dict[str, Counter] = defaultdict(Counter)
    seen: dict[str, set] = defaultdict(set)
    for src in sources:
        diagrams = list(read_jsonl(proc / f"{src}.jsonl"))
        multi_split = len({d.split for d in diagrams}) > 1
        for d in diagrams:
            c = Counter(r.type for r in d.asserted)
            c["_diagrams"] = 1
            for col in (f"{src}_{d.split}", f"{src}_total")[: 2 if multi_split else 1]:
                cols[col].update(c)
                fig = cluster.get(d.id)
                if fig is not None and fig not in seen[col]:
                    seen[col].add(fig)
                    cols[f"{col}_unique"].update(c)
    order = [k for src in sources for k in sorted(cols) if k.startswith(src)]
    return order, cols


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    order, cols = census(proc)
    tier1 = [t for t in cfg["relations"]["tier1"] if t != "none"]
    tier2 = cfg["relations"]["tier2"]
    thresh = cfg["relations"]["tier2_min_positives"]
    rule_col = "geometry3k_train_unique" if "geometry3k_train_unique" in cols else "geometry3k_train"

    out = REPO_ROOT / cfg["paths"]["tables"] / "relation_census.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relation", "tier", *order, f"tier2_rule_{rule_col}>={thresh}"])
        w.writerow(["_diagrams", "", *[cols[c]["_diagrams"] for c in order], ""])
        for t in RELATION_TYPES:
            tier = "1" if t in tier1 else "2" if t in tier2 else ""
            verdict = ("include" if cols[rule_col][t] >= thresh else "exclude") if tier == "2" else ""
            w.writerow([t, tier, *[cols[c][t] for c in order], verdict])
    print(open(out).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
