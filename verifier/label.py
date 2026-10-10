"""Geometric label generation (plan §3.7) and not-to-scale flags (plan §3.6).

For every diagram of every processed set, enumerate all candidate queries (data.queries) and run
the checks with the calibrated tolerances. Positives go to `Diagram.geometric` (source
"geometric", with margin); every candidate, positive or not, goes to
data/processed/queries/<set>.csv.gz with its margin and both labels.

Not-to-scale: a diagram is flagged when any asserted Tier 1 relation fails its check by a large
margin: parallel/perpendicular off by more than `not_to_scale.angle_deg` degrees, equal_length /
on_circle / tangent relative error above `not_to_scale.rel_length`. Flagged diagrams stay in the
data (`flags.not_to_scale`, `flags.not_to_scale_relations`). Collinear is not used for the flag.
Table: reports/tables/not_to_scale.csv.

    python -m verifier.label [--sets geometry3k_figures pgdp5k_heldout ...]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from data.queries import asserted_lookup, enumerate_queries
from data.schema import Diagram, Relation, read_jsonl, write_jsonl
from experiments.run import REPO_ROOT, load_config
from verifier.checks import evaluate
from verifier.primitives import bbox_diagonal

SETS = ("geometry3k", "pgdp5k", "pgps9k", "geometry3k_figures", "pgdp5k_heldout", "pgps9k_pool")
NTS_ANGLE = ("parallel", "perpendicular")
NTS_LENGTH = ("equal_length", "on_circle", "tangent")
QUERY_FIELDS = ["diagram_id", "split", "kind", "refs", "carrier_key", "relation", "margin", "geometric", "asserted"]


def not_to_scale(d: Diagram, nts: dict) -> list[dict]:
    """Asserted Tier 1 relations that fail their check by a large margin."""
    out = []
    for r in d.asserted:
        if r.type not in NTS_ANGLE + NTS_LENGTH:
            continue
        _, m = evaluate(d, r.type, r.args)
        if math.isnan(m):
            continue
        if r.type in NTS_ANGLE:
            dev = math.degrees(math.asin(min(m, 1.0)))
            if dev > nts["angle_deg"]:
                out.append({"type": r.type, "args": r.args, "deviation_deg": round(dev, 2)})
        elif m > nts["rel_length"]:
            out.append({"type": r.type, "args": r.args, "relative_error": round(m, 4)})
    return out


def label_diagram(d: Diagram, tol: dict, nts: dict, tier2: tuple[str, ...]) -> list[dict]:
    """Fill d.geometric and the not-to-scale flags in place; return one row per (query, relation)."""
    scale = bbox_diagonal(d.points.values())
    is_asserted = asserted_lookup(d)
    rows, geometric = [], []
    for q in enumerate_queries(d, tier2):
        for rel in q.relations:
            ok, m = evaluate(d, rel, q.refs, tol, scale=scale)
            a = is_asserted(q, rel)
            rows.append({"diagram_id": d.id, "split": d.split, "kind": q.kind, "refs": q.key(),
                         "carrier_key": q.carrier_key, "relation": rel,
                         "margin": "" if math.isnan(m) else f"{m:.6g}", "geometric": int(bool(ok)), "asserted": int(a)})
            if ok:
                geometric.append(Relation(rel, q.refs, source="geometric", margin=float(m)))
    d.geometric = geometric
    bad = not_to_scale(d, nts)
    d.flags["not_to_scale"] = bool(bad)
    if bad:
        d.flags["not_to_scale_relations"] = bad
    else:
        d.flags.pop("not_to_scale_relations", None)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tol, nts = cfg["tolerances"], cfg["not_to_scale"]
    missing = [k for k in ("parallel_deg", "perpendicular_deg", "equal_length_rel", "on_circle_rel", "tangent_rel")
               if tol.get(k) is None]
    if missing:
        raise SystemExit(f"tolerances not calibrated: {missing}; run python -m verifier.tolerance first")
    tier2 = tuple(cfg["relations"].get("tier2_included", []))
    proc = REPO_ROOT / cfg["paths"]["processed"]
    (proc / "queries").mkdir(parents=True, exist_ok=True)

    report = []
    for name in args.sets:
        path = proc / f"{name}.jsonl"
        if not path.exists():
            continue
        diagrams = list(read_jsonl(path))
        geo, asr, n_rows = Counter(), Counter(), 0
        with gzip.open(proc / "queries" / f"{name}.csv.gz", "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=QUERY_FIELDS)
            w.writeheader()
            for d in diagrams:
                rows = label_diagram(d, tol, nts, tier2)
                w.writerows(rows)
                n_rows += len(rows)
                for r in rows:
                    geo[r["relation"]] += r["geometric"]
                    asr[r["relation"]] += r["asserted"]
        write_jsonl(diagrams, path)
        flagged = [d for d in diagrams if d.flags.get("not_to_scale")]
        by_type = Counter(x["type"] for d in flagged for x in d.flags["not_to_scale_relations"])
        report.append({"set": name, "diagrams": len(diagrams), "not_to_scale": len(flagged),
                       "pct": round(100 * len(flagged) / max(len(diagrams), 1), 1),
                       **{f"by_{t}": by_type.get(t, 0) for t in NTS_ANGLE + NTS_LENGTH}})
        print(f"{name:20s} {len(diagrams):5d} diagrams, {n_rows:7d} query-relations; geometric+ {dict(geo)}; "
              f"asserted+ {dict(asr)}; not_to_scale {len(flagged)} ({report[-1]['pct']}%)")

    out = REPO_ROOT / cfg["paths"]["tables"] / "not_to_scale.csv"
    rows_old = []
    if out.exists():
        with open(out) as f:
            rows_old = [r for r in csv.DictReader(f) if r["set"] not in {r["set"] for r in report}]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(sorted(rows_old + report, key=lambda r: SETS.index(r["set"]) if r["set"] in SETS else 99))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
