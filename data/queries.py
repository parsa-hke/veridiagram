"""Enumerate candidate n-ary queries per diagram (plan §5.1-5.2; enumeration used from Phase 3.7).

Query kinds and the relations each carries:
  seg_pair    [seg, seg]          parallel, perpendicular, equal_length
                                  (parallel/perpendicular omitted when both lie on one carrier line;
                                  pairs on one line that overlap, e.g. AM inside AB, are skipped:
                                  highlighted together they look like one segment)
  pt_circle   [pt, circle]        on_circle   (the centre point itself is skipped)
  seg_circle  [seg, circle]       tangent
  pt_line     [pt P, pt A, pt B]  collinear   (A, B = extreme points of a carrier line; P not A, B)
Circles without centre or radius are skipped. Tier 2 kinds are included only for relations in
`relations.tier2_included`.

Carrier lines: a segment's carrier is the annotated collinear group containing both endpoints, or
the segment itself. parallel, perpendicular, tangent and collinear are properties of lines, so
`asserted_lookup` resolves asserted relations to carriers: two sub-segments of asserted-parallel
lines are an asserted-parallel query. equal_length and on_circle are matched exactly.

Index (plan §5.1-5.4, `python -m data.queries`): every candidate query of a dataset unit gets a
stable id (hash of kind + refs), a multi-label `geo` vector (verifier, training target) and `asr`
vector (annotation) over `vocabulary(cfg)`, an applicability `mask` (a relation the query kind
cannot have, e.g. on_circle for a segment pair, is masked, not a negative) and the margins. Then at
most `max_queries_per_diagram` per diagram (training: all positives kept; evaluation: uniform, so
the natural ratio is kept), and for training only, all-negative ("none") queries are downsampled to
`train_none_ratio` x the positive queries of the same kind (each relation belongs to one kind).
Written to data/processed/index/<unit>[.synth<k>][.n<max_diagrams>].jsonl.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from data.schema import Diagram, circle_ref, pt, seg

TIER1_KINDS = {"seg_pair": ("parallel", "perpendicular", "equal_length"),
               "pt_circle": ("on_circle",), "seg_circle": ("tangent",)}


@dataclass
class Query:
    kind: str
    refs: list
    relations: tuple[str, ...]
    carrier_key: str            # identical for queries that share carrier lines (dedupe for line relations)

    def key(self) -> str:
        return json.dumps(self.refs, separators=(",", ":"))


class Carriers:
    def __init__(self, d: Diagram):
        self.groups = [g for g in d.lines if len(g) >= 2]
        self.member = {}
        for i, g in enumerate(self.groups):
            for p in g:
                self.member.setdefault(p, set()).add(i)

    def of(self, a: str, b: str) -> str:
        common = sorted(self.member.get(a, set()) & self.member.get(b, set()))
        return f"L{common[0]}" if common else "S:" + ",".join(sorted((a, b)))

    def lines(self, d: Diagram) -> list[list[str]]:
        """Carrier lines as ordered point lists: groups plus segments outside any group."""
        out = [list(g) for g in self.groups]
        for a, b in d.segments:
            if self.of(a, b).startswith("S:"):
                out.append([a, b])
        return out


def _overlap(d: Diagram, s1, s2) -> bool:
    """Two segments on one line overlap in more than a point (positions projected on s1's direction)."""
    import numpy as np
    a = np.asarray(d.points[s1[0]], float)
    u = np.asarray(d.points[s1[1]], float) - a
    n = float(np.hypot(*u)) or 1.0
    u /= n
    proj = lambda p: float(np.dot(np.asarray(d.points[p], float) - a, u))  # noqa: E731
    lo1, hi1 = sorted(map(proj, s1))
    lo2, hi2 = sorted(map(proj, s2))
    return min(hi1, hi2) - max(lo1, lo2) > 1e-6 * n


def usable_circles(d: Diagram) -> list[int]:
    return [i for i, c in enumerate(d.circles) if c.center_xy is not None and c.radius]


def enumerate_queries(d: Diagram, tier2: tuple[str, ...] = ()) -> list[Query]:
    car = Carriers(d)
    out: list[Query] = []
    for s1, s2 in itertools.combinations(d.segments, 2):
        c1, c2 = car.of(*s1), car.of(*s2)
        if c1 == c2 and _overlap(d, s1, s2):
            continue                      # nested/overlapping on one line: indistinguishable when highlighted
        rels = ("equal_length",) if c1 == c2 else TIER1_KINDS["seg_pair"]
        out.append(Query("seg_pair", [seg(*s1), seg(*s2)], rels, "|".join(sorted((c1, c2)))))
    circles = usable_circles(d)
    for i in circles:
        for p in d.points:
            if p != d.circles[i].center:
                out.append(Query("pt_circle", [pt(p), circle_ref(i)], ("on_circle",), f"{p}|C{i}"))
        for s in d.segments:
            out.append(Query("seg_circle", [seg(*s), circle_ref(i)], ("tangent",), f"{car.of(*s)}|C{i}"))
    if "collinear" in tier2:
        for line in car.lines(d):
            a, b = line[0], line[-1]
            key = car.of(a, b)
            for p in d.points:
                if p not in (a, b):
                    out.append(Query("pt_line", [pt(p), *sorted([pt(a), pt(b)])], ("collinear",), f"{p}|{key}"))
    return out


def asserted_lookup(d: Diagram):
    """Returns f(query, relation) -> bool: is this query asserted for this relation?"""
    car = Carriers(d)
    line_rel = {"parallel": set(), "perpendicular": set()}
    exact = {"equal_length": set(), "on_circle": set()}
    tangent, collinear = set(), set()
    for r in d.asserted:
        if r.type in line_rel:
            line_rel[r.type].add(frozenset((car.of(*r.args[0][1:]), car.of(*r.args[1][1:]))))
        elif r.type in exact:
            exact[r.type].add(json.dumps(r.args, separators=(",", ":")))
        elif r.type == "tangent":
            tangent.add((car.of(*r.args[0][1:]), r.args[1][1]))
        elif r.type == "collinear":
            collinear.add((r.args[0][1], car.of(r.args[1][1], r.args[2][1])))

    def f(q: Query, rel: str) -> bool:
        if rel in line_rel:
            return frozenset((car.of(*q.refs[0][1:]), car.of(*q.refs[1][1:]))) in line_rel[rel]
        if rel in exact:
            return q.key() in exact[rel]
        if rel == "tangent":
            return (car.of(*q.refs[0][1:]), q.refs[1][1]) in tangent
        if rel == "collinear":
            return (q.refs[0][1], car.of(q.refs[1][1], q.refs[2][1])) in collinear
        return False

    return f


# ============================================================================ index (plan §5.1-5.4)

# Dataset units: name -> list of (processed set, split inside that set). "train" adds the first
# `data.synthetic_train_size` synth_train diagrams.
UNITS = {
    "train": [("geometry3k_figures", "train")],
    "val": [("geometry3k_figures", "val")],
    "geometry3k_test": [("geometry3k_figures", "test")],
    "pgdp5k_heldout": [("pgdp5k_heldout", "test")],
    "synth_extrap": [("synth_extrap", "extrap")],
}
TRAIN_UNITS = {"train"}


def vocabulary(cfg: dict) -> list[str]:
    """Label order: Tier 1 without `none` (none = all-zero vector), then included Tier 2."""
    rel = cfg["relations"]
    return [r for r in rel["tier1"] if r != "none"] + list(rel.get("tier2_included", []))


def query_id(q: Query) -> str:
    import hashlib
    return hashlib.sha1(f"{q.kind}|{q.key()}".encode()).hexdigest()[:12]


def label_rows(d: Diagram, vocab: list[str], tol: dict, tier2: tuple[str, ...], set_name: str) -> list[dict]:
    """One row per candidate query: multi-label geometric and asserted vectors, applicability mask,
    margins (None where not applicable)."""
    from verifier.checks import evaluate
    from verifier.primitives import bbox_diagonal

    scale = bbox_diagonal(d.points.values())
    is_asserted = asserted_lookup(d)
    rows = []
    for q in enumerate_queries(d, tier2):
        geo, asr, mask, margins = [], [], [], []
        for r in vocab:
            if r in q.relations:
                ok, m = evaluate(d, r, q.refs, tol, scale=scale)
                geo.append(int(bool(ok)))
                asr.append(int(is_asserted(q, r)))
                mask.append(1)
                margins.append(None if m != m else round(float(m), 6))
            else:
                geo.append(0)
                asr.append(0)
                mask.append(0)
                margins.append(None)
        rows.append({"qid": query_id(q), "diagram_id": d.id, "set": set_name, "split": d.split, "kind": q.kind,
                     "refs": q.refs, "geo": geo, "asr": asr, "mask": mask, "margins": margins})
    return rows


def _positive(r: dict) -> bool:
    return any(r["geo"]) or any(r["asr"])


def cap_rows(rows: list[dict], cap: int, rng, keep_positives: bool) -> list[dict]:
    """At most `cap` queries per diagram. Training keeps every positive and fills with negatives;
    evaluation samples uniformly so the natural positive rate is kept in expectation."""
    if len(rows) <= cap:
        return rows
    if not keep_positives:
        idx = sorted(rng.choice(len(rows), cap, replace=False))
        return [rows[i] for i in idx]
    pos = [r for r in rows if _positive(r)]
    neg = [r for r in rows if not _positive(r)]
    k = max(0, cap - len(pos))
    idx = sorted(rng.choice(len(neg), min(k, len(neg)), replace=False)) if k else []
    return pos + [neg[i] for i in idx]


def downsample_none(rows: list[dict], ratio: float, rng) -> list[dict]:
    """Training only: per query kind, keep at most ratio x (positive queries) all-negative queries."""
    out = []
    for kind in sorted({r["kind"] for r in rows}):
        rk = [r for r in rows if r["kind"] == kind]
        pos = [r for r in rk if _positive(r)]
        neg = [r for r in rk if not _positive(r)]
        keep = min(len(neg), int(ratio * len(pos)))
        idx = set(rng.choice(len(neg), keep, replace=False).tolist()) if keep else set()
        out += pos + [r for i, r in enumerate(neg) if i in idx]
    return out


def index_path(cfg: dict, unit: str) -> "Path":
    from pathlib import Path
    from experiments.run import REPO_ROOT

    n = cfg["data"].get("max_diagrams")
    synth = cfg["data"].get("synthetic_train_size", 0) if unit in TRAIN_UNITS else 0
    tag = (f".synth{synth}" if synth else "") + (f".n{n}" if n else "")
    return Path(REPO_ROOT) / cfg["paths"]["processed"] / "index" / f"{unit}{tag}.jsonl"


def unit_sources(cfg: dict, unit: str) -> list[tuple[str, str, int | None]]:
    """(set, split, max diagrams) parts of a unit, honouring synthetic_train_size and max_diagrams."""
    n = cfg["data"].get("max_diagrams")
    parts = [(s, sp, n) for s, sp in UNITS[unit]]
    size = cfg["data"].get("synthetic_train_size", 0)
    if unit in TRAIN_UNITS and size:
        parts.append(("synth_train", "train", min(size, n) if n else size))
    return parts


def build_index(cfg: dict, unit: str) -> tuple["Path", dict]:
    """Write the query index of a dataset unit; returns its path and summary counts."""
    import json as _json
    import zlib
    from pathlib import Path
    import numpy as np
    from data.schema import read_jsonl
    from experiments.run import REPO_ROOT

    proc = Path(REPO_ROOT) / cfg["paths"]["processed"]
    vocab = vocabulary(cfg)
    tier2 = tuple(cfg["relations"].get("tier2_included", []))
    train = unit in TRAIN_UNITS
    cap = cfg["data"]["max_queries_per_diagram"]
    rows_out, n_diag = [], 0
    for set_name, split, limit in unit_sources(cfg, unit):
        taken = 0
        for d in read_jsonl(proc / f"{set_name}.jsonl"):
            if d.split != split:
                continue
            if limit is not None and taken >= limit:
                break
            taken += 1
            rng = np.random.default_rng([cfg["data"]["index_seed"], zlib.crc32(d.id.encode())])
            rows = cap_rows(label_rows(d, vocab, cfg["tolerances"], tier2, set_name), cap, rng, keep_positives=train)
            if train:
                rows = downsample_none(rows, cfg["data"]["train_none_ratio"], rng)
            rows_out += rows
        n_diag += taken
    path = index_path(cfg, unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows_out:
            f.write(_json.dumps(r, separators=(",", ":")) + "\n")
    geo = np.array([r["geo"] for r in rows_out]) if rows_out else np.zeros((0, len(vocab)))
    summary = {"unit": unit, "diagrams": n_diag, "queries": len(rows_out),
               "none": int((geo.sum(1) == 0).sum()) if len(geo) else 0,
               **{f"pos_{r}": int(geo[:, i].sum()) for i, r in enumerate(vocab)}}
    return path, summary


def main(argv: list[str] | None = None) -> int:
    """python -m data.queries [--config X] [--units train val ...]: build query indexes."""
    import argparse
    from experiments.run import REPO_ROOT, load_config

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--units", nargs="+", default=list(UNITS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for unit in args.units:
        path, s = build_index(cfg, unit)
        print(f"{unit:16s} -> {path.name}: " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "unit"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
