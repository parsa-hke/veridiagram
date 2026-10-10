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

The per-diagram cap (60, keep all positives) is applied in Phase 5, not here.
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
