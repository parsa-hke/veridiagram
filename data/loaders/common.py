"""Helpers shared by the loaders: equivalence-class closure and derived Tier 2 relations.

"Asserted" relations are what the annotation states. Two derivations are applied uniformly to
every source so the census compares like with like:
- parallel / equal_length / equal_angle are equivalence relations: an annotation that marks
  three segments with the same tick class asserts all three pairs.
- midpoint and perp_bisector have no predicate in any source; they are derived from asserted
  collinear + equal_length (+ perpendicular) relations.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict

from data.schema import Relation, pt, seg


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return [sorted(g) for g in out.values()]


def close_equivalences(rels: list[Relation]) -> list[Relation]:
    """Replace parallel / equal_length / equal_angle pair lists with all pairs of their classes."""
    out = [r for r in rels if r.type not in ("parallel", "equal_length", "equal_angle")]
    for rtype in ("parallel", "equal_length", "equal_angle"):
        uf = UnionFind()
        for r in rels:
            if r.type == rtype:
                a, b = (json.dumps(x) for x in r.args)
                uf.union(a, b)
        for group in uf.groups():
            for a, b in itertools.combinations(group, 2):
                out.append(Relation(rtype, [json.loads(a), json.loads(b)]))
    return dedupe(out)


def dedupe(rels: list[Relation]) -> list[Relation]:
    seen, out = set(), []
    for r in rels:
        if r.key() not in seen:
            seen.add(r.key())
            out.append(r)
    return out


def derive_tier2(rels: list[Relation], lines: list[list[str]]) -> list[Relation]:
    """Add midpoint and perp_bisector relations implied by asserted relations.

    midpoint(M, AB): M is asserted on line AB, strictly between A and B along an annotated
    collinear group, and |AM| = |MB| is asserted.
    perp_bisector(L, AB): midpoint(M, AB) holds and a segment L through M (M is an endpoint of
    L or lies on it) is asserted perpendicular to AB or to a sub-segment of line AB through M.
    """
    eq = {r.key()[1] for r in rels if r.type == "equal_length"}
    perp = [r for r in rels if r.type == "perpendicular"]
    out = list(rels)
    mids = []
    for g in lines:
        for i, j in itertools.combinations(range(len(g)), 2):
            a, b = g[i], g[j]
            for m in g[i + 1:j]:
                if json.dumps(sorted([seg(a, m), seg(m, b)], key=json.dumps)) in eq:
                    out.append(Relation("midpoint", [pt(m), seg(a, b)]))
                    mids.append((m, a, b, g))
    for m, a, b, g in mids:
        on_ab = set(g)
        for r in perp:
            for s1, s2 in (r.args, r.args[::-1]):
                if set(s2[1:]) <= on_ab and _through(m, s1, lines) and not set(s1[1:]) <= on_ab:
                    out.append(Relation("perp_bisector", [s1, seg(a, b)]))
    return dedupe(out)


def _through(m: str, s: list, lines: list[list[str]]) -> bool:
    """True if point m is an endpoint of segment s or lies on an annotated line containing s."""
    if m in s[1:]:
        return True
    return any(m in g and s[1] in g and s[2] in g for g in lines)


def order_along(names: list[str], points: dict[str, tuple[float, float]]) -> list[str]:
    """Sort collinear points along their principal direction."""
    pts = [points[n] for n in names]
    (x0, y0), (x1, y1) = min(pts), max(pts)
    dx, dy = x1 - x0, y1 - y0
    return sorted(names, key=lambda n: (points[n][0] - x0) * dx + (points[n][1] - y0) * dy)


def segments_from_lines(lines: list[list[str]], extra: list[tuple[str, str]] = ()) -> list[tuple[str, str]]:
    """All sub-segments of every collinear group, plus any extra segments; sorted, unique."""
    out = set()
    for g in lines:
        for a, b in itertools.combinations(g, 2):
            out.add(tuple(sorted((a, b))))
    for a, b in extra:
        if a != b:
            out.add(tuple(sorted((a, b))))
    return sorted(out)
