"""Deductive closure over predicted relations; returns contradictions and closure size (plan §7.10).

A relation set is a list of facts `(relation, a, b)`. The entities are opaque strings chosen by the
caller: for `parallel` / `perpendicular` they should be carrier lines (two sub-segments of one
drawn line are the same line), for `equal_length` segments, for the other relations whatever the
query refs are. `same_line` groups (optional) say which entities lie on one line: they are parallel
by construction, so `perpendicular` between two of them is a contradiction and a perpendicular to
one is a perpendicular to all.

Rules (to a fixpoint):
  R1 parallel is transitive (and symmetric; same-line entities are parallel)
  R2 perpendicular(a, b) and parallel(b, c) -> perpendicular(a, c)
  R3 perpendicular(a, c) and perpendicular(b, c) -> parallel(a, b)
  R4 equal_length is transitive
  C1 contradiction: a pair both parallel and perpendicular (incl. an entity perpendicular to itself)
  C2 contradiction: a derived fact that the caller listed among `negatives` (facts predicted false)
Other relations (on_circle, tangent, collinear) have no rules: they enter the closure unchanged.

    contradictions, facts = closure(positives, negatives=(), same_line=())
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

LINE_RELATIONS = ("parallel", "perpendicular")


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True

    def groups(self) -> dict:
        out = defaultdict(set)
        for x in list(self.p):
            out[self.find(x)].add(x)
        return out


def _fact(rel: str, a, b) -> tuple:
    return (rel, *sorted((a, b)))


def closure(positives: list, negatives=(), same_line=()) -> tuple[list, set]:
    """Returns (contradictions, closure facts). Each contradiction is a dict with its kind
    ('parallel_and_perpendicular' or 'derived_negated'), the fact and, for C2, nothing else needed."""
    par, eq = _UF(), _UF()
    perp_edges, other = set(), set()
    for group in same_line:
        group = list(group)
        for x in group:
            par.find(x)
        for x in group[1:]:
            par.union(group[0], x)
    for rel, a, b in positives:
        if rel == "parallel":
            par.union(a, b)
        elif rel == "perpendicular":
            par.find(a), par.find(b)
            perp_edges.add((a, b))
        elif rel == "equal_length":
            eq.union(a, b)
        else:
            other.add(_fact(rel, a, b))

    # R2/R3 on parallel classes: every class perpendicular to a class C is one class (R3), and
    # perpendicularity holds between whole classes (R2). Merge until nothing changes.
    changed = True
    while changed:
        changed = False
        perp_of = defaultdict(set)
        for a, b in perp_edges:
            ra, rb = par.find(a), par.find(b)
            perp_of[ra].add(rb)
            perp_of[rb].add(ra)
        for root, partners in perp_of.items():
            partners = sorted(partners)
            for x in partners[1:]:
                if par.union(partners[0], x):
                    changed = True

    contradictions = []
    classes = par.groups()
    perp_roots = {tuple(sorted((par.find(a), par.find(b)))) for a, b in perp_edges}
    facts = set(other)
    for members in classes.values():
        for a, b in combinations(sorted(members), 2):
            facts.add(_fact("parallel", a, b))
    for ra, rb in sorted(perp_roots):
        if ra == rb:
            contradictions.append({"kind": "parallel_and_perpendicular", "class": sorted(classes[ra])})
            continue
        for a in classes[ra]:
            for b in classes[rb]:
                facts.add(_fact("perpendicular", a, b))
    for members in eq.groups().values():
        for a, b in combinations(sorted(members), 2):
            facts.add(_fact("equal_length", a, b))

    for rel, a, b in negatives:
        f = _fact(rel, a, b)
        if f in facts:
            contradictions.append({"kind": "derived_negated", "fact": list(f)})
    return contradictions, facts
