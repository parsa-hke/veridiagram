"""Random diagrams with injected relations, then jittered (plan §4.1).

A diagram is built step by step on a square canvas. At each step, with probability
`inject_rate`, one construction adds a relation exactly:
  parallel       a new segment along the direction of an existing one
  perpendicular  a new segment at right angles, from an endpoint or from a foot on the segment
  equal_length   a new segment of the same length, from a shared endpoint or elsewhere
  collinear      a new point on an existing segment (sometimes its midpoint, with tick marks)
  on_circle      a new point on a circle
  tangent        a segment along the tangent at a point of a circle (tangency point sometimes
                 labelled, sometimes with the radius drawn and a right-angle mark)
otherwise a random point (minimum separation) or a random segment between existing points that
passes clear of every other point. Circles (0-2) are placed first, their centre labelled with
probability `center_label_prob`.

The constructions are the synthetic "asserted" labels, and each records the mark the renderer
draws for it (right-angle square, tick marks, parallel arrows). The whole diagram is then rotated
slightly and every coordinate jittered by N(0, jitter_px), so geometric truth is decided by the
verifier, not by the construction. The exact (pre-jitter) coordinates are kept in
`meta.exact_points` for the sanity check of plan §4.4.

Each diagram is a pure function of (seed, pool, index), so pools are reproducible and can be
generated in parallel.
"""

from __future__ import annotations

import math
import string
from collections import Counter

import numpy as np

from data.loaders.common import close_equivalences, dedupe, derive_tier2, order_along, segments_from_lines
from data.schema import Circle, Diagram, Relation, circle_ref, pt, seg

NAMES = list(string.ascii_uppercase)
POOL_CODE = {"train": 1, "extrap": 2, "unlabelled": 3, "test": 4}
CLEARANCE_PX = 8.0          # a random segment must pass at least this far from other points
MIN_SEG_PX = 35.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.zeros(2)


def _rot(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _point_seg_dist(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.hypot(*(p - (a + t * ab))))


class Builder:
    def __init__(self, rng: np.random.Generator, cfg: dict, n_points: int, n_segments: int):
        self.rng, self.cfg = rng, cfg
        self.size = cfg["canvas_px"]
        self.lo, self.hi = cfg["margin_px"], cfg["canvas_px"] - cfg["margin_px"]
        self.N, self.M = n_points, n_segments
        self.sep = cfg["min_separation_px"]
        self.pts: dict[str, np.ndarray] = {}
        self.drawn: list[tuple[str, str]] = []
        self.groups: list[list[str]] = []
        self.circles: list[dict] = []
        self.rels: list[Relation] = []
        self.marks: list[dict] = []
        self.eq_class: dict[tuple, int] = {}
        self.par_class: dict[tuple, int] = {}
        self.made = Counter()

    # ------------------------------------------------------------------ primitives
    def inside(self, p) -> bool:
        return self.lo <= p[0] <= self.hi and self.lo <= p[1] <= self.hi

    def free(self, p) -> bool:
        return all(np.hypot(*(p - q)) >= self.sep for q in self.pts.values())

    def can_add(self, k: int) -> bool:
        return len(self.pts) + k <= self.N

    def ok_point(self, p) -> bool:
        return self.inside(p) and self.free(p)

    def add_point(self, p) -> str:
        name = NAMES[len(self.pts)]
        self.pts[name] = np.asarray(p, dtype=float)
        return name

    def random_free_point(self, tries: int = 60):
        for _ in range(tries):
            p = self.rng.uniform(self.lo, self.hi, size=2)
            if self.free(p):
                return p
        return None

    def clear(self, a: str, b: str, ignore: tuple = ()) -> bool:
        """Segment ab is new, long enough, and passes clear of all other points."""
        if a == b or (a, b) in self.drawn or (b, a) in self.drawn:
            return False
        pa, pb = self.pts[a], self.pts[b]
        if np.hypot(*(pb - pa)) < MIN_SEG_PX:
            return False
        return all(_point_seg_dist(q, pa, pb) >= CLEARANCE_PX
                   for n, q in self.pts.items() if n not in (a, b) and n not in ignore)

    def add_segment(self, a: str, b: str) -> None:
        self.drawn.append((a, b))

    def group_of(self, a: str, b: str) -> list[str]:
        for g in self.groups:
            if a in g and b in g:
                return g
        g = [a, b]
        self.groups.append(g)
        return g

    def pick_segment(self):
        return self.drawn[self.rng.integers(len(self.drawn))] if self.drawn else None

    def _class(self, table: dict, key: tuple, other: tuple) -> int:
        k = table.get(key, table.get(key[::-1]))
        if k is None:
            k = len(set(table.values())) + 1
        for s in (key, other):
            table[s] = k
        return k

    # ------------------------------------------------------------------ constructions
    def c_parallel(self) -> bool:
        s = self.pick_segment()
        if s is None or not self.can_add(2) or len(self.drawn) >= self.M:
            return False
        a, b = s
        d = _unit(self.pts[b] - self.pts[a])
        length = float(np.clip(np.hypot(*(self.pts[b] - self.pts[a])) * self.rng.uniform(0.6, 1.4), 50, 260))
        for _ in range(30):
            c = self.random_free_point(10)
            if c is None:
                return False
            v = c - self.pts[a]
            off = abs(float(d[0] * v[1] - d[1] * v[0]))
            e = c + self.rng.choice([-1, 1]) * length * d
            if off >= 30 and self.ok_point(e) and np.hypot(*(e - c)) >= self.sep:
                cn = self.add_point(c)
                en = self.add_point(e)
                if not self.clear(cn, en):
                    self._undo(2)
                    continue
                self.add_segment(cn, en)
                self.rels.append(Relation("parallel", [seg(a, b), seg(cn, en)]))
                k = self._class(self.par_class, (a, b), (cn, en))
                self.marks += [{"type": "arrows", "seg": [a, b], "n": (k - 1) % 2 + 1},
                               {"type": "arrows", "seg": [cn, en], "n": (k - 1) % 2 + 1}]
                return True
        return False

    def c_perpendicular(self) -> bool:
        s = self.pick_segment()
        if s is None or len(self.drawn) >= self.M:
            return False
        a, b = s
        pa, pb = self.pts[a], self.pts[b]
        nrm = _rot(_unit(pb - pa), math.pi / 2) * self.rng.choice([-1, 1])
        length = self.rng.uniform(50, 200)
        if self.rng.random() < 0.5 and self.can_add(1):                       # right angle at an endpoint
            v, other = (a, b) if self.rng.random() < 0.5 else (b, a)
            e = self.pts[v] + length * nrm
            if not self.ok_point(e):
                return False
            en = self.add_point(e)
            if not self.clear(v, en):
                self._undo(1)
                return False
            self.add_segment(v, en)
            self.rels.append(Relation("perpendicular", [seg(a, b), seg(v, en)]))
            self.marks.append({"type": "right_angle", "vertex": v, "rays": [other, en]})
            return True
        if not self.can_add(2):
            return False
        f = pa + self.rng.uniform(0.25, 0.75) * (pb - pa)                      # right angle at a foot
        e = f + length * nrm
        if not (self.ok_point(f) and self.ok_point(e)):
            return False
        fn = self.add_point(f)
        en = self.add_point(e)
        if not self.clear(fn, en):
            self._undo(2)
            return False
        g = self.group_of(a, b)
        g.append(fn)
        self.add_segment(fn, en)
        self.rels += [Relation("perpendicular", [seg(a, b), seg(fn, en)]),
                      Relation("collinear", [pt(fn), pt(a), pt(b)])]
        self.marks.append({"type": "right_angle", "vertex": fn, "rays": [self.rng.choice([a, b]), en]})
        return True

    def c_equal_length(self) -> bool:
        s = self.pick_segment()
        if s is None or not self.can_add(1) or len(self.drawn) >= self.M:
            return False
        a, b = s
        pa, pb = self.pts[a], self.pts[b]
        length = float(np.hypot(*(pb - pa)))
        if self.rng.random() < 0.5:                                           # shared endpoint (isosceles)
            v = a if self.rng.random() < 0.5 else b
            other = b if v == a else a
            ang = self.rng.choice([-1, 1]) * math.radians(self.rng.uniform(30, 150))
            e = self.pts[v] + length * _rot(_unit(self.pts[other] - self.pts[v]), ang)
            start = v
            need_new_start = False
        else:                                                                 # elsewhere
            if not self.can_add(2):
                return False
            c = self.random_free_point(10)
            if c is None:
                return False
            e = c + length * _rot(np.array([1.0, 0.0]), self.rng.uniform(0, 2 * math.pi))
            start, need_new_start = c, True
        if not self.ok_point(e):
            return False
        if need_new_start:
            start = self.add_point(start)
        en = self.add_point(e)
        if not self.clear(start, en):
            self._undo(2 if need_new_start else 1)
            return False
        self.add_segment(start, en)
        self.rels.append(Relation("equal_length", [seg(a, b), seg(start, en)]))
        k = self._class(self.eq_class, (a, b), (start, en))
        self.marks += [{"type": "ticks", "seg": [a, b], "n": (k - 1) % 3 + 1},
                       {"type": "ticks", "seg": [start, en], "n": (k - 1) % 3 + 1}]
        return True

    def c_collinear(self) -> bool:
        s = self.pick_segment()
        if s is None or not self.can_add(1):
            return False
        a, b = s
        g = self.group_of(a, b)
        mid = self.rng.random() < 0.3 and len(g) == 2
        t = 0.5 if mid else self.rng.uniform(0.2, 0.8)
        m = self.pts[a] + t * (self.pts[b] - self.pts[a])
        if not self.ok_point(m):
            return False
        mn = self.add_point(m)
        g.append(mn)
        self.rels.append(Relation("collinear", [pt(mn), pt(a), pt(b)]))
        if mid:
            self.rels.append(Relation("equal_length", [seg(a, mn), seg(mn, b)]))
            k = self._class(self.eq_class, (a, mn), (mn, b))
            self.marks += [{"type": "ticks", "seg": [a, mn], "n": (k - 1) % 3 + 1},
                           {"type": "ticks", "seg": [mn, b], "n": (k - 1) % 3 + 1}]
        return True

    def c_on_circle(self) -> bool:
        if not self.circles or not self.can_add(1):
            return False
        i = int(self.rng.integers(len(self.circles)))
        c = self.circles[i]
        p = c["c"] + c["r"] * _rot(np.array([1.0, 0.0]), self.rng.uniform(0, 2 * math.pi))
        if not self.ok_point(p):
            return False
        pn = self.add_point(p)
        c["on"].append(pn)
        self.rels.append(Relation("on_circle", [pt(pn), circle_ref(i)]))
        return True

    def c_tangent(self) -> bool:
        if not self.circles or len(self.drawn) >= self.M:
            return False
        i = int(self.rng.integers(len(self.circles)))
        c = self.circles[i]
        radial = _rot(np.array([1.0, 0.0]), self.rng.uniform(0, 2 * math.pi))
        t = c["c"] + c["r"] * radial
        u = _rot(radial, math.pi / 2)
        p = t - self.rng.uniform(40, 170) * u
        q = t + self.rng.uniform(40, 170) * u
        label_t = self.rng.random() < 0.6
        if not self.can_add(3 if label_t else 2) or not (self.ok_point(p) and self.ok_point(q)):
            return False
        if label_t and not self.ok_point(t):
            label_t = False
        pn, qn = self.add_point(p), self.add_point(q)
        added = 2
        tn = None
        if label_t:
            tn = self.add_point(t)
            added = 3
        if not self.clear(pn, qn, ignore=(tn,) if tn else ()):
            self._undo(added)
            return False
        self.add_segment(pn, qn)
        self.rels.append(Relation("tangent", [seg(pn, qn), circle_ref(i)]))
        if tn:
            self.groups.append([pn, qn, tn])
            c["on"].append(tn)
            self.rels += [Relation("on_circle", [pt(tn), circle_ref(i)]),
                          Relation("collinear", [pt(tn), pt(pn), pt(qn)])]
            if c["name"] and self.rng.random() < 0.5 and len(self.drawn) < self.M:
                self.add_segment(c["name"], tn)
                self.rels.append(Relation("perpendicular", [seg(c["name"], tn), seg(pn, qn)]))
                self.marks.append({"type": "right_angle", "vertex": tn, "rays": [c["name"], self.rng.choice([pn, qn])]})
        return True

    def _undo(self, k: int) -> None:
        for _ in range(k):
            self.pts.pop(NAMES[len(self.pts) - 1])

    # ------------------------------------------------------------------ random filling
    def random_point(self) -> bool:
        if not self.can_add(1):
            return False
        p = self.random_free_point()
        if p is None:
            return False
        self.add_point(p)
        return True

    def isolated(self) -> list[str]:
        used = {n for s in self.drawn for n in s} | {n for g in self.groups for n in g}
        used |= {n for c in self.circles for n in c["on"]} | {c["name"] for c in self.circles if c["name"]}
        return [n for n in self.pts if n not in used]

    def random_segment(self) -> bool:
        """A random segment, starting from an isolated point when there is one (textbook figures
        rarely have loose points)."""
        names = list(self.pts)
        if len(names) < 2 or len(self.drawn) >= self.M:
            return False
        lonely = self.isolated()
        for _ in range(20):
            a = self.rng.choice(lonely) if lonely else self.rng.choice(names)
            b = self.rng.choice([n for n in names if n != a])
            if self.clear(a, b):
                self.add_segment(a, b)
                return True
        return False

    def add_circles(self, k: int) -> None:
        lo_r, hi_r = self.cfg["radius_px"]
        for _ in range(k):
            for _ in range(40):
                r = self.rng.uniform(lo_r, min(hi_r, (self.hi - self.lo) / 2))
                c = self.rng.uniform(self.lo + r, self.hi - r, size=2)
                if all(np.hypot(*(c - o["c"])) >= 0.6 * (r + o["r"]) for o in self.circles):
                    name = None
                    if self.rng.random() < self.cfg["center_label_prob"] and self.can_add(1) and self.free(c):
                        name = self.add_point(c)
                    self.circles.append({"c": c, "r": r, "name": name, "on": []})
                    break

    def build(self) -> None:
        self.add_circles(int(self.rng.integers(self.cfg["n_circles"][0], self.cfg["n_circles"][1] + 1)))
        for _ in range(2):
            self.random_point()
        self.random_segment()
        weights = self.cfg.get("construction_weights", {})
        constructions = [self.c_parallel, self.c_perpendicular, self.c_equal_length, self.c_collinear,
                         self.c_on_circle, self.c_tangent]
        w = np.array([float(weights.get(f.__name__[2:], 1.0)) for f in constructions])
        w = w / w.sum()
        for _ in range(400):
            if len(self.pts) >= self.N and len(self.drawn) >= self.M:
                break
            if self.rng.random() < self.cfg["inject_rate"]:
                f = constructions[int(self.rng.choice(len(constructions), p=w))]
                if f():
                    self.made[f.__name__[2:]] += 1
                    continue
            if len(self.pts) < self.N and (len(self.drawn) >= self.M or self.rng.random() < 0.5):
                self.random_point()
            else:
                self.random_segment()


def generate_one(cfg: dict, pool: str, index: int, seed: int,
                 n_points: tuple[int, int] | None = None, n_segments: tuple[int, int] | None = None) -> Diagram:
    rng = np.random.default_rng([seed, POOL_CODE[pool], index])
    n_points = n_points or tuple(cfg["n_points"])
    n_segments = n_segments or tuple(cfg["n_segments"])
    b = Builder(rng, cfg, int(rng.integers(n_points[0], n_points[1] + 1)),
                int(rng.integers(n_segments[0], n_segments[1] + 1)))
    b.build()

    # slight rotation about the canvas centre, then jitter
    size = cfg["canvas_px"]
    ang = math.radians(rng.uniform(-1, 1) * cfg["render"]["rotation_deg"])
    centre = np.array([size / 2, size / 2])
    turn = lambda p: centre + _rot(np.asarray(p) - centre, ang)  # noqa: E731
    exact = {n: turn(p) for n, p in b.pts.items()}
    jit = cfg["jitter_px"]
    points = {n: tuple(float(v) for v in p + rng.normal(0, jit, 2)) for n, p in exact.items()}
    circles = []
    for c in b.circles:
        cxy = points[c["name"]] if c["name"] else tuple(float(v) for v in turn(c["c"]) + rng.normal(0, jit, 2))
        circles.append(Circle(center=c["name"], center_xy=cxy, radius=float(c["r"]), points_on=sorted(set(c["on"]))))

    lines = [order_along(g, points) for g in b.groups if len(g) >= 3]
    rels = derive_tier2(close_equivalences(dedupe(b.rels)), lines)
    idx = f"{index:06d}"
    return Diagram(
        id=f"synthetic:{pool}_{idx}", source="synthetic", split=pool,
        image_path=f"synthetic/{pool}/{idx}.png", width=size, height=size,
        points=points, segments=segments_from_lines(lines, b.drawn), lines=lines, circles=circles,
        asserted=rels, flags={},
        meta={"drawn": [list(s) for s in b.drawn], "marks": b.marks, "constructions": dict(b.made),
              "exact_points": {n: [float(v) for v in p] for n, p in exact.items()},
              "exact_circles": [[float(v) for v in turn(c["c"])] + [float(c["r"])] for c in b.circles],
              "rotation_deg": round(math.degrees(ang), 3), "seed": [seed, POOL_CODE[pool], index]},
    )


def generate(n: int, seed: int, cfg: dict, pool: str = "train", start: int = 0, **ranges) -> list[Diagram]:
    """Generate diagrams start..start+n-1 of a pool (no rendering)."""
    return [generate_one(cfg, pool, i, seed, **ranges) for i in range(start, start + n)]
