"""One function per relation, each returning (is_true, margin) (plan §3.2).

`margin` is the raw quantity before thresholding, so its distribution can be plotted:
  parallel(s1, s2)        |sin θ| between the directions;   true if <= sin(tol_deg) and the two
                          segments are not on one line (coincident lines are not parallel; "one
                          line" uses the collinear tolerance, collinear_rel * diagram_scale)
  perpendicular(s1, s2)   |cos θ|;                           true if <= sin(tol_deg)
  equal_length(s1, s2)    | |s1| - |s2| | / max(|s1|, |s2|); true if <= tol_rel
  on_circle(p, c)         | |p - centre| - r | / r;          true if <= tol_rel
  tangent(s, c)           | dist(line(s), centre) - r | / r; true if <= tol_rel, the foot of the
                          perpendicular from the centre lies on the segment (slack: max(FOOT_SLACK
                          of its length, tol_rel * r), so tangency at an endpoint counts; no dataset
                          here marks infinite lines), and the segment is not a chord (both endpoints
                          within tol_rel * r of the circle)
  collinear(p, a, b)      |cross(b - a, p - a)| / (|b - a| * diagram_scale)
                          = distance of p from line ab / diagram_scale; true if <= tol_rel
  midpoint(m, s)          |m - mid(s)| / |s|;                true if <= tol_rel
  equal_angle(A1, A2)     |angle1 - angle2| in degrees;     true if <= tol_deg
  perp_bisector(s1, s2)   max(|cos θ|, dist(mid(s2), line(s1)) / |s2|);
                          true if |cos θ| <= sin(tol_deg) and the distance term <= tol_rel
diagram_scale = diagonal of the bounding box of all points of the diagram.

Segments are pairs of points, circles are (centre, radius). Degenerate input (zero-length segment,
missing radius) gives margin nan and is_true False.

`evaluate(diagram, rtype, args, tol)` resolves schema references (["seg", A, B], ...) to coordinates.
Tolerances come from the `tolerances` config section (keys in TOL_KEYS), fixed in Phase 3.4.
"""

from __future__ import annotations

import math

from verifier import primitives as P

FOOT_SLACK = 0.02
COINCIDENT_REL = 0.02
TOL_KEYS = {
    "parallel": "parallel_deg", "perpendicular": "perpendicular_deg", "equal_length": "equal_length_rel",
    "on_circle": "on_circle_rel", "tangent": "tangent_rel", "collinear": "collinear_rel",
    "midpoint": "midpoint_rel", "equal_angle": "equal_angle_deg", "perp_bisector": "perpendicular_deg",
}
NAN = float("nan")


def _ok(margin: float, thr: float | None) -> bool | None:
    if thr is None:
        return None
    return bool(margin <= thr) if not math.isnan(margin) else False


def _sin(deg: float | None) -> float | None:
    return None if deg is None else math.sin(math.radians(deg))


# ----------------------------------------------------------------------------- Tier 1

def parallel(s1, s2, tol_deg: float | None = None, coincident_px: float | None = None) -> tuple[bool | None, float]:
    l1, l2 = P.length(*s1), P.length(*s2)
    if l1 < P.EPS or l2 < P.EPS:
        return _ok(NAN, tol_deg), NAN
    s, _ = P.sin_cos_between(s1, s2)
    ok = _ok(s, _sin(tol_deg))
    if ok and coincident(s1, s2, coincident_px):
        ok = False                       # one line, not two parallel lines
    return ok, s


def coincident(s1, s2, tol_px: float | None = None) -> bool:
    """Both segments lie on one line: s2's endpoints within tol_px of line(s1). evaluate() passes
    collinear_rel * diagram_scale (the collinear relation's own tolerance); the fallback is
    COINCIDENT_REL * the longer segment."""
    if tol_px is None:
        tol_px = COINCIDENT_REL * max(P.length(*s1), P.length(*s2))
    return all(P.point_line_distance(e, *s1) <= tol_px for e in s2)


def perpendicular(s1, s2, tol_deg: float | None = None) -> tuple[bool | None, float]:
    if P.length(*s1) < P.EPS or P.length(*s2) < P.EPS:
        return _ok(NAN, tol_deg), NAN
    _, c = P.sin_cos_between(s1, s2)
    return _ok(c, _sin(tol_deg)), c


def equal_length(s1, s2, tol_rel: float | None = None) -> tuple[bool | None, float]:
    l1, l2 = P.length(*s1), P.length(*s2)
    m = abs(l1 - l2) / max(l1, l2) if max(l1, l2) > P.EPS else NAN
    return _ok(m, tol_rel), m


def on_circle(p, c, tol_rel: float | None = None) -> tuple[bool | None, float]:
    centre, r = c
    m = abs(P.length(p, centre) - r) / r if r and r > P.EPS else NAN
    return _ok(m, tol_rel), m


def tangent(s, c, tol_rel: float | None = None) -> tuple[bool | None, float]:
    centre, r = c
    if not r or r <= P.EPS or P.length(*s) < P.EPS:
        return _ok(NAN, tol_rel), NAN
    m = abs(P.point_line_distance(centre, *s) - r) / r
    ok = _ok(m, tol_rel)
    if ok is None:
        return None, m
    _, t = P.foot_of_perpendicular(centre, *s)
    slack = max(FOOT_SLACK, tol_rel * r / P.length(*s))          # tangency may sit at an endpoint
    on_seg = -slack <= t <= 1 + slack
    chord = all(abs(P.length(e, centre) - r) <= tol_rel * r for e in s)   # both ends on the circle
    return bool(ok and on_seg and not chord), m


# ----------------------------------------------------------------------------- Tier 2

def collinear(p, a, b, diagram_scale: float, tol_rel: float | None = None) -> tuple[bool | None, float]:
    ab = P.length(a, b)
    if ab < P.EPS or diagram_scale <= P.EPS:
        return _ok(NAN, tol_rel), NAN
    m = abs(P.cross(P.vec(b) - P.vec(a), P.vec(p) - P.vec(a))) / (ab * diagram_scale)
    return _ok(m, tol_rel), m


def midpoint(m_pt, s, tol_rel: float | None = None) -> tuple[bool | None, float]:
    ln = P.length(*s)
    m = P.length(m_pt, P.midpoint(*s)) / ln if ln > P.EPS else NAN
    return _ok(m, tol_rel), m


def equal_angle(a1, a2, tol_deg: float | None = None) -> tuple[bool | None, float]:
    m = abs(P.angle_between_rays(*a1) - P.angle_between_rays(*a2))
    return _ok(m, tol_deg), m


def perp_bisector(s1, s2, tol_deg: float | None = None, tol_rel: float | None = None) -> tuple[bool | None, float]:
    l2 = P.length(*s2)
    if P.length(*s1) < P.EPS or l2 < P.EPS:
        return (False if tol_deg is not None else None), NAN
    _, c = P.sin_cos_between(s1, s2)
    d = P.point_line_distance(P.midpoint(*s2), *s1) / l2
    m = max(c, d)
    if tol_deg is None or tol_rel is None:
        return None, m
    return bool(c <= _sin(tol_deg) and d <= tol_rel), m


# ----------------------------------------------------------------------------- schema refs

def resolve(d, ref):
    """Coordinates for a schema reference: pt -> point, seg -> (a, b), angle -> (a, v, b),
    circle -> (centre_xy, radius)."""
    kind = ref[0]
    if kind == "pt":
        return d.points[ref[1]]
    if kind == "seg":
        return d.points[ref[1]], d.points[ref[2]]
    if kind == "angle":
        return d.points[ref[1]], d.points[ref[2]], d.points[ref[3]]
    if kind == "circle":
        c = d.circles[ref[1]]
        return c.center_xy, c.radius
    raise ValueError(f"unknown ref {ref!r}")


def evaluate(d, rtype: str, args: list, tol: dict | None = None, scale: float | None = None) -> tuple[bool | None, float]:
    """Run the check for a relation given in schema form on diagram d."""
    tol = tol or {}
    t = tol.get(TOL_KEYS[rtype])
    for a in args:
        if a[0] == "circle" and (d.circles[a[1]].center_xy is None or not d.circles[a[1]].radius):
            return (False if t is not None else None), NAN
    x = [resolve(d, a) for a in args]
    if rtype == "parallel":
        s = scale if scale is not None else P.bbox_diagonal(d.points.values())
        c = tol.get("collinear_rel")
        return parallel(x[0], x[1], t, c * s if c is not None else None)
    if rtype == "perpendicular":
        return perpendicular(x[0], x[1], t)
    if rtype == "equal_length":
        return equal_length(x[0], x[1], t)
    if rtype == "on_circle":
        return on_circle(x[0], x[1], t)
    if rtype == "tangent":
        return tangent(x[0], x[1], t)
    if rtype == "collinear":
        s = scale if scale is not None else P.bbox_diagonal(d.points.values())
        return collinear(x[0], x[1], x[2], s, t)
    if rtype == "midpoint":
        return midpoint(x[0], x[1], t)
    if rtype == "equal_angle":
        return equal_angle(x[0], x[1], t)
    if rtype == "perp_bisector":
        return perp_bisector(x[0], x[1], t, tol.get("midpoint_rel"))
    raise ValueError(f"unknown relation {rtype!r}")
