"""Vector helpers in float64: direction, length, cross, dot, point-to-line distance,
foot of perpendicular, angle between rays (plan §3.1).

Points are anything convertible to a length-2 float64 array, in image pixels (y down). All
functions are orientation-agnostic where it matters (sin/cos of the angle between lines are
taken in absolute value by the callers).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EPS = 1e-12


def vec(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(2)


def direction(a, b) -> np.ndarray:
    """Unit vector from a to b (zeros if a == b)."""
    d = vec(b) - vec(a)
    n = math.hypot(d[0], d[1])
    return d / n if n > EPS else np.zeros(2)


def length(a, b) -> float:
    d = vec(b) - vec(a)
    return math.hypot(d[0], d[1])


def dot(u, v) -> float:
    u, v = vec(u), vec(v)
    return float(u[0] * v[0] + u[1] * v[1])


def cross(u, v) -> float:
    """z-component of the 2-D cross product."""
    u, v = vec(u), vec(v)
    return float(u[0] * v[1] - u[1] * v[0])


def sin_cos_between(s1, s2) -> tuple[float, float]:
    """(|sin θ|, |cos θ|) of the angle θ between the directions of segments s1=(a,b), s2=(c,d)."""
    u, v = direction(*s1), direction(*s2)
    return abs(cross(u, v)), abs(dot(u, v))


def foot_of_perpendicular(p, a, b) -> tuple[np.ndarray, float]:
    """Foot F of the perpendicular from p to line ab, and t with F = a + t (b - a)."""
    a, b, p = vec(a), vec(b), vec(p)
    d = b - a
    dd = dot(d, d)
    if dd < EPS:
        return a.copy(), 0.0
    t = dot(p - a, d) / dd
    return a + t * d, float(t)


def point_line_distance(p, a, b) -> float:
    """Distance from p to the infinite line through a and b (|p - a| if a == b)."""
    ln = length(a, b)
    if ln < EPS:
        return length(p, a)
    return abs(cross(vec(b) - vec(a), vec(p) - vec(a))) / ln


def angle_between_rays(a, v, b) -> float:
    """Angle a-v-b in degrees, in [0, 180]; nan if a ray has zero length."""
    u, w = vec(a) - vec(v), vec(b) - vec(v)
    nu, nw = math.hypot(*u), math.hypot(*w)
    if nu < EPS or nw < EPS:
        return float("nan")
    c = max(-1.0, min(1.0, dot(u, w) / (nu * nw)))
    return math.degrees(math.acos(c))


def midpoint(a, b) -> np.ndarray:
    return (vec(a) + vec(b)) / 2.0


def bbox_diagonal(points: Iterable) -> float:
    """Diagonal of the bounding box of the points (the diagram_scale of plan §3.2)."""
    arr = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    if len(arr) == 0:
        return 0.0
    span = arr.max(0) - arr.min(0)
    return math.hypot(span[0], span[1])
