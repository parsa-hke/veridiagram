"""verifier/primitives.py and verifier/checks.py on hand-constructed coordinates (plan §3.3):
for every check an exact positive, an exact negative and a boundary case (just inside / outside)."""

import math

import pytest

from data.schema import Circle, Diagram, angle, circle_ref, pt, seg
from verifier import checks as C
from verifier import primitives as P

TOL_DEG, TOL_REL = 2.0, 0.05
IN, OUT = 0.999, 1.001  # boundary factors


def ray(deg, length=100.0, origin=(0.0, 0.0)):
    r = math.radians(deg)
    return origin, (origin[0] + length * math.cos(r), origin[1] + length * math.sin(r))


def test_primitives():
    assert P.length((0, 0), (3, 4)) == 5.0
    assert P.cross((1, 0), (0, 1)) == 1.0 and P.dot((1, 2), (3, 4)) == 11.0
    assert P.point_line_distance((0, 5), (-1, 0), (1, 0)) == 5.0
    f, t = P.foot_of_perpendicular((2, 7), (0, 0), (4, 0))
    assert tuple(f) == (2.0, 0.0) and t == 0.5
    assert P.angle_between_rays((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)
    assert P.bbox_diagonal([(0, 0), (3, 4), (1, 1)]) == 5.0
    assert P.vec((1, 2)).dtype.name == "float64"
    assert math.isnan(P.angle_between_rays((0, 0), (0, 0), (1, 0)))


def test_parallel():
    assert C.parallel(ray(30), ray(30, origin=(0, 50)), TOL_DEG)[0]
    assert C.parallel(ray(30), ray(210, origin=(0, 50)), TOL_DEG)[0]   # opposite direction
    assert not C.parallel(ray(0), ray(45), TOL_DEG)[0]
    assert C.parallel(ray(0), ray(TOL_DEG * IN), TOL_DEG)[0]
    assert not C.parallel(ray(0), ray(TOL_DEG * OUT), TOL_DEG)[0]
    assert C.parallel(ray(0), ray(10))[1] == pytest.approx(math.sin(math.radians(10)))
    assert not C.parallel(((0, 0), (10, 0)), ((30, 0.1), (50, 0.1)), TOL_DEG)[0]     # one line, not parallel


def test_perpendicular():
    assert C.perpendicular(ray(20), ray(110), TOL_DEG)[0]
    assert not C.perpendicular(ray(20), ray(60), TOL_DEG)[0]
    assert C.perpendicular(ray(0), ray(90 - TOL_DEG * IN), TOL_DEG)[0]
    assert not C.perpendicular(ray(0), ray(90 - TOL_DEG * OUT), TOL_DEG)[0]


def test_equal_length():
    assert C.equal_length(((0, 0), (10, 0)), ((5, 5), (5, 15)), TOL_REL)[0]
    assert not C.equal_length(((0, 0), (10, 0)), ((0, 0), (5, 0)), TOL_REL)[0]
    assert C.equal_length(((0, 0), (100, 0)), ((0, 0), (100 * (1 - TOL_REL * IN), 0)), TOL_REL)[0]
    assert not C.equal_length(((0, 0), (100, 0)), ((0, 0), (100 * (1 - TOL_REL * OUT), 0)), TOL_REL)[0]


def test_on_circle():
    c = ((0.0, 0.0), 50.0)
    assert C.on_circle((30, 40), c, TOL_REL)[0]
    assert not C.on_circle((10, 10), c, TOL_REL)[0]
    assert C.on_circle((50 * (1 + TOL_REL * IN), 0), c, TOL_REL)[0]
    assert not C.on_circle((50 * (1 + TOL_REL * OUT), 0), c, TOL_REL)[0]
    assert C.on_circle((1, 1), ((0, 0), None), TOL_REL) == (False, C.NAN) or math.isnan(C.on_circle((1, 1), ((0, 0), None), TOL_REL)[1])


def test_tangent():
    c = ((0.0, 0.0), 50.0)
    assert C.tangent(((-40, 50), (40, 50)), c, TOL_REL)[0]                    # touches at (0, 50)
    assert not C.tangent(((-40, 20), (40, 20)), c, TOL_REL)[0]                # secant
    assert not C.tangent(((10, 50), (90, 50)), c, TOL_REL)[0]                 # tangent line, foot off the segment
    y_in, y_out = 50 * (1 + TOL_REL * IN), 50 * (1 + TOL_REL * OUT)
    assert C.tangent(((-40, y_in), (40, y_in)), c, TOL_REL)[0]
    assert not C.tangent(((-40, y_out), (40, y_out)), c, TOL_REL)[0]
    assert C.tangent(((-80, 51), (0.5, 51)), c, TOL_REL)[0]                    # tangency at an endpoint, annotated 1 px off
    chord_half = 50 * math.sin(math.radians(20))                                # short chord: |d - r| / r = 1 - cos 20° = 6%
    y = 50 * math.cos(math.radians(20))
    assert C.tangent(((-chord_half, y), (chord_half, y)), c, 0.08)[1] < 0.08
    assert not C.tangent(((-chord_half, y), (chord_half, y)), c, 0.08)[0]       # chord, not tangent


def test_collinear():
    assert C.collinear((5, 5), (0, 0), (10, 10), 100.0, TOL_REL)[0]
    assert not C.collinear((0, 50), (0, 0), (100, 0), 100.0, TOL_REL)[0]
    assert C.collinear((50, 100 * TOL_REL * IN), (0, 0), (100, 0), 100.0, TOL_REL)[0]
    assert not C.collinear((50, 100 * TOL_REL * OUT), (0, 0), (100, 0), 100.0, TOL_REL)[0]


def test_midpoint():
    s = ((0, 0), (100, 0))
    assert C.midpoint((50, 0), s, TOL_REL)[0]
    assert not C.midpoint((20, 0), s, TOL_REL)[0]
    assert C.midpoint((50 + 100 * TOL_REL * IN, 0), s, TOL_REL)[0]
    assert not C.midpoint((50 + 100 * TOL_REL * OUT, 0), s, TOL_REL)[0]


def test_equal_angle():
    a1 = ((10, 0), (0, 0), (0, 10))                                            # 90 deg
    assert C.equal_angle(a1, ((0, 5), (5, 5), (5, 0)), TOL_DEG)[0]
    assert not C.equal_angle(a1, ((10, 0), (0, 0), (10, 10)), TOL_DEG)[0]     # 45 deg
    v = (0.0, 0.0)
    assert C.equal_angle(a1, (ray(0)[1], v, ray(90 - TOL_DEG * IN)[1]), TOL_DEG)[0]
    assert not C.equal_angle(a1, (ray(0)[1], v, ray(90 - TOL_DEG * OUT)[1]), TOL_DEG)[0]


def test_perp_bisector():
    s2 = ((0, 0), (100, 0))
    assert C.perp_bisector(((50, -30), (50, 30)), s2, TOL_DEG, TOL_REL)[0]
    assert not C.perp_bisector(((20, -30), (20, 30)), s2, TOL_DEG, TOL_REL)[0]   # perpendicular, off-centre
    assert not C.perp_bisector(((40, -30), (60, 30)), s2, TOL_DEG, TOL_REL)[0]   # through midpoint, not perpendicular
    x_in, x_out = 50 + 100 * TOL_REL * IN, 50 + 100 * TOL_REL * OUT
    assert C.perp_bisector(((x_in, -30), (x_in, 30)), s2, TOL_DEG, TOL_REL)[0]
    assert not C.perp_bisector(((x_out, -30), (x_out, 30)), s2, TOL_DEG, TOL_REL)[0]


def test_margin_only_when_no_tolerance():
    ok, m = C.parallel(ray(0), ray(5))
    assert ok is None and m > 0


def test_evaluate_resolves_schema_refs():
    d = Diagram(id="synthetic:0", source="synthetic", split="train", image_path="", width=200, height=200,
                points={"A": (0, 0), "B": (100, 0), "C": (0, 70), "D": (100, 70), "O": (50, 120), "T": (50, 70)},
                circles=[Circle("O", (50.0, 120.0), 50.0, ["T"]), Circle(None, None, None)])
    tol = {"parallel_deg": TOL_DEG, "on_circle_rel": TOL_REL, "tangent_rel": TOL_REL, "equal_angle_deg": TOL_DEG}
    assert C.evaluate(d, "parallel", [seg("A", "B"), seg("C", "D")], tol)[0]
    assert C.evaluate(d, "on_circle", [pt("T"), circle_ref(0)], tol)[0]
    assert C.evaluate(d, "tangent", [seg("C", "D"), circle_ref(0)], tol)[0]
    assert C.evaluate(d, "equal_angle", [angle("B", "A", "C"), angle("A", "B", "D")], tol)[0]
    assert C.evaluate(d, "on_circle", [pt("T"), circle_ref(1)], tol)[0] is False
