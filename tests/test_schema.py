"""data/schema.py and data/loaders/common.py: canonical relations, JSON round trip, derivations."""

import pytest

from data.loaders.common import close_equivalences, derive_tier2, order_along, segments_from_lines
from data.schema import Circle, Diagram, Relation, angle, circle_ref, pt, read_jsonl, seg, write_jsonl


def test_relation_canonical_order():
    a = Relation("parallel", [["seg", "D", "C"], ["seg", "B", "A"]])
    b = Relation("parallel", [seg("A", "B"), seg("C", "D")])
    assert a.key() == b.key() and a.args == [["seg", "A", "B"], ["seg", "C", "D"]]
    assert Relation("equal_angle", [angle("C", "B", "A"), angle("F", "E", "D")]).args[0] == ["angle", "A", "B", "C"]
    assert Relation("collinear", [pt("M"), pt("B"), pt("A")]).args == [["pt", "M"], ["pt", "A"], ["pt", "B"]]
    with pytest.raises(ValueError):
        Relation("bogus", [])


def _diagram():
    return Diagram(
        id="synthetic:1", source="synthetic", split="train", image_path="x.png", width=100, height=80,
        points={"A": (0.0, 0.0), "B": (10.0, 0.0), "O": (50.0, 40.0), "P": (60.0, 40.0)},
        segments=[("A", "B")], lines=[["A", "B"]],
        circles=[Circle(center="O", center_xy=(50.0, 40.0), radius=10.0, points_on=["P"])],
        asserted=[Relation("on_circle", [pt("P"), circle_ref(0)])], flags={"not_to_scale": False})


def test_diagram_json_roundtrip(tmp_path):
    d = _diagram()
    assert Diagram.from_json(d.to_json()).to_dict() == d.to_dict()
    assert write_jsonl([d, d], tmp_path / "x.jsonl") == 2
    back = list(read_jsonl(tmp_path / "x.jsonl"))
    assert len(back) == 2 and back[0].circles[0].center_xy == (50.0, 40.0)
    with pytest.raises(ValueError):
        Diagram(id="x", source="nope", split="train", image_path="", width=1, height=1)


def test_equivalence_closure():
    rels = [Relation("equal_length", [seg("A", "B"), seg("C", "D")]),
            Relation("equal_length", [seg("C", "D"), seg("E", "F")])]
    out = close_equivalences(rels)
    assert sorted(r.key() for r in out) == sorted(Relation("equal_length", [x, y]).key() for x, y in [
        (seg("A", "B"), seg("C", "D")), (seg("A", "B"), seg("E", "F")), (seg("C", "D"), seg("E", "F"))])


def test_midpoint_and_perp_bisector_derivation():
    lines = [["A", "M", "B"], ["M", "P"]]
    rels = [Relation("equal_length", [seg("A", "M"), seg("M", "B")]),
            Relation("perpendicular", [seg("M", "P"), seg("A", "B")])]
    types = {r.type: r for r in derive_tier2(rels, lines)}
    assert types["midpoint"].args == [pt("M"), seg("A", "B")]
    assert types["perp_bisector"].args == [seg("M", "P"), seg("A", "B")]


def test_order_along_and_segments():
    pts = {"A": (0, 0), "B": (10, 10), "M": (5, 5)}
    assert order_along(["B", "A", "M"], pts) in (["A", "M", "B"], ["B", "M", "A"])
    assert segments_from_lines([["A", "M", "B"]]) == [("A", "B"), ("A", "M"), ("B", "M")]
