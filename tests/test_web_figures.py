"""web/figures.py on hand-made diagrams: hero, query kinds and the unmarked-relations pick."""

import math

import numpy as np

from data.schema import Circle, Diagram, Relation, seg
from web import figures

VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]


def _diagram(i, circle=True):
    pts = {"A": (0.0, 0.0), "B": (400.0, 0.0), "C": (0.0, 200.0), "D": (400.0, 205.0), "E": (10.0, 10.0),
           "O": (200.0, 100.0), "P": (200.0, 20.0)}
    return Diagram(id=f"synthetic:extrap_{i}", source="synthetic", split="extrap", image_path="", width=512,
                   height=512, points=pts, segments=[("A", "B"), ("C", "D"), ("A", "E")],
                   circles=[Circle("O", (200.0, 100.0), 80.0, ["P"])] if circle else [])


def _row(did, kind, refs, geo, margins=None):
    mask = {"seg_pair": [1, 1, 1, 0, 0, 0], "pt_circle": [0, 0, 0, 1, 0, 0], "seg_circle": [0, 0, 0, 0, 1, 0],
            "pt_line": [0, 0, 0, 0, 0, 1]}[kind]
    return {"qid": f"{did}-{kind}-{len(refs)}", "diagram_id": did, "kind": kind, "refs": refs, "geo": geo, "mask": mask,
            "margins": margins or [0.01 if m else None for m in mask]}


def _rows(did):
    return [_row(did, "seg_pair", [seg("A", "B"), seg("C", "D")], [1, 0, 0, 0, 0, 0], [0.0125, 0.99, 0.01, None, None, None]),
            _row(did, "seg_pair", [seg("A", "B"), seg("A", "E")], [0, 0, 0, 0, 0, 0]),
            _row(did, "pt_circle", [["pt", "P"], ["circle", 0]], [0, 0, 0, 1, 0, 0]),
            _row(did, "seg_circle", [seg("A", "B"), ["circle", 0]], [0, 0, 0, 0, 0, 0]),
            _row(did, "pt_line", [["pt", "E"], ["pt", "A"], ["pt", "B"]], [0, 0, 0, 0, 0, 0])]


def test_hero_and_queries():
    d1, d2 = _diagram(1), _diagram(2, circle=False)
    rows = _rows(d1.id) + _rows(d2.id)
    probs = np.zeros((len(rows), 6))
    probs[0, 0] = probs[5, 0] = 0.97
    hero = figures.pick_hero(rows, {d1.id: d1, d2.id: d2}, probs, VOCAB)
    assert hero["diagram"]["id"] == d1.id                              # d2 has no circle
    assert hero["verdicts"]["parallel"] == {"prob": 0.97, "truth": True, "margin": 0.0125}
    assert hero["angle_off_deg"] == round(math.degrees(math.asin(0.0125)), 2)
    assert "on_circle" not in hero["verdicts"]                         # only applicable relations
    assert hero["diagram"]["circles"][0] == {"center": "O", "cx": 200.0, "cy": 100.0, "r": 80.0}
    probs[0, 0] = 0.4                                                  # the model gets it wrong: no hero
    assert figures.pick_hero(rows, {d1.id: d1, d2.id: d2}, probs, VOCAB) is None
    qs = figures.pick_queries(rows, probs, VOCAB, d1.id)
    assert [q["kind"] for q in qs] == list(figures.KINDS)
    assert qs[0]["refs"] == [seg("A", "B"), seg("C", "D")]             # the true pair is preferred


def test_unmarked_pick():
    pts = {"A": (0, 0), "B": (100, 0), "C": (0, 50), "D": (100, 50), "E": (0, 100), "F": (100, 101), "G": (0, 150),
           "H": (100, 150)}
    segs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    rel = lambda t, a, b, src: Relation(t, [seg(*a), seg(*b)], src, 0.01 if src == "geometric" else None)  # noqa: E731
    d = Diagram(id="synthetic:train_1", source="synthetic", split="train", image_path="", width=200, height=200,
                points=pts, segments=segs,
                asserted=[rel("parallel", "AB", "CD", "asserted")],
                geometric=[rel("parallel", "AB", "CD", "geometric"), rel("parallel", "AB", "EF", "geometric"),
                           rel("equal_length", "EF", "GH", "geometric"), rel("parallel", "CD", "GH", "geometric"),
                           rel("equal_length", "AB", "CD", "geometric")])
    u = figures.pick_unmarked([d], want=2)
    assert u["marked"][0]["type"] == "parallel" and len(u["unmarked"]) == 2
    assert u["unmarked"][0]["off_deg"] == round(math.degrees(math.asin(0.01)), 2)
    assert "off_pct" not in u["unmarked"][1] and figures._off("equal_length", 0.031) == {"off_pct": 3.1}
    segs_used = [s for r in u["unmarked"] for s in map(str, r["args"])]
    assert len(segs_used) == len(set(segs_used))                       # one colour per segment
    assert len(figures.pick_unmarked([d], want=3)["unmarked"]) == 2     # falls back to want - 1
    assert figures.pick_unmarked([d], want=4) is None
