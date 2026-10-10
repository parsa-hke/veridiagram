"""data/queries.py, verifier/label.py, verifier/tolerance.py, verifier/audit.py, verifier/gold.py
on hand-made inputs (plan §3.4-3.8, §5.1)."""

import numpy as np

from data.queries import asserted_lookup, enumerate_queries
from data.schema import Circle, Diagram, Relation, circle_ref, pt, seg
from verifier import audit, gold, tolerance
from verifier.label import label_diagram, not_to_scale

TOL = {"parallel_deg": 2.0, "perpendicular_deg": 2.0, "equal_length_rel": 0.02, "on_circle_rel": 0.05,
       "tangent_rel": 0.05, "collinear_rel": 0.01}
NTS = {"angle_deg": 10.0, "rel_length": 0.15}


def _square(skew: float = 0.0) -> Diagram:
    # A-M-B is one line (M midpoint); ABCD a square of side 100; circle inscribed (r 50) at O
    return Diagram(
        id="synthetic:sq", source="synthetic", split="train", image_path="", width=200, height=200,
        points={"A": (0.0, 0.0), "M": (50.0, 0.0), "B": (100.0, 0.0), "C": (100.0 + skew, 100.0),
                "D": (0.0, 100.0), "O": (50.0, 50.0)},
        segments=[("A", "B"), ("A", "M"), ("B", "M"), ("B", "C"), ("C", "D"), ("A", "D")],
        lines=[["A", "M", "B"]],
        circles=[Circle("O", (50.0, 50.0), 50.0, [])],
        asserted=[Relation("perpendicular", [seg("A", "B"), seg("B", "C")]),
                  Relation("collinear", [pt("M"), pt("A"), pt("B")])])


def test_queries_kinds_and_carriers():
    d = _square()
    qs = enumerate_queries(d, ("collinear",))
    kinds = {q.kind for q in qs}
    assert kinds == {"seg_pair", "pt_circle", "seg_circle", "pt_line"}
    assert not [q for q in qs if q.refs == [seg("A", "B"), seg("A", "M")]]            # nested on one line: skipped
    same_line = [q for q in qs if q.kind == "seg_pair" and q.refs == [seg("A", "M"), seg("B", "M")]]
    assert same_line and same_line[0].relations == ("equal_length",)
    f = asserted_lookup(d)
    sub = next(q for q in qs if q.refs == [seg("A", "M"), seg("B", "C")])   # sub-segment of AB
    assert f(sub, "perpendicular") and not f(sub, "parallel")
    col = next(q for q in qs if q.kind == "pt_line" and q.refs[0] == pt("M"))
    assert f(col, "collinear")


def test_label_and_not_to_scale():
    d = _square()
    rows = label_diagram(d, TOL, NTS, ("collinear",))
    geo = {(r.type, str(r.args)) for r in d.geometric}
    assert ("parallel", str([seg("A", "B"), seg("C", "D")])) in geo
    assert ("tangent", str([seg("A", "B"), circle_ref(0)])) in geo                 # AB touches at M
    assert ("equal_length", str([seg("A", "M"), seg("B", "M")])) in geo
    assert ("collinear", str([pt("M"), pt("A"), pt("B")])) in geo
    assert all(r["margin"] != "" for r in rows) and not d.flags["not_to_scale"]
    skewed = _square(skew=30.0)                                                  # angle ABC ~ 73 deg
    assert not_to_scale(skewed, NTS) and label_diagram(skewed, TOL, NTS, ()) and skewed.flags["not_to_scale"]


def test_valley_and_knee():
    rng = np.random.default_rng(0)
    m = np.concatenate([10 ** rng.normal(-3, 0.3, 300), 10 ** rng.normal(-0.3, 0.2, 3000)])
    a = np.arange(len(m)) < 300
    v = tolerance.valley(m, a)
    assert v is not None and 1e-2 < v < 1e-1
    flat = 10 ** rng.uniform(-4, 0, 5000)                                        # no valley
    assert tolerance.valley(flat, np.arange(5000) < 500) is None
    assert 1e-4 < tolerance.knee(10 ** rng.normal(-2.5, 0.4, 400)) < 1e-1


def test_audit_table_counts():
    recs = [{"relation": "parallel", "asserted": a, "geometric": g} for a, g in
            [(True, True), (True, False), (False, True), (False, False), (False, False)]]
    row = audit.table(recs)[0]
    assert (row["asserted_and_geometric"], row["asserted_not_geometric"], row["geometric_not_asserted"], row["neither"]) == (1, 1, 1, 2)
    assert row["pct_asserted_confirmed"] == 50.0


def test_gold_helpers():
    assert sum(gold.ALLOC.values()) == 100 and set(gold.ALLOC) == set(gold.QUESTIONS)
    assert gold._split(17, 3) == [6, 6, 5]
    lo, hi = (float(x) for x in gold._wilson(1, 50).strip("[]").split(","))
    assert lo < 0.02 < hi < 0.11


def test_gold_weighted_rates():
    items = [{"gold_id": "a", "relation": "parallel", "weight": "9"},    # far negative, many like it
             {"gold_id": "b", "relation": "parallel", "weight": "1"},    # near-tolerance negative
             {"gold_id": "c", "relation": "parallel", "weight": "1"},
             {"gold_id": "d", "relation": "parallel", "weight": "1"}]
    labels = {"a": {"label": "no"}, "b": {"label": "no"}, "c": {"label": "yes"}, "d": {"label": "unsure"}}
    verdict = {"a": "no", "b": "yes", "c": "yes", "d": "no"}
    raw = gold._rows("r", items, labels, verdict)[-1]
    w = gold._rows("r", items, labels, verdict, weighted=True)[-1]
    assert (raw["fp"], raw["tn"], raw["unsure"]) == (1, 1, 1) and raw["fp_rate"] == "0.500"
    assert w["fp_rate"] == "0.100" and w["fn_rate"] == "0.000"


def test_gold_adjudication(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(gold, "GOLD_DIR", tmp_path)
    (tmp_path / "adjudication.csv").write_text(
        "gold_id,raw_label,adjudicated_label,reason,confirmed_by,date\nb,yes,no,slip,labeller,2026-10-10\n")
    raw = {"a": {"gold_id": "a", "label": "yes"}, "b": {"gold_id": "b", "label": "yes"}}
    adj = gold.adjudicated(raw)
    assert adj["b"]["label"] == "no" and raw["b"]["label"] == "yes" and adj["a"]["label"] == "yes"
    with pytest.raises(AssertionError):
        gold.adjudicated({"b": {"gold_id": "b", "label": "no"}})      # raw label must match the record
