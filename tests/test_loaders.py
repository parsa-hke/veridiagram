"""Loaders (plan §2.4): round-trip 3 diagrams per source, check invariants, render a visual check
to reports/figures/loader_check_<source>.png. Skipped when data/raw is not present (fresh clone)."""

import json
from pathlib import Path

import pytest

from data.loaders import geometry3k, pgdp5k, pgps9k
from data.schema import Diagram
from data.visualize import save_grid
from experiments.run import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
FIGS = REPO_ROOT / "reports" / "figures"
CASES = {
    # offset + known coordinate error / perpendicular+parallel / circle
    "geometry3k": (geometry3k.load, {"split": "test", "ids": ["2500", "2886"]}, {"split": "train", "ids": ["0"]}),
    # same image as geometry3k:0 / equal-angle marks / circle
    "pgdp5k": (pgdp5k.load, {"split": "train", "ids": ["4238", "1386", "4867"]}, None),
    # circle / unlabelled points / Geometry3K-origin diagram
    "pgps9k": (pgps9k.load, {"ids": ["img_0", "img_1", "img_1934"]}, None),
}

needs_raw = pytest.mark.skipif(not (RAW / "geometry3k").exists(), reason="data/raw not downloaded")


def _load(source):
    fn, a, b = CASES[source]
    out = fn(RAW, **a)
    if b:
        out += fn(RAW, **b)
    return out


def _check_refs(d: Diagram):
    names = set(d.points)
    for a, b in d.segments:
        assert a in names and b in names and a != b
    for g in d.lines:
        assert set(g) <= names and len(g) >= 2
    for c in d.circles:
        assert set(c.points_on) <= names
    for r in d.asserted:
        for ref in r.args:
            if ref[0] in ("pt", "seg", "angle"):
                assert set(ref[1:]) <= names, (d.id, r)
            elif ref[0] == "circle":
                assert 0 <= ref[1] < len(d.circles)


@needs_raw
@pytest.mark.parametrize("source", list(CASES))
def test_loader_roundtrip_and_render(source):
    ds = _load(source)
    assert len(ds) == 3
    for d in ds:
        assert d.source == source and d.width > 0 and d.height > 0
        assert (RAW / d.image_path).exists()
        _check_refs(d)
        again = Diagram.from_json(d.to_json())
        assert again.to_dict() == json.loads(d.to_json())
        assert [r.key() for r in again.asserted] == [r.key() for r in d.asserted]
    out = save_grid(ds, RAW, FIGS / f"loader_check_{source}.png")
    assert out.exists() and out.stat().st_size > 10_000


@needs_raw
def test_geometry3k_offset_recovered():
    d = geometry3k.load(RAW, split="train", ids=["0"])[0]
    assert d.flags["offset_inliers"] == "6/6"
    dx, dy = d.flags["offset"]
    assert abs(dx - 123) < 1.5 and abs(dy - 124) < 1.5
    assert all(0 <= x <= d.width and 0 <= y <= d.height for x, y in d.points.values())


@needs_raw
def test_pgdp_parser_matches_authors_logic_forms():
    """Parallel and equal-length marks agree exactly with PGDP5K's own derived logic forms."""
    lf = json.loads((RAW / "pgdp5k/PGDP5K/our_diagram_logic_forms_annot.json").read_text())
    for key in ["4238", "1386", "1009", "1766", "120"]:
        split = next(s for s in ("train", "val", "test") if key in json.loads(
            (RAW / f"pgdp5k/PGDP5K/annotations/{s}.json").read_text()))
        d = pgdp5k.load(RAW, split=split, ids=[key])[0]
        n_eq = sum(f.startswith("Equals(LengthOf(Line") and f.count("LengthOf") == 2 for f in lf[key]["diagram_logic_forms"])
        n_par = sum(f.startswith("Parallel(") for f in lf[key]["diagram_logic_forms"])
        assert sum(r.type == "equal_length" for r in d.asserted) == n_eq, key
        assert sum(r.type == "parallel" for r in d.asserted) == n_par, key
