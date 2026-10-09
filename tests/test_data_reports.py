"""data/overlap.py, data/sanity.py, data/census.py on small hand-made inputs."""

import csv

import numpy as np
from PIL import Image

from data import census, overlap, sanity
from data.schema import Diagram, Relation, pt, seg, write_jsonl


def test_sanity_outside():
    assert not sanity.outside(0, 0, 10, 10) and not sanity.outside(11.5, 5, 10, 10)
    assert sanity.outside(-5, 5, 10, 10) and sanity.outside(5, 20, 10, 10)


def test_overlap_confirm(tmp_path):
    rng = np.random.default_rng(0)
    a = np.full((120, 160), 255, np.uint8)
    a[20:24, 10:150] = 0
    a[20:100, 70:74] = 0
    b = np.full((120, 160), 255, np.uint8)
    for _ in range(6):
        y, x = rng.integers(0, 110), rng.integers(0, 150)
        b[y:y + 4, x:x + 10] = 0
    for name, arr in (("a.png", a), ("a2.png", a), ("b.png", b)):
        Image.fromarray(arr).save(tmp_path / name)
    item = lambda n, names: {"path": str(tmp_path / n), "names": frozenset(names)}  # noqa: E731
    assert overlap.confirm(item("a.png", "AB"), item("a2.png", "AB"))[0]
    assert not overlap.confirm(item("a.png", "AB"), item("b.png", "CD"))[0]


def test_census_counts_unique_figures(tmp_path):
    def d(i, split):
        return Diagram(id=f"geometry3k:{i}", source="geometry3k", split=split, image_path="", width=10, height=10,
                       points={"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)},
                       asserted=[Relation("parallel", [seg("A", "B"), seg("C", "D")]),
                                 Relation("collinear", [pt("A"), pt("B"), pt("C")])])
    write_jsonl([d(1, "train"), d(2, "train"), d(3, "test")], tmp_path / "geometry3k.jsonl")
    with open(tmp_path / "figure_clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "cluster"])
        w.writerows([["geometry3k:1", "geometry3k", "train", "f0"], ["geometry3k:2", "geometry3k", "train", "f0"],
                     ["geometry3k:3", "geometry3k", "test", "f1"]])
    order, cols = census.census(tmp_path, sources=("geometry3k",))
    assert cols["geometry3k_train"]["parallel"] == 2 and cols["geometry3k_train_unique"]["parallel"] == 1
    assert cols["geometry3k_total"]["_diagrams"] == 3 and cols["geometry3k_total_unique"]["_diagrams"] == 2
