"""data/splits.py: figure-level splits, PGDP5K annotation preference, leak checks (plan §5.7)."""

from pathlib import Path

import pytest

from data.schema import Circle, Diagram
from data.splits import check_disjoint, make_splits
from experiments.run import REPO_ROOT

SPLIT_DIR = REPO_ROOT / "data" / "processed" / "splits"


def _d(i, split, circle=False):
    src = i.split(":")[0]
    return Diagram(id=i, source=src, split=split, image_path=f"{i}.png", width=10, height=10,
                   points={"A": (1.0, 1.0)}, circles=[Circle("A", (1.0, 1.0), 1.0)] if circle else [])


def _toy():
    # fig0: g3k 1 (train) + g3k 7 (test) + pgdp 40 (train) + pgps 0   -> one figure, PGDP annotation
    # fig1: g3k 2 only                                                -> Geometry3K annotation
    # fig2: pgdp 50 (test), no g3k                                    -> held-out
    # fig3: pgps 1 only                                               -> pool
    # fig4: pgdp 60 (test) + pgps 2                                   -> held-out; pgps 2 excluded from pool
    diagrams = {
        "geometry3k": [_d("geometry3k:1", "train"), _d("geometry3k:7", "test"), _d("geometry3k:2", "val", circle=True)],
        "pgdp5k": [_d("pgdp5k:40", "train"), _d("pgdp5k:50", "test"), _d("pgdp5k:60", "test")],
        "pgps9k": [_d("pgps9k:img_0", "all"), _d("pgps9k:img_1", "all"), _d("pgps9k:img_2", "all")],
    }
    cluster_of = {"geometry3k:1": "f0", "geometry3k:7": "f0", "pgdp5k:40": "f0", "pgps9k:img_0": "f0",
                  "geometry3k:2": "f1", "pgdp5k:50": "f2", "pgps9k:img_1": "f3",
                  "pgdp5k:60": "f4", "pgps9k:img_2": "f4"}
    return diagrams, cluster_of


def test_one_diagram_per_figure_with_pgdp_preference():
    s = make_splits(*_toy(), fractions=(1.0, 0.0, 0.0), seed=0)
    g3k = {d.id: d for k in ("train", "val", "test") for d in s[k]}
    assert set(g3k) == {"geometry3k:1", "geometry3k:2"}          # 3 problems, 2 figures
    assert g3k["geometry3k:1"].meta["annotation"] == "pgdp5k"
    assert g3k["geometry3k:1"].image_path == "pgdp5k:40.png"
    assert g3k["geometry3k:1"].meta["geometry3k_ids"] == ["geometry3k:1", "geometry3k:7"]
    assert g3k["geometry3k:2"].meta["annotation"] == "geometry3k"
    assert [d.id for d in s["pgdp5k_heldout"]] == ["pgdp5k:50", "pgdp5k:60"]
    pool = {d.id: d.meta["in_train"] for d in s["pgps9k_pool"]}
    assert pool == {"pgps9k:img_0": True, "pgps9k:img_1": False}   # img_2 shares a held-out figure
    check_disjoint(s)


def test_eval_figure_never_in_pool():
    s = make_splits(*_toy(), fractions=(0.0, 0.0, 1.0), seed=0)     # every g3k figure -> test
    assert {d.id for d in s["pgps9k_pool"]} == {"pgps9k:img_1"}
    check_disjoint(s)


def test_check_disjoint_rejects_leak():
    s = make_splits(*_toy(), fractions=(1.0, 0.0, 0.0), seed=0)
    s["test"] = [s["train"][0]]
    with pytest.raises(AssertionError):
        check_disjoint(s)


@pytest.mark.skipif(not (SPLIT_DIR / "train.txt").exists(), reason="splits not built")
def test_real_manifests_are_disjoint():
    import csv
    rows = list(csv.DictReader(open(SPLIT_DIR / "figures.csv")))
    by_fig = {}
    for r in rows:
        by_fig.setdefault(r["figure"], set()).add(r["split"])
    for fig, names in by_fig.items():
        assert len(names) == 1 or names == {"train", "pgps9k_pool"}, (fig, names)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for name in ("train", "val", "test", "pgdp5k_heldout", "pgps9k_pool"):
        assert sum(1 for r in rows if r["split"] == name) == len((SPLIT_DIR / f"{name}.txt").read_text().split())
