"""data/synthetic/{generator,render,build,check}.py (plan Phase 4)."""

import math

import numpy as np
from PIL import Image

from data.queries import asserted_lookup, enumerate_queries
from data.synthetic import build, check
from data.synthetic.generator import generate, generate_one
from data.synthetic.render import render
from experiments.run import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")
S = CFG["synthetic"]


def test_counts_and_determinism():
    ds = generate(60, 0, S)
    for d in ds:
        assert S["n_points"][0] <= len(d.points) <= S["n_points"][1]
        assert len(d.meta["drawn"]) <= S["n_segments"][1]
        assert len(d.circles) <= S["n_circles"][1]
        xy = np.array(list(d.meta["exact_points"].values()))
        dist = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)) + np.eye(len(xy)) * 1e9
        assert dist.min() >= S["min_separation_px"] - 1e-6
    assert generate_one(S, "train", 5, 0).to_json() == generate_one(S, "train", 5, 0).to_json()
    assert generate_one(S, "train", 5, 0).to_json() != generate_one(S, "train", 6, 0).to_json()
    assert sum(sum(d.meta["constructions"].values()) for d in ds) > 30


def test_extrap_ranges():
    ds = generate(20, 0, S, pool="extrap", n_points=tuple(S["extrap_points"]), n_segments=tuple(S["extrap_segments"]))
    assert all(S["extrap_points"][0] <= len(d.points) <= S["extrap_points"][1] for d in ds)


def test_constructions_exact_before_jitter():
    worst = check.exact_margins(generate(80, 1, S))
    assert worst and max(worst.values()) < 1e-6


def test_injected_relations_are_queries():
    for d in generate(40, 2, S):
        f = asserted_lookup(d)
        qs = enumerate_queries(d, ("collinear",))
        for r in d.asserted:
            if r.type in ("parallel", "perpendicular", "on_circle", "tangent", "collinear"):
                assert any(f(q, r.type) for q in qs if r.type in q.relations), (d.id, r)


def test_render(tmp_path):
    d = generate_one(S, "train", 3, 0)
    st = render(d, tmp_path / "x.png", S["render"], d.meta["seed"])
    im = Image.open(tmp_path / "x.png")
    assert im.size == (S["canvas_px"], S["canvas_px"])
    a = np.asarray(im.convert("L"))
    assert (a < 128).mean() > 0.002 and set(st) >= {"line_width", "jpeg_quality", "scan_noise"}


def test_seal_hides_annotation(tmp_path):
    ds = generate(3, 0, S, pool="unlabelled")
    digest = build.seal(ds, tmp_path)
    public = (tmp_path / "synth_unlabelled.jsonl").read_text()
    assert len(digest) == 64 and '"points":{}' in public and '"asserted":[]' in public
    assert (tmp_path / "sealed" / "synth_unlabelled.jsonl").exists()
