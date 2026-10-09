"""Figure-level, leak-free splits of the real data (plan §0.2 #9, revised 2026-10-10).

Inputs: data/processed/{geometry3k,pgdp5k,pgps9k}.jsonl and figure_clusters.csv (data.overlap).
A "figure" is a duplicate-image cluster; every output holds one diagram per figure. Evaluation
outputs (val, test, pgdp5k_heldout) share no figure with any other output; the only allowed overlap
is train <-> pgps9k_pool, which is on the training side (recommendation (d)).

(a) Geometry3K figures are re-split train/val/test by `data.split_fractions` with `data.split_seed`,
    stratified by whether the figure has a circle.
(c) A Geometry3K figure takes its annotation from PGDP5K when PGDP5K re-annotated it (pixel-accurate,
    labels from drawn marks); otherwise from Geometry3K. `meta.annotation` records which.
(b) Held-out source test: PGDP5K test figures that have no Geometry3K copy.
(d) Part B pool: PGPS9K figures not in any val/test output (`meta.in_train` marks the ones that are
    Geometry3K training figures).

Writes data/processed/{geometry3k_figures,pgdp5k_heldout,pgps9k_pool}.jsonl and
data/processed/splits/{train,val,test,pgdp5k_heldout,pgps9k_pool}.txt plus figures.csv.

    python -m data.splits
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from data.schema import Diagram, read_jsonl, write_jsonl
from experiments.run import REPO_ROOT, load_config

SPLIT_FILES = ("train", "val", "test", "pgdp5k_heldout", "pgps9k_pool")


def _native(i: str) -> tuple:
    n = i.split(":", 1)[1].removeprefix("img_")
    return (0, int(n), "") if n.isdigit() else (1, 0, n)


def _copy(d: Diagram, **changes) -> Diagram:
    out = Diagram.from_dict(d.to_dict())
    for k, v in changes.items():
        setattr(out, k, v)
    return out


def assign(clusters: list[str], has_circle: dict[str, bool], fractions, seed: int) -> dict[str, str]:
    """Seeded, circle-stratified assignment of clusters to train/val/test."""
    rng = random.Random(seed)
    out = {}
    for stratum in (False, True):
        cs = sorted(c for c in clusters if has_circle[c] == stratum)
        rng.shuffle(cs)
        n_train = round(fractions[0] * len(cs))
        n_val = round(fractions[1] * len(cs))
        for k, c in enumerate(cs):
            out[c] = "train" if k < n_train else "val" if k < n_train + n_val else "test"
    return out


def make_splits(diagrams: dict[str, list[Diagram]], cluster_of: dict[str, str],
                fractions=(0.7, 0.1, 0.2), seed: int = 0) -> dict[str, list[Diagram]]:
    by_id = {d.id: d for ds in diagrams.values() for d in ds}
    members = defaultdict(list)
    for i in by_id:
        members[cluster_of[i]].append(i)

    def of(c, src):
        return sorted((i for i in members[c] if i.startswith(src + ":")), key=_native)

    # (a) + (c): one diagram per Geometry3K figure
    g3k_clusters = sorted(c for c in members if of(c, "geometry3k"))
    reps = {}
    for c in g3k_clusters:
        g_ids, p_ids = of(c, "geometry3k"), of(c, "pgdp5k")
        base = by_id[p_ids[0]] if p_ids else by_id[g_ids[0]]
        reps[c] = _copy(base, id=g_ids[0], source="geometry3k",
                        meta={**base.meta, "annotation": base.source, "figure": c,
                              "geometry3k_ids": g_ids, "pgdp5k_ids": p_ids,
                              "published_splits": sorted({by_id[i].split for i in g_ids})})
    split_of = assign(g3k_clusters, {c: bool(reps[c].circles) for c in g3k_clusters}, fractions, seed)
    for c, d in reps.items():
        d.split = split_of[c]

    # (b) held-out source: PGDP5K test figures without any Geometry3K copy
    heldout = []
    for c in sorted(members):
        p_test = [i for i in of(c, "pgdp5k") if by_id[i].split == "test"]
        if p_test and not of(c, "geometry3k"):
            d = by_id[p_test[0]]
            heldout.append(_copy(d, meta={**d.meta, "annotation": "pgdp5k", "figure": c}))

    # (d) PGPS9K pool: not in any val/test figure
    eval_figs = {c for c, s in split_of.items() if s in ("val", "test")} | {d.meta["figure"] for d in heldout}
    pool = []
    for c in sorted(members):
        ps = of(c, "pgps9k")
        if ps and c not in eval_figs:
            d = by_id[ps[0]]
            pool.append(_copy(d, meta={**d.meta, "figure": c, "in_train": split_of.get(c) == "train"}))

    order = lambda ds: sorted(ds, key=lambda d: _native(d.id))  # noqa: E731
    return {
        "train": order(d for d in reps.values() if d.split == "train"),
        "val": order(d for d in reps.values() if d.split == "val"),
        "test": order(d for d in reps.values() if d.split == "test"),
        "pgdp5k_heldout": order(heldout),
        "pgps9k_pool": order(pool),
    }


ALLOWED_OVERLAP = {frozenset(("train", "pgps9k_pool"))}


def check_disjoint(splits: dict[str, list[Diagram]]) -> None:
    """No diagram id in two outputs; no figure in two outputs except train <-> pgps9k_pool."""
    ids, figs = defaultdict(set), defaultdict(set)
    for name, ds in splits.items():
        for d in ds:
            ids[d.id].add(name)
            figs[d.meta["figure"]].add(name)
    for key, names in list(ids.items()) + list(figs.items()):
        if len(names) > 1 and frozenset(names) not in ALLOWED_OVERLAP:
            raise AssertionError(f"{key} in {sorted(names)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc = REPO_ROOT / cfg["paths"]["processed"]
    split_dir = REPO_ROOT / cfg["paths"]["splits"]
    with open(proc / "figure_clusters.csv") as f:
        cluster_of = {r["id"]: r["cluster"] for r in csv.DictReader(f)}
    diagrams = {s: list(read_jsonl(proc / f"{s}.jsonl")) for s in ("geometry3k", "pgdp5k", "pgps9k")}
    splits = make_splits(diagrams, cluster_of, cfg["data"]["split_fractions"], cfg["data"]["split_seed"])
    check_disjoint(splits)

    write_jsonl(splits["train"] + splits["val"] + splits["test"], proc / "geometry3k_figures.jsonl")
    write_jsonl(splits["pgdp5k_heldout"], proc / "pgdp5k_heldout.jsonl")
    write_jsonl(splits["pgps9k_pool"], proc / "pgps9k_pool.jsonl")
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in SPLIT_FILES:
        (split_dir / f"{name}.txt").write_text("".join(d.id + "\n" for d in splits[name]))
        rows += [(d.id, d.meta["figure"], name, d.meta.get("annotation", d.source)) for d in splits[name]]
    with open(split_dir / "figures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "figure", "split", "annotation"])
        w.writerows(rows)

    for name in SPLIT_FILES:
        ds = splits[name]
        ann = Counter(d.meta.get("annotation", d.source) for d in ds)
        extra = f", in_train {sum(d.meta['in_train'] for d in ds)}" if name == "pgps9k_pool" else ""
        print(f"{name:15s} {len(ds):5d} figures  annotation {dict(ann)}  with circle {sum(bool(d.circles) for d in ds)}{extra}")
    print(f"manifests -> {split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
