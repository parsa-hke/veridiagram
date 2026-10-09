"""Duplicate images within Geometry3K and across Geometry3K / PGDP5K / PGPS9K.

PGDP5K reuses 1,813 Geometry3K images and PGPS9K reuses 1,738 (per their papers), and Geometry3K
is split by problem, not by diagram, so a "held-out" set is only held out after de-duplication.

Matching, in two stages:
1. Candidates: 64-bit difference hash (dHash, image flattened on white) within HAMMING_MAX bits.
2. Confirmation (calibrated by eye on the borderline cases, see CHANGELOG 2026-10-10):
   - mean |pixel diff| at 64x64 grayscale <= 4 (near-identical), or
   - ink IoU >= 0.6 (dark pixels at 160x160, 3x3 dilation) and the labelled point names do not
     disagree (Jaccard >= 0.8 when both have >= 2 names), or
   - ink IoU >= 0.4 and labelled point names agree (re-cropped copies of the same figure).
   Low-resolution similarity alone matched different simple triangles, so it is not enough.

Writes data/processed/overlap.csv (one row per confirmed pair), data/processed/figure_clusters.csv
(every image with its duplicate-cluster id, for leak-free splits in Phase 5.7) and
reports/tables/source_overlap.csv (images per source/split that have a duplicate elsewhere).

    python -m data.overlap
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from data.loaders.common import UnionFind
from data.schema import read_jsonl
from experiments.run import REPO_ROOT, load_config

HAMMING_MAX = 6
MAD_NEAR_IDENTICAL = 4.0
MAD_CANDIDATE = 12.0
SOURCES = ("geometry3k", "pgdp5k", "pgps9k")


@lru_cache(maxsize=4096)
def _flat(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im).convert("L")


def gray(path: str, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(_flat(path).resize(size, Image.Resampling.LANCZOS), dtype=np.float32)


def ink(path: str) -> np.ndarray:
    g = gray(path, (160, 160)) < 200
    return np.asarray(Image.fromarray(g.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0


def dhash(path: str) -> int:
    a = gray(path, (9, 8))
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def index(source: str, proc: Path, raw: Path) -> tuple[list[dict], np.ndarray]:
    items = [{"id": d.id, "split": d.split, "path": str(raw / d.image_path), "size": f"{d.width}x{d.height}",
              "names": frozenset(n for n in d.points if not n.startswith("_"))}
             for d in read_jsonl(proc / f"{source}.jsonl")]
    return items, np.array([dhash(it["path"]) for it in items], dtype=np.uint64)


def confirm(a: dict, b: dict) -> tuple[bool, float, float, float | None]:
    mad = float(np.abs(gray(a["path"], (64, 64)) - gray(b["path"], (64, 64))).mean())
    if mad > MAD_CANDIDATE:
        return False, mad, 0.0, None
    ia, ib = ink(a["path"]), ink(b["path"])
    iou = float((ia & ib).sum() / max((ia | ib).sum(), 1))
    na, nb = a["names"], b["names"]
    jac = len(na & nb) / len(na | nb) if len(na) >= 2 and len(nb) >= 2 else None
    ok = (mad <= MAD_NEAR_IDENTICAL
          or (iou >= 0.6 and (jac is None or jac >= 0.8))
          or (iou >= 0.4 and jac is not None and jac >= 0.8))
    return ok, mad, iou, jac


def match(A: tuple, B: tuple, same: bool) -> list[dict]:
    (ia, ha), (ib, hb) = A, B
    out = []
    for i, h in enumerate(ha):
        dist = np.bitwise_count(np.bitwise_xor(hb, h))
        for j in np.nonzero(dist <= HAMMING_MAX)[0]:
            if same and j <= i:
                continue
            ok, mad, iou, jac = confirm(ia[i], ib[j])
            if ok:
                out.append({"id_a": ia[i]["id"], "split_a": ia[i]["split"], "id_b": ib[j]["id"], "split_b": ib[j]["split"],
                            "dhash_dist": int(dist[j]), "mad64": round(mad, 2), "ink_iou": round(iou, 3),
                            "name_jaccard": "" if jac is None else round(jac, 3),
                            "size_a": ia[i]["size"], "size_b": ib[j]["size"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    raw, proc = REPO_ROOT / cfg["paths"]["raw"], REPO_ROOT / cfg["paths"]["processed"]
    idx = {s: index(s, proc, raw) for s in SOURCES}

    rows = match(idx["geometry3k"], idx["geometry3k"], same=True)
    for sa, sb in (("geometry3k", "pgdp5k"), ("geometry3k", "pgps9k"), ("pgdp5k", "pgps9k")):
        rows += match(idx[sa], idx[sb], same=False)
    with open(proc / "overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # hit[(source, split, other_source, other_split)] = images of (source, split) that have a
    # duplicate in (other_source, other_split)
    hit = defaultdict(set)
    for r in rows:
        sa, sb = r["id_a"].split(":")[0], r["id_b"].split(":")[0]
        hit[(sa, r["split_a"], sb, r["split_b"])].add(r["id_a"])
        hit[(sb, r["split_b"], sa, r["split_a"])].add(r["id_b"])
    sizes = {(s, it["split"]): 0 for s in SOURCES for it in idx[s][0]}
    for s in SOURCES:
        for it in idx[s][0]:
            sizes[(s, it["split"])] += 1
    out = REPO_ROOT / cfg["paths"]["tables"] / "source_overlap.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "n_images", "duplicated_in_source", "duplicated_in_split", "n_duplicated", "pct"])
        for (src, sp, other, osp), ids in sorted(hit.items()):
            n = sizes[(src, sp)]
            w.writerow([src, sp, n, other, osp, len(ids), round(100 * len(ids) / n, 1)])
    # duplicate clusters ("figures"): connected components of the confirmed pairs
    uf = UnionFind()
    for src in SOURCES:
        for it in idx[src][0]:
            uf.find(it["id"])
    for r in rows:
        uf.union(r["id_a"], r["id_b"])
    split_of = {it["id"]: it["split"] for src in SOURCES for it in idx[src][0]}
    with open(proc / "figure_clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "cluster"])
        for k, g in enumerate(sorted(uf.groups())):
            for x in g:
                w.writerow([x, x.split(":")[0], split_of[x], f"fig{k:05d}"])
    print(f"{len(rows)} confirmed pairs -> {proc / 'overlap.csv'}; {len(uf.groups())} figures -> {proc / 'figure_clusters.csv'}")
    print(open(out).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
