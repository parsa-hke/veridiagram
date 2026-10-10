"""Loader: Geometry3K (Lu et al., 2021; Inter-GPS) raw files -> `Diagram` (plan §2.4).

Layout under <raw>/geometry3k/: {train,val,test}/<id>/ with
  img_diagram.png        the diagram (RGBA)
  img_diagram_point.png  same image with each annotated point drawn as a pure-red (255,0,0)
                         dot and its name in pure blue
  data.json              problem text, choices, img_width, img_height, source, comment, ...
  logic_form.json        text_logic_form, dissolved_text_logic_form, diagram_logic_form,
                         line_instances ("AB" strings), point_positions {name: [x, y]},
                         circle_instances (centre names, "" for none)
plus logic_forms/*.json (the same diagram logic forms keyed by id) and symbols/<id>.{png,xml}
(VOC boxes of symbols and text; not used here).

`point_positions` are NOT image pixels: they are shifted by a per-diagram offset (e.g. +123/+124 px
on id 0). The offset is recovered by matching the positions to the red dots in
img_diagram_point.png; the recovered offset and inlier count are stored in `flags`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from data.loaders.common import close_equivalences, dedupe, derive_tier2, order_along, segments_from_lines
from data.schema import Circle, Diagram, Relation, angle, circle_ref, pt, seg

SPLITS = ("train", "val", "test")
NAME = r"[A-Za-z0-9']+"
LINE = rf"Line\(\s*({NAME})\s*,\s*({NAME})\s*\)"
RE = {
    "on_line": re.compile(rf"^PointLiesOnLine\(\s*({NAME})\s*,\s*{LINE}\s*\)$"),
    "on_circle": re.compile(rf"^PointLiesOnCircle\(\s*({NAME})\s*,\s*Circle\(\s*({NAME})\s*(?:,\s*[\w.]+)?\s*\)\s*\)$"),
    "perpendicular": re.compile(rf"^Perpendicular\(\s*{LINE}\s*,\s*{LINE}\s*\)$"),
    "parallel": re.compile(rf"^Parallel\(\s*{LINE}\s*,\s*{LINE}\s*\)$"),
    "tangent": re.compile(rf"^Tangent\(\s*{LINE}\s*,\s*Circle\(\s*({NAME})\s*(?:,\s*[\w.]+)?\s*\)\s*\)$"),
    "equal_length": re.compile(rf"^Equals\(\s*LengthOf\(\s*{LINE}\s*\)\s*,\s*LengthOf\(\s*{LINE}\s*\)\s*\)$"),
    "equal_angle": re.compile(
        rf"^Equals\(\s*MeasureOf\(\s*Angle\(\s*({NAME})\s*,\s*({NAME})\s*,\s*({NAME})\s*\)\s*\)\s*,"
        rf"\s*MeasureOf\(\s*Angle\(\s*({NAME})\s*,\s*({NAME})\s*,\s*({NAME})\s*\)\s*\)\s*\)$"),
    "angle": re.compile(rf"Angle\(\s*({NAME})\s*,\s*({NAME})\s*,\s*({NAME})\s*\)"),
}


def load(raw_dir: str | Path, split: str | None = None, ids: Iterable[str] | None = None,
         limit: int | None = None, recover_offset: bool = True) -> list[Diagram]:
    root = Path(raw_dir) / "geometry3k"
    want = set(map(str, ids)) if ids is not None else None
    out = []
    for sp in ([split] if split else SPLITS):
        dirs = sorted((p for p in (root / sp).iterdir() if p.is_dir()), key=lambda p: int(p.name))
        for d in dirs:
            if want is not None and d.name not in want:
                continue
            out.append(load_one(d, sp, recover_offset=recover_offset))
            if limit and len(out) >= limit:
                return out
    return out


def load_one(d: Path, split: str, recover_offset: bool = True) -> Diagram:
    data = json.loads((d / "data.json").read_text())
    lf = json.loads((d / "logic_form.json").read_text())
    flags: dict = {}
    raw_pts = {k: (float(v[0]), float(v[1])) for k, v in lf["point_positions"].items()}

    with Image.open(d / "img_diagram.png") as im:
        width, height = im.size
    if (width, height) != (data["img_width"], data["img_height"]):
        flags["size_mismatch_data_json"] = [data["img_width"], data["img_height"]]

    dx = dy = 0.0
    if recover_offset and raw_pts:
        res = find_offset(d / "img_diagram_point.png", d / "img_diagram.png", raw_pts)
        if res is None:
            flags["offset_unresolved"] = True
        else:
            dx, dy, inl, ndots = res
            flags["offset"] = [round(dx, 2), round(dy, 2)]
            flags["offset_inliers"] = f"{inl}/{len(raw_pts)}"
            if inl < len(raw_pts):
                flags["offset_partial"] = True
    points = {k: (x + dx, y + dy) for k, (x, y) in raw_pts.items()}

    known = set(points)
    missing: set[str] = set()

    def ok(*names: str) -> bool:
        bad = [n for n in names if n not in known]
        missing.update(bad)
        return not bad

    # ---- logic forms
    rels: list[Relation] = []
    on_line = defaultdict(set)
    circles_by_center: dict[str, int] = {}
    circles: list[Circle] = []
    angles = set()
    unparsed = 0

    def circle(cn: str) -> int:
        if cn not in circles_by_center:
            circles_by_center[cn] = len(circles)
            circles.append(Circle(center=cn, center_xy=points.get(cn), radius=None, points_on=[]))
        return circles_by_center[cn]

    for c in lf.get("circle_instances", []):
        if c:
            circle(c)
    for f in lf.get("diagram_logic_form", []):
        f = f.strip()
        if not f:
            continue
        for a, v, b in RE["angle"].findall(f):
            if ok(a, v, b):
                angles.add(tuple(angle(a, v, b)[1:]))
        if m := RE["on_line"].match(f):
            p, a, b = m.groups()
            if ok(p, a, b):
                on_line[tuple(sorted((a, b)))].add(p)
                rels.append(Relation("collinear", [pt(p), pt(a), pt(b)]))
        elif m := RE["on_circle"].match(f):
            p, c = m.groups()
            i = circle(c)
            if ok(p):
                circles[i].points_on.append(p)
                rels.append(Relation("on_circle", [pt(p), circle_ref(i)]))
        elif m := RE["perpendicular"].match(f):
            a, b, c, e = m.groups()
            if ok(a, b, c, e):
                rels.append(Relation("perpendicular", [seg(a, b), seg(c, e)]))
        elif m := RE["parallel"].match(f):
            a, b, c, e = m.groups()
            if ok(a, b, c, e):
                rels.append(Relation("parallel", [seg(a, b), seg(c, e)]))
        elif m := RE["tangent"].match(f):
            a, b, c = m.groups()
            i = circle(c)
            if ok(a, b):
                rels.append(Relation("tangent", [seg(a, b), circle_ref(i)]))
        elif m := RE["equal_length"].match(f):
            a, b, c, e = m.groups()
            if ok(a, b, c, e):
                rels.append(Relation("equal_length", [seg(a, b), seg(c, e)]))
        elif m := RE["equal_angle"].match(f):
            a, v, b, c, w, e = m.groups()
            if ok(a, v, b, c, w, e):
                rels.append(Relation("equal_angle", [angle(a, v, b), angle(c, w, e)]))
        elif not f.startswith("Equals("):
            unparsed += 1   # Equals(...) with values/areas/numbered angles carry no relation
    if unparsed:
        flags["unparsed_logic_forms"] = unparsed

    for c in circles:
        c.points_on = sorted(set(c.points_on))
        if c.center_xy is not None and c.points_on:
            cx, cy = c.center_xy
            c.radius = float(np.median([np.hypot(points[p][0] - cx, points[p][1] - cy) for p in c.points_on]))
        else:
            flags["circle_without_radius"] = flags.get("circle_without_radius", 0) + 1

    # ---- lines and segments
    lines = [order_along(sorted(g), points) for g in _merge_groups([{a, b} | ps for (a, b), ps in on_line.items()])]
    extra, bad_instances = [], 0
    for s in lf.get("line_instances", []):
        if not s.strip():
            continue
        pair = _split_instance(s, known | missing)
        if pair is None:
            bad_instances += 1
        elif ok(*pair):
            extra.append(pair)
    if bad_instances:
        flags["unparsed_line_instances"] = bad_instances
    if missing:
        flags["missing_points"] = sorted(missing)

    rels = derive_tier2(close_equivalences(dedupe(rels)), lines)
    return Diagram(
        id=f"geometry3k:{d.name}", source="geometry3k", split=split,
        image_path=f"geometry3k/{split}/{d.name}/img_diagram.png", width=width, height=height,
        points=points, segments=segments_from_lines(lines, extra), lines=lines, circles=circles,
        angles=sorted(angles), asserted=rels, flags=flags,
        meta={"native_id": int(d.name), "raw_point_positions": {k: list(v) for k, v in raw_pts.items()},
              "comment": data.get("comment", ""), "textbook_source": data.get("source", "")},
    )


def _merge_groups(groups: list[set[str]]) -> list[set[str]]:
    """Merge collinear groups that share two or more points (one line stated with different endpoints)."""
    groups = [set(g) for g in groups]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if len(groups[i] & groups[j]) >= 2:
                    groups[i] |= groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    return groups


def _split_instance(s: str, names: set[str]) -> tuple[str, str] | None:
    """Split a line instance like "AB" or "A'B" into two known point names."""
    s = s.strip()
    for i in range(1, len(s)):
        if s[:i] in names and s[i:] in names:
            return s[:i], s[i:]
    if len(s) == 2:
        return s[0], s[1]
    return None


def find_offset(point_img: Path, base_img: Path, pts: dict[str, tuple[float, float]],
                tol: float = 4.0) -> tuple[float, float, int, int] | None:
    """Translation mapping logic-form positions onto the red dots drawn in img_diagram_point.png.

    Returns (dx, dy, n_inlier_points, n_dots), or None if no dots are found.
    """
    with Image.open(point_img) as a, Image.open(base_img) as b:
        p = np.asarray(a.convert("RGB")).astype(np.int16)
        q = np.asarray(b.convert("RGB")).astype(np.int16)
    if p.shape != q.shape:
        return None
    red = (p[..., 0] == 255) & (p[..., 1] == 0) & (p[..., 2] == 0) & (np.abs(p - q).sum(-1) > 0)
    dots = _blob_centroids(red)
    if not dots:
        return None
    dots = np.array(dots)
    P = np.array(list(pts.values()))
    best = (-1, np.inf, 0.0, 0.0)
    for c in dots:
        for x in P:
            t = c - x
            d = np.min(np.linalg.norm((P + t)[:, None, :] - dots[None, :, :], axis=-1), axis=1)
            inl = int((d <= tol).sum())
            res = float(d[d <= tol].mean()) if inl else np.inf
            if (inl, -res) > (best[0], -best[1]):
                best = (inl, res, t[0], t[1])
    inl, _, tx, ty = best
    t = np.array([tx, ty])
    d = np.linalg.norm((P + t)[:, None, :] - dots[None, :, :], axis=-1)
    j = d.argmin(1)
    m = d.min(1) <= tol
    if m.any():
        t = (dots[j[m]] - P[m]).mean(0)
    return float(t[0]), float(t[1]), int(inl), len(dots)


def _blob_centroids(mask: np.ndarray) -> list[tuple[float, float]]:
    """Centroids (x, y) of 8-connected blobs in a small boolean mask."""
    ys, xs = np.nonzero(mask)
    todo = set(zip(ys.tolist(), xs.tolist()))
    out = []
    while todo:
        stack = [todo.pop()]
        blob = []
        while stack:
            y, x = stack.pop()
            blob.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (y + dy, x + dx)
                    if n in todo:
                        todo.remove(n)
                        stack.append(n)
        if len(blob) >= 5:
            arr = np.array(blob, dtype=float)
            out.append((arr[:, 1].mean(), arr[:, 0].mean()))
    return out
