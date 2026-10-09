"""Parser for the PGDP primitive annotation format, shared by PGDP5K and PGPS9K.

Raw record (see data/README.md for the full field list):
  geos.points   [{id: "p0", loc: [[x, y]]}]                        pixels
  geos.lines    [{id: "l0", loc: [[x1, y1], [x2, y2]]}]            longest drawn segment of a line
  geos.circles  [{id: "c0", loc: [[cx, cy], r, "1111"]}]           "1111": quadrant flags (arcs)
  symbols       [{id, sym_class, text_class?, text_content?, bbox: [x, y, w, h]}]
  relations.geo2geo  [p, l|c, "endpoint" | "online" | "oncircle" | "center"]
  relations.sym2geo  [s, [ids...]]   text->point label; perpendicular [p, l, l]; bar* [p, p, l];
                                     parallel* [l]; angle* [p, l, l]

Asserted relations come from the marks drawn in the diagram (symbols) and from point-on-line /
point-on-circle incidences. Text values (lengths, degrees) are not turned into relations.
"""

from __future__ import annotations

import math
from collections import defaultdict

from data.loaders.common import close_equivalences, dedupe, derive_tier2, order_along, segments_from_lines
from data.schema import Circle, Diagram, Relation, angle, circle_ref, pt, seg

BAR_CLASSES = ("bar", "double bar", "triple bar", "quad bar")
PARALLEL_CLASSES = ("parallel", "double parallel", "triple parallel")
ANGLE_CLASSES = ("angle", "double angle", "triple angle", "quad angle", "penta angle")


def parse_record(rec: dict, *, diagram_id: str, source: str, split: str, image_path: str) -> Diagram:
    flags: dict = {}
    geos = rec["geos"]
    pid_xy = {p["id"]: tuple(map(float, p["loc"][0])) for p in geos["points"]}
    sym = {s["id"]: s for s in rec["symbols"]}

    # ---- names: point-label text symbols attached to points
    names: dict[str, str] = {}
    used: set[str] = set()
    for sid, targets in rec["relations"]["sym2geo"]:
        s = sym.get(sid, {})
        if s.get("sym_class") == "text" and s.get("text_class") == "point":
            for t in targets:
                if t in pid_xy and t not in names:
                    names[t] = _unique(s["text_content"].strip(), used, flags)
    for pid in pid_xy:
        if pid not in names:
            names[pid] = _unique(f"_{pid}", used, flags)
            flags["unlabelled_points"] = flags.get("unlabelled_points", 0) + 1
    points = {names[p]: xy for p, xy in pid_xy.items()}

    # ---- incidences
    endpoints, online, oncircle, center = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for p, g, kind in rec["relations"]["geo2geo"]:
        if p not in names:
            continue
        if kind == "endpoint":
            endpoints[g].append(names[p])
        elif kind == "online":
            online[g].append(names[p])
        elif kind == "oncircle":
            oncircle[g].append(names[p])
        elif kind == "center":
            center[g] = names[p]

    # ---- lines: endpoints (synthesised if not annotated) + on-line points, ordered along the line
    line_group: dict[str, list[str]] = {}
    for ln in geos["lines"]:
        ends = list(dict.fromkeys(endpoints.get(ln["id"], [])))
        if len(ends) < 2:
            for k, xy in enumerate(ln["loc"][:2]):
                if len(ends) >= 2:
                    break
                near = _nearest(points, xy, ends)
                if near is None:
                    near = _unique(f"_{ln['id']}e{k}", used, flags)
                    points[near] = tuple(map(float, xy))
                    flags["synthesised_endpoints"] = flags.get("synthesised_endpoints", 0) + 1
                ends.append(near)
        group = list(dict.fromkeys(ends[:2] + online.get(ln["id"], [])))
        line_group[ln["id"]] = order_along(group, points)

    circles, circ_index = [], {}
    for c in geos["circles"]:
        (cx, cy), r = c["loc"][0], c["loc"][1]
        circ_index[c["id"]] = len(circles)
        circles.append(Circle(center=center.get(c["id"]), center_xy=(float(cx), float(cy)),
                              radius=float(r), points_on=sorted(set(oncircle.get(c["id"], [])))))

    lines = [g for g in line_group.values() if len(g) >= 2]
    rels: list[Relation] = []
    full = {lid: seg(g[0], g[-1]) for lid, g in line_group.items() if len(g) >= 2}

    for g in lines:
        for p in g[1:-1]:
            rels.append(Relation("collinear", [pt(p), pt(g[0]), pt(g[-1])]))
    for cid, i in circ_index.items():
        for p in circles[i].points_on:
            rels.append(Relation("on_circle", [pt(p), circle_ref(i)]))

    # ---- marks
    by_class = defaultdict(list)
    angles = []
    for sid, targets in rec["relations"]["sym2geo"]:
        s = sym.get(sid)
        if s is None or s["sym_class"] == "text":
            continue
        cls = s["sym_class"]
        tl = [t for t in targets if t in full]
        tp = [names[t] for t in targets if t in names]
        if cls == "perpendicular" and len(tl) >= 2:
            rels.append(Relation("perpendicular", [full[tl[0]], full[tl[1]]]))
        elif cls in PARALLEL_CLASSES:
            by_class[cls].extend(full[t] for t in tl)
        elif cls in BAR_CLASSES:
            by_class[cls].append(seg(tp[0], tp[1]) if len(tp) >= 2 else (full[tl[0]] if tl else None))
        elif cls in ANGLE_CLASSES and tp and len(tl) >= 2:
            a = _angle_from_mark(tp[0], line_group[tl[0]], line_group[tl[1]], points, s["bbox"])
            if a is None:
                flags["unresolved_angle_marks"] = flags.get("unresolved_angle_marks", 0) + 1
                continue
            angles.append(tuple(a[1:]))
            by_class[cls].append(a)

    for cls, refs in by_class.items():
        refs = [r for r in dict.fromkeys(map(tuple, (x for x in refs if x)))]
        rtype = ("parallel" if cls in PARALLEL_CLASSES else
                 "equal_length" if cls in BAR_CLASSES else "equal_angle")
        for a, b in zip(refs, refs[1:]):
            if a != b:
                rels.append(Relation(rtype, [list(a), list(b)]))

    rels = derive_tier2(close_equivalences(dedupe(rels)), lines)
    return Diagram(
        id=diagram_id, source=source, split=split, image_path=image_path,
        width=int(rec["width"]), height=int(rec["height"]),
        points=points, segments=segments_from_lines(lines), lines=lines, circles=circles,
        angles=sorted(set(angles)), asserted=rels, flags=flags,
        meta={"native_file": rec["file_name"], "pgdp_names": {k: v for k, v in names.items()}},
    )


def _unique(name: str, used: set[str], flags: dict) -> str:
    if not name:
        name = "_unnamed"
    out, k = name, 2
    while out in used:
        out, k = f"{name}#{k}", k + 1
    if out != name:
        flags["duplicate_point_names"] = flags.get("duplicate_point_names", 0) + 1
    used.add(out)
    return out


def _nearest(points: dict, xy, exclude: list[str], tol: float = 3.0) -> str | None:
    best, bd = None, tol
    for n, (x, y) in points.items():
        d = math.hypot(x - xy[0], y - xy[1])
        if n not in exclude and d <= bd:
            best, bd = n, d
    return best


def _angle_from_mark(v: str, g1: list[str], g2: list[str], points: dict, bbox) -> list | None:
    """Pick the rays from vertex v along lines g1, g2 whose bisector points at the mark's bbox centre."""
    if v not in g1 or v not in g2:
        return None
    vx, vy = points[v]
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    tx, ty = cx - vx, cy - vy

    def rays(g):
        i = g.index(v)
        return [g[j] for j in (i - 1, i + 1) if 0 <= j < len(g)]

    best, bscore = None, -2.0
    for a in rays(g1):
        for b in rays(g2):
            ua, ub = _unit(points[a], (vx, vy)), _unit(points[b], (vx, vy))
            bis = (ua[0] + ub[0], ua[1] + ub[1])
            nb, nt = math.hypot(*bis), math.hypot(tx, ty)
            if nb < 1e-9 or nt < 1e-9:
                continue
            score = (bis[0] * tx + bis[1] * ty) / (nb * nt)
            if score > bscore:
                best, bscore = angle(a, v, b), score
    return best


def _unit(p, o):
    dx, dy = p[0] - o[0], p[1] - o[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n
