"""Real diagrams for the website's constructions (WEBSITE_PLAN.md §1.2, §2; milestone 3).

The page's figures are drawn from actual project data, never invented: the frontispiece shows a real
parallel query from the synthetic extrapolation set with the demo model's own probability for it,
Book II shows four query kinds on the same diagram, and Proposition I shows a renderer-drawn diagram
whose true-but-unmarked relations get Byrne's colours. Synthetic diagrams only (our own, no licence
question, plan §7). Called by web.export, which writes figures.json.
"""

from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path

import numpy as np

from data.schema import Diagram, read_jsonl

KINDS = ("seg_pair", "pt_circle", "seg_circle", "pt_line")
SEG_RELATIONS = ("parallel", "perpendicular", "equal_length")


def geometry(d: Diagram) -> dict:
    """What the browser needs to redraw a diagram: coordinates only, rounded to 0.1 px."""
    r1 = lambda v: round(float(v), 1)  # noqa: E731
    return {"id": d.id, "width": d.width, "height": d.height,
            "points": {k: [r1(x), r1(y)] for k, (x, y) in d.points.items()},
            "segments": [list(s) for s in d.segments],
            "circles": [{"center": c.center, "cx": r1(c.center_xy[0]), "cy": r1(c.center_xy[1]), "r": r1(c.radius)}
                        for c in d.circles if c.center_xy is not None and c.radius]}


def _len(d: Diagram, ref) -> float:
    (x1, y1), (x2, y2) = d.points[ref[1]], d.points[ref[2]]
    return math.hypot(x2 - x1, y2 - y1)


def _line_gap(d: Diagram, r1, r2) -> float:
    """Distance from the midpoint of r2 to the carrier line of r1."""
    (x1, y1), (x2, y2) = d.points[r1[1]], d.points[r1[2]]
    mx = (d.points[r2[1]][0] + d.points[r2[2]][0]) / 2
    my = (d.points[r2[1]][1] + d.points[r2[2]][1]) / 2
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    return abs((x2 - x1) * (y1 - my) - (x1 - mx) * (y2 - y1)) / L


def _verdicts(row: dict, probs: np.ndarray, vocab: list[str]) -> dict:
    return {r: {"prob": round(float(probs[i]), 4), "truth": bool(row["geo"][i]),
                "margin": None if row["margins"][i] is None else round(float(row["margins"][i]), 6)}
            for i, r in enumerate(vocab) if row["mask"][i]}


def pick_hero(rows: list[dict], diagrams: dict[str, Diagram], probs: np.ndarray, vocab: list[str]) -> dict | None:
    """A parallel pair the model gets right, drawn well: long segments, well apart, in a diagram with a
    circle (the compass) and all four query kinds among its evaluation rows (for Book II), preferring
    diagrams where more kinds have a true query."""
    ip = vocab.index("parallel")
    kinds_of: dict[str, set] = {}
    true_kinds: dict[str, set] = {}
    for r in rows:
        kinds_of.setdefault(r["diagram_id"], set()).add(r["kind"])
        if sum(r["geo"]):
            true_kinds.setdefault(r["diagram_id"], set()).add(r["kind"])
    best, best_score = None, (-1, -1.0)
    for i, r in enumerate(rows):
        if r["kind"] != "seg_pair" or not r["geo"][ip] or sum(r["geo"]) != 1 or probs[i, ip] < 0.9:
            continue
        d = diagrams[r["diagram_id"]]
        if not any(c.center_xy is not None and c.radius for c in d.circles) or not set(KINDS) <= kinds_of[d.id]:
            continue
        size = math.hypot(d.width, d.height)
        a, b = r["refs"]
        score = (len(true_kinds.get(d.id, ())),                      # Book II shows a true query of each kind
                 min(_len(d, a), _len(d, b)) / size * min(_line_gap(d, a, b) / size, 0.25))
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    r, d = rows[best], diagrams[rows[best]["diagram_id"]]
    v = _verdicts(r, probs[best], vocab)
    m = v["parallel"]["margin"]
    return {"diagram": geometry(d), "query": {"qid": r["qid"], "kind": r["kind"], "refs": r["refs"]},
            "verdicts": v, "angle_off_deg": None if m is None else round(math.degrees(math.asin(min(m, 1.0))), 2)}


def pick_queries(rows: list[dict], probs: np.ndarray, vocab: list[str], diagram_id: str) -> list[dict]:
    """One query of each kind from a diagram's evaluation rows, a true one when there is one."""
    out = []
    for kind in KINDS:
        cand = [(i, r) for i, r in enumerate(rows)
                if r["diagram_id"] == diagram_id and r["kind"] == kind]
        if not cand:
            continue
        i, r = max(cand, key=lambda ir: (sum(ir[1]["geo"]) > 0, -ir[0]))
        out.append({"qid": r["qid"], "kind": kind, "refs": r["refs"], "verdicts": _verdicts(r, probs[i], vocab)})
    return out


def _off(rtype: str, margin: float | None) -> dict:
    """The verifier margin in words a reader knows: degrees off for angles, percent for lengths."""
    if margin is None:
        return {}
    if rtype in ("parallel", "perpendicular"):
        return {"off_deg": round(math.degrees(math.asin(min(margin, 1.0))), 2)}
    if rtype == "equal_length":
        return {"off_pct": round(100 * margin, 1)}
    return {}


def pick_unmarked(diagrams, max_segments: int = 7, want: int = 3) -> dict | None:
    """Proposition I: a diagram whose marks (asserted) miss relations the ruler confirms (geometric).
    `want` unmarked segment relations on pairwise disjoint segments, so each gets one Byrne colour;
    the first diagram with `want` of them, else the first with `want - 1`."""
    fallback = None
    for d in diagrams:
        if d.circles or not 4 <= len(d.segments) <= max_segments:
            continue
        marked = [r for r in d.asserted if r.type in SEG_RELATIONS]
        if not marked:
            continue
        mkeys = {r.key() for r in marked}
        used: set[str] = set()
        chosen = []
        for r in d.geometric:
            segs = {json.dumps(a) for a in r.args}
            if r.type not in SEG_RELATIONS or r.key() in mkeys or segs & used:
                continue
            chosen.append(r)
            used |= segs
            if len(chosen) == want:
                break
        rel = lambda r: {"type": r.type, "args": r.args, "margin": r.margin, **_off(r.type, r.margin)}  # noqa: E731
        pick = {"diagram": geometry(d), "marked": [rel(r) for r in marked], "unmarked": [rel(r) for r in chosen],
                "geometric_total": sum(r.type in SEG_RELATIONS for r in d.geometric)}
        if len(chosen) == want:
            return pick
        if fallback is None and len(chosen) == want - 1:
            fallback = pick
    return fallback


def figures(cfg: dict, runs: Path, processed: Path, demo_run: str, vocab: list[str]) -> dict:
    """figures.json: hero + Book II queries (demo run's synth_extrap predictions) and Proposition I."""
    out: dict = {"source": "synthetic diagrams drawn by the project's renderer"}
    pred = runs / demo_run / "predictions" / "synth_extrap.npz"
    try:
        from analysis.phase7 import eval_rows
        rows, diagrams = eval_rows(cfg, "synth_extrap")
        z = np.load(pred)
        if len(z["qid"]) != len(rows) or (z["qid"] != np.array([r["qid"] for r in rows])).any():
            raise ValueError(f"{pred} does not match the synth_extrap evaluation rows")
        hero = pick_hero(rows, diagrams, z["probs"], vocab)
        out["hero"] = hero
        out["queries"] = None if hero is None else pick_queries(
            rows, z["probs"], vocab, hero["diagram"]["id"])
        out["model_run"] = demo_run
    except (FileNotFoundError, ValueError, KeyError) as e:
        out["hero"], out["queries"], out["hero_reason"] = None, None, f"{type(e).__name__}: {str(e)[:160]}"
    synth = processed / "synth_train.jsonl"
    out["unmarked"] = pick_unmarked(islice(read_jsonl(synth), 5000)) if synth.exists() else None
    return out
