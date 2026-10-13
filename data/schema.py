"""Unified Diagram / Relation dataclasses and JSON (de)serialisation (plan §2.3).

Conventions
- Coordinates are float pixels in the image, origin top-left, x right, y down.
- Point names are strings and may be multi-character (``B'``, ``_p3`` for unlabelled points),
  so primitives are always referenced as lists of names, never concatenated strings.
- Primitive references inside ``Relation.args`` are tagged lists:
    ["pt", P]            a point
    ["seg", A, B]        segment / line through A and B (A < B, sorted)
    ["circle", i]        index into ``Diagram.circles``
    ["angle", A, V, B]   angle at vertex V between rays V->A and V->B (A < B, sorted)
- Relation args for each type (unordered pairs are sorted so each relation has one key):
    parallel, perpendicular, equal_length   [seg, seg]
    on_circle                               [pt, circle]
    tangent                                 [seg, circle]
    collinear                               [pt P, pt A, pt B]   P on line AB (A < B)
    midpoint                                [pt M, seg AB]
    equal_angle                             [angle, angle]
    perp_bisector                           [seg L, seg AB]      L is the perpendicular bisector of AB

Fields beyond the plan's sketch, all required by the raw data:
``split`` (published split), ``circles[i].center_xy`` (circle centres are not always a
named point), ``meta`` (source ids, provenance).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SOURCES = ("geometry3k", "pgdp5k", "pgps9k", "synthetic", "user")   # user: drawn or uploaded on the website
RELATION_TYPES = (
    "parallel", "perpendicular", "equal_length", "on_circle", "tangent",            # Tier 1
    "collinear", "midpoint", "equal_angle", "perp_bisector",                       # Tier 2
)

Ref = list  # tagged primitive reference, see module docstring


def seg(a: str, b: str) -> Ref:
    return ["seg", *sorted((a, b))]


def angle(a: str, v: str, b: str) -> Ref:
    a, b = sorted((a, b))
    return ["angle", a, v, b]


def pt(p: str) -> Ref:
    return ["pt", p]


def circle_ref(i: int) -> Ref:
    return ["circle", int(i)]


@dataclass
class Relation:
    type: str
    args: list[Ref]
    source: str = "asserted"           # 'asserted' | 'geometric'
    margin: float | None = None        # raw verifier quantity (Phase 3); None for asserted

    def __post_init__(self) -> None:
        if self.type not in RELATION_TYPES:
            raise ValueError(f"unknown relation type {self.type!r}")
        if self.source not in ("asserted", "geometric"):
            raise ValueError(f"unknown relation source {self.source!r}")
        self.args = canonical_args(self.type, [list(a) for a in self.args])

    def key(self) -> tuple:
        """Hashable identity (type + canonical args), independent of source and margin."""
        return (self.type, json.dumps(self.args))


def canonical_args(rtype: str, args: list[Ref]) -> list[Ref]:
    """Sort inside each ref and order unordered pairs, so equal relations compare equal."""
    out = []
    for a in args:
        if a[0] == "seg":
            a = seg(a[1], a[2])
        elif a[0] == "angle":
            a = angle(a[1], a[2], a[3])
        out.append(a)
    if rtype in ("parallel", "perpendicular", "equal_length", "equal_angle"):
        out = sorted(out, key=json.dumps)
    elif rtype == "collinear":
        out = [out[0], *sorted(out[1:], key=json.dumps)]
    return out


@dataclass
class Circle:
    center: str | None                 # name of the centre point, if one is annotated
    center_xy: tuple[float, float] | None
    radius: float | None               # pixels; None if it cannot be determined
    points_on: list[str] = field(default_factory=list)


@dataclass
class Diagram:
    id: str                            # globally unique: "<source>:<native id>"
    source: str
    split: str                         # train | val | test | all
    image_path: str                    # relative to the raw data root (cfg paths.raw)
    width: int
    height: int
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    segments: list[tuple[str, str]] = field(default_factory=list)
    lines: list[list[str]] = field(default_factory=list)       # collinear groups, ordered along the line
    circles: list[Circle] = field(default_factory=list)
    angles: list[tuple[str, str, str]] = field(default_factory=list)
    asserted: list[Relation] = field(default_factory=list)
    geometric: list[Relation] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d["points"] = {k: [float(v[0]), float(v[1])] for k, v in self.points.items()}
        d["segments"] = [list(s) for s in self.segments]
        d["angles"] = [list(a) for a in self.angles]
        for c in d["circles"]:
            c["center_xy"] = list(c["center_xy"]) if c["center_xy"] is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Diagram":
        d = dict(d)
        d["points"] = {k: (float(v[0]), float(v[1])) for k, v in d.get("points", {}).items()}
        d["segments"] = [tuple(s) for s in d.get("segments", [])]
        d["angles"] = [tuple(a) for a in d.get("angles", [])]
        d["circles"] = [Circle(**{**c, "center_xy": tuple(c["center_xy"]) if c.get("center_xy") else None})
                        for c in d.get("circles", [])]
        d["asserted"] = [Relation(**r) for r in d.get("asserted", [])]
        d["geometric"] = [Relation(**r) for r in d.get("geometric", [])]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Diagram":
        return cls.from_dict(json.loads(s))


def write_jsonl(diagrams: Iterable[Diagram], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for d in diagrams:
            f.write(d.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Diagram]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Diagram.from_json(line)
