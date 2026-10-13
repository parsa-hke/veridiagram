"""Inference for the website (WEBSITE_PLAN.md §3, §4.1): no web framework here, only the model.

Parity rule: every pixel the model sees is produced by the repository's own code. A drawing from the
Construct canvas becomes a `user` Diagram rendered by the synthetic renderer (data.synthetic.render,
the style at the middle of the training ranges, no noise); an uploaded image is used as it is.
Then, per query, data.dataset.overlay_image (flatten, fit the long side to 448, draw the green
highlight) and data.dataset.normalise (pad to the patch multiple, normalise), exactly as
QueryDataset does. Which relations a query can have and the Ruler's verdicts come from
data.queries.label_rows (query enumeration, applicability mask, verifier.checks with the fixed
tolerances); "read the whole figure" samples queries as evaluation does (data.queries.cap_rows) and
checks the model's confident answers with the Part B closure (verifier.consistency via
experiments.selftrain.query_facts). Nothing is stored.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.dataset import normalise, overlay_image, pad_batch, pad_to_multiple
from data.queries import Carriers, cap_rows, label_rows, vocabulary
from data.schema import Circle, Diagram, seg
from data.synthetic.render import FONTS, render_array
from experiments.selftrain import query_facts
from models.common import autocast
from verifier.consistency import closure

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_QUERIES = 60
MAX_POINTS = 26
MAX_SEGMENTS = 40
MAX_CIRCLES = 6
MAX_SIDE = 4096
KINDS = {"seg_pair": ("seg", "seg"), "pt_circle": ("pt", "circle"), "seg_circle": ("seg", "circle"),
         "pt_line": ("pt", "pt", "pt")}


class InputError(ValueError):
    """A request the server cannot answer; the message is shown to the visitor."""


@dataclass
class Prepared:
    diagram: Diagram
    image: Image.Image
    basis: str                     # "construction" (exact coordinates) or "clicks" (approximate)


# ------------------------------------------------------------------ requests -> diagrams

NAME = re.compile(r"^[A-Za-z0-9_']{1,6}$")


def _name(n) -> str:
    if not isinstance(n, str) or not NAME.match(n):
        raise InputError(f"point names are 1-6 letters, digits, _ or ', got {n!r}")
    return n


def _points(raw: dict) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("no points")
    if len(raw) > MAX_POINTS:
        raise InputError(f"at most {MAX_POINTS} points")
    out = {}
    for k, v in raw.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) and math.isfinite(c) for c in v)):
            raise InputError(f"point {k!r} needs [x, y]")
        out[_name(k)] = (float(v[0]), float(v[1]))
    return out


def _segments(raw: list, pts: dict) -> list[tuple[str, str]]:
    if len(raw or []) > MAX_SEGMENTS:
        raise InputError(f"at most {MAX_SEGMENTS} segments")
    out = []
    for s in raw or []:
        if len(s) != 2 or s[0] not in pts or s[1] not in pts or s[0] == s[1]:
            raise InputError(f"segment {s!r} must join two different known points")
        t = tuple(seg(*s)[1:])
        if t not in out:
            out.append(t)
    return out


def _circles(raw: list, pts: dict) -> list[Circle]:
    if len(raw or []) > MAX_CIRCLES:
        raise InputError(f"at most {MAX_CIRCLES} circles")
    out = []
    for c in raw or []:
        center = c.get("center")
        if center is not None and center not in pts:
            raise InputError(f"circle centre {center!r} is not a known point")
        cxy = pts[center] if center is not None else c.get("center_xy")
        if cxy is None or len(cxy) != 2:
            raise InputError("a circle needs a centre point or center_xy")
        r = c.get("radius")
        if r is None and c.get("through") in pts:
            t = pts[c["through"]]
            r = math.hypot(t[0] - cxy[0], t[1] - cxy[1])
        if not (isinstance(r, (int, float)) and r > 0 and math.isfinite(r)):
            raise InputError("a circle needs a positive radius or a point it passes through")
        out.append(Circle(center, (float(cxy[0]), float(cxy[1])), float(r), []))
    return out


def decode_image(b64: str) -> Image.Image:
    if not isinstance(b64, str):
        raise InputError("image must be base64")
    if "," in b64[:64]:                                           # data:image/png;base64,....
        b64 = b64.split(",", 1)[1]
    if len(b64) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise InputError("image larger than 5 MB")
    try:
        raw = base64.b64decode(b64, validate=True)
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:  # noqa: BLE001
        raise InputError(f"not a readable PNG or JPEG image ({type(e).__name__})") from e
    if max(im.size) > MAX_SIDE:
        raise InputError(f"image larger than {MAX_SIDE} px on a side")
    return im


def encode_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def clean_style(cfg: dict) -> dict:
    """The renderer's style at the middle of every training range, without noise or JPEG artefacts."""
    r = cfg["synthetic"]["render"]
    mid = lambda k: float(np.mean(r[k]))  # noqa: E731
    family, slant = FONTS[3]                                     # serif italic, as textbooks letter points
    return {"line_width": mid("line_width"), "font_size": mid("font_size"), "label_offset": mid("label_offset_px"),
            "font_family": family, "font_style": slant, "dots": True, "mark_size": mid("mark_size_px"),
            "scan_noise": 0.0, "jpeg_quality": 0}


def from_construction(c: dict, cfg: dict) -> Prepared:
    """A Construct-canvas drawing in any coordinates -> a user Diagram on the synthetic canvas,
    rendered by the project's renderer (aspect and every relation preserved by a similarity map)."""
    pts = _points(c.get("points"))
    segs = _segments(c.get("segments"), pts)
    circles = _circles(c.get("circles"), pts)
    S = cfg["synthetic"]["canvas_px"]
    M = cfg["synthetic"]["margin_px"]
    xs = [p[0] for p in pts.values()] + [ci.center_xy[0] + s * ci.radius for ci in circles for s in (-1, 1)]
    ys = [p[1] for p in pts.values()] + [ci.center_xy[1] + s * ci.radius for ci in circles for s in (-1, 1)]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    k = (S - 2 * M) / span
    ox = S / 2 - k * (min(xs) + max(xs)) / 2
    oy = S / 2 - k * (min(ys) + max(ys)) / 2
    f = lambda p: (ox + k * p[0], oy + k * p[1])  # noqa: E731
    d = Diagram(id="user:construction", source="user", split="all", image_path="", width=S, height=S,
                points={n: f(p) for n, p in pts.items()}, segments=segs,
                circles=[Circle(ci.center, f(ci.center_xy), ci.radius * k, []) for ci in circles],
                meta={"drawn": [list(s) for s in segs], "marks": []})
    return Prepared(d, Image.fromarray(render_array(d, clean_style(cfg))), "construction")


def from_upload(image_b64: str, prims: dict) -> Prepared:
    im = decode_image(image_b64)
    pts = _points(prims.get("points"))
    d = Diagram(id="user:upload", source="user", split="all", image_path="", width=im.width, height=im.height,
                points=pts, segments=_segments(prims.get("segments"), pts), circles=_circles(prims.get("circles"), pts))
    return Prepared(d, im, "clicks")


def pad_square(x: torch.Tensor, L: int, white: torch.Tensor) -> torch.Tensor:
    """A (3, H, W) normalised tensor, padded right/bottom with normalised white to (3, L, L)."""
    return pad_batch([white.view(3, 1, 1).repeat(1, L, L), x], white)[1]


# ------------------------------------------------------------------ the engine

def _key(kind: str, refs: list) -> str:
    """Order-free identity of a query: segment pairs and the two line points of pt_line are unordered."""
    refs = [list(seg(*r[1:])) if r[0] == "seg" else list(r) for r in refs]
    if kind == "seg_pair":
        refs = sorted(refs, key=json.dumps)
    elif kind == "pt_line":
        refs = [refs[0], *sorted(refs[1:], key=json.dumps)]
    return kind + json.dumps(refs, separators=(",", ":"))


def ruler_entry(rel: str, row: dict, j: int, cfg: dict) -> dict:
    """The Ruler's verdict in a reader's units: degrees off for angles, percent for lengths."""
    tol = cfg["tolerances"]
    m = row["margins"][j]
    out = {"holds": bool(row["geo"][j]), "margin": m}
    if rel in ("parallel", "perpendicular"):
        t = tol[f"{rel}_deg"]
        out.update(unit="deg", tolerance=t, measured=None if m is None else round(math.degrees(math.asin(min(m, 1.0))), 2))
    else:
        t = tol[{"equal_length": "equal_length_rel", "on_circle": "on_circle_rel", "tangent": "tangent_rel",
                 "collinear": "collinear_rel"}[rel]]
        out.update(unit="pct", tolerance=round(100 * t, 2), measured=None if m is None else round(100 * m, 2))
    return out


class Engine:
    """A loaded model plus the project's preprocessing. `predict` is serialised by a lock."""

    def __init__(self, cfg: dict, model: torch.nn.Module, device: torch.device, name: str):
        self.cfg = cfg
        self.model = model.eval()
        self.device = device
        self.name = name
        self.vocab = vocabulary(cfg)
        self.tier2 = tuple(cfg["relations"].get("tier2_included", []))
        self.L = cfg["input"]["long_side"]
        self.patch = cfg["input"]["pad_to_multiple"]
        self.mean = torch.tensor(cfg["input"]["mean"]).view(3, 1, 1)
        self.std = torch.tensor(cfg["input"]["std"]).view(3, 1, 1)
        self.white = ((torch.ones(3, 1, 1) - self.mean) / self.std).view(3)       # as QueryDataset.white
        self.batch = int(cfg.get("server", {}).get("batch_size", 8))
        self.pad = cfg.get("server", {}).get("pad", "square")             # square | patch (QueryDataset's own)
        self.lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, cfg: dict, checkpoint: str | Path, device: str = "auto", run: str = "") -> "Engine":
        """Build the architecture from the backbone's configuration (no pretrained download: the
        checkpoint holds every weight) and load best.pt."""
        from transformers import AutoConfig, AutoModel

        from experiments.run import get_device
        from models.backbone import Backbone
        from models.finetune import build_classifier

        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        name = sd.get("backbone") or cfg["model"]["backbone"]
        if name != cfg["model"]["backbone"]:
            raise ValueError(f"checkpoint was trained on {name}, config says {cfg['model']['backbone']}")
        hf = AutoModel.from_config(AutoConfig.from_pretrained(name))
        model = build_classifier(cfg, Backbone(hf, name), len(vocabulary(cfg)))
        model.load_state_dict(sd["model"])
        dev = get_device(device)
        short = name.split("/")[-1].replace("-large", "-l")
        return cls(cfg, model.float().to(dev), dev, f"{short} / {run}" if run else short)

    # -------------------------------------------------------------- model input and output

    def model_input(self, p: Prepared, refs: list) -> tuple[torch.Tensor, Image.Image]:
        """The exact tensor the model sees for one query, and that image as the visitor will see it.
        QueryDataset's tensor, then (pad "square", the default) white padding to long_side x long_side:
        the pooled feature averages the padding tokens too, so evaluation's answers depended on the
        largest image in each batch; a fixed square makes an answer depend on its own query only
        (python -m web.server.parity measures the three ways of asking)."""
        img = overlay_image(p.image, p.diagram, refs, self.L, self.cfg["overlay"], self.cfg["overlay"].get("enabled", True))
        x = normalise(img, self.patch, self.mean, self.std)
        if self.pad == "square":
            x = pad_square(x, self.L, self.white)
            view = Image.new("RGB", (self.L, self.L), (255, 255, 255))
            view.paste(img, (0, 0))
            return x, view
        return x, pad_to_multiple(img, self.patch)

    @torch.no_grad()
    def predict(self, xs: list[torch.Tensor]) -> np.ndarray:
        out = []
        with self.lock:
            for i in range(0, len(xs), self.batch):
                x = pad_batch(xs[i: i + self.batch], self.white).to(self.device)
                with autocast(self.device, self.cfg):
                    out.append(torch.sigmoid(self.model(x).float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.vocab)))

    # -------------------------------------------------------------- the API's two questions

    def rows(self, p: Prepared) -> list[dict]:
        return label_rows(p.diagram, self.vocab, self.cfg["tolerances"], self.tier2, "user")

    def _result(self, p: Prepared, row: dict, probs: np.ndarray, view: Image.Image | None) -> dict:
        rels = [r for j, r in enumerate(self.vocab) if row["mask"][j]]
        idx = {r: self.vocab.index(r) for r in rels}
        res = {"kind": row["kind"], "refs": row["refs"], "applicable": rels,
               "probs": {r: round(float(probs[idx[r]]), 4) for r in rels},
               "ruler": {"basis": p.basis, **{r: ruler_entry(r, row, idx[r], self.cfg) for r in rels}}}
        if view is not None:
            res["model_input"] = encode_png(view)
        return res

    def ask(self, p: Prepared, queries: list[dict], images: bool = True) -> list[dict]:
        if not queries:
            raise InputError("no queries")
        if len(queries) > MAX_QUERIES:
            raise InputError(f"at most {MAX_QUERIES} queries per request")
        by_key = {_key(r["kind"], r["refs"]): r for r in self.rows(p)}
        chosen = []
        for q in queries:
            kind, refs = q.get("kind"), q.get("refs")
            if kind not in KINDS or not isinstance(refs, list) or [r[0] for r in refs] != list(KINDS[kind]):
                raise InputError(f"unknown query {kind!r} {refs!r}")
            row = by_key.get(_key(kind, refs))
            if row is None:
                raise InputError(f"the model cannot be asked about {refs!r}: it is not a query of this figure "
                                 "(segments on one line, a circle's own centre, or unknown parts)")
            chosen.append(row)
        inputs = [self.model_input(p, r["refs"]) for r in chosen]
        probs = self.predict([x for x, _ in inputs])
        return [self._result(p, r, probs[i], inputs[i][1] if images else None) for i, r in enumerate(chosen)]

    def sample(self, p: Prepared, cap: int = MAX_QUERIES) -> tuple[list[dict], int]:
        """The figure's queries, sampled to `cap` as data.queries.build_index samples evaluation units
        (deterministic for a figure, so chunks of one read line up)."""
        rows = self.rows(p)
        rng = np.random.default_rng([self.cfg["data"].get("index_seed", 0), zlib.crc32(p.diagram.id.encode())])
        return cap_rows(rows, min(cap, MAX_QUERIES), rng, keep_positives=False), len(rows)

    def summary(self, p: Prepared, rows: list[dict], probs: np.ndarray) -> dict:
        """The model's relation set, the Ruler's, and the contradictions in the model's confident
        answers after deductive closure (the Part B check: τ and rule C2 from configs `selftrain`)."""
        st = self.cfg["selftrain"]
        mask = np.array([r["mask"] for r in rows], dtype=bool)
        labels = probs > 0.5
        use = mask & (np.maximum(probs, 1 - probs) > st["tau"])
        car = Carriers(p.diagram)
        pos, neg = [], []
        for i, r in enumerate(rows):
            a, b = query_facts(p.diagram, r, self.vocab, labels[i], use[i], car)
            pos += a
            neg += b
        contradictions, _ = closure(pos, neg if st.get("consistency_negatives") else ())
        model, ruler = [], []
        for i, r in enumerate(rows):
            for j, rel in enumerate(self.vocab):
                if r["mask"][j] and probs[i, j] > 0.5:
                    model.append({"relation": rel, "kind": r["kind"], "refs": r["refs"], "prob": round(float(probs[i, j]), 4)})
                if r["mask"][j] and r["geo"][j]:
                    ruler.append({"relation": rel, "kind": r["kind"], "refs": r["refs"]})
        return {"relations": model, "ruler_relations": ruler, "contradictions": contradictions, "closure_tau": st["tau"]}

    def read(self, p: Prepared, cap: int = MAX_QUERIES, offset: int = 0, limit: int | None = None) -> dict:
        """Every query of the figure (uniform sample of `cap`, as in evaluation) answered, with the
        summary. With `limit`, only queries [offset, offset + limit) and no summary: the page reads a
        figure in chunks to show progress, then asks /api/closure with all the answers."""
        rows, n_all = self.sample(p, cap)
        if not rows:
            raise InputError("this figure has no queries: draw at least two segments, or a circle and a point")
        part = rows[offset: offset + limit] if limit else rows
        probs = self.predict([self.model_input(p, r["refs"])[0] for r in part])
        out = {"results": [self._result(p, r, probs[i], None) for i, r in enumerate(part)], "offset": offset,
               "queries_total": n_all, "queries_asked": len(rows)}
        if limit is None:
            out.update(self.summary(p, rows, probs))
        return out

    def closure(self, p: Prepared, answers: list[dict]) -> dict:
        """The summary for answers the page has collected chunk by chunk ({kind, refs, probs})."""
        if not answers or len(answers) > MAX_QUERIES:
            raise InputError(f"send 1 to {MAX_QUERIES} answers")
        by_key = {_key(r["kind"], r["refs"]): r for r in self.rows(p)}
        rows, probs = [], np.zeros((len(answers), len(self.vocab)))
        for i, a in enumerate(answers):
            row = by_key.get(_key(a.get("kind"), a.get("refs") or []))
            pr = a.get("probs")
            if row is None or not isinstance(pr, dict):
                raise InputError(f"unknown answer {a.get('refs')!r}")
            for j, rel in enumerate(self.vocab):
                v = pr.get(rel, 0.0) if row["mask"][j] else 0.0
                if not isinstance(v, (int, float)) or not 0 <= v <= 1:
                    raise InputError("probabilities lie in [0, 1]")
                probs[i, j] = v
            rows.append(row)
        return self.summary(p, rows, probs)


def timed(fn, *a, **k) -> tuple[object, int]:
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, round(1000 * (time.perf_counter() - t0))
