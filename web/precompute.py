"""Precompute the website's server-free demos (WEBSITE_PLAN.md §3.3, §3.4): the game and the gallery.

    python -m web.precompute [--device auto] [--game 200] [--gallery 12]

Needs the demo checkpoint (runs/<demo_run>/checkpoints/best.pt). Answers come from the website's own
engine (web.server.core: the same preprocessing and the fixed 448 x 448 padding as the live demo), so a
precomputed answer equals what the server would say. Synthetic diagrams only (our own renderer; the
demo model was trained on real diagrams only, so none of them was seen in training). Writes to
web/site/public/data/:
  game.json + game/<i>.png        near-threshold parallel questions from the extrapolation set: the
                                  model's view (green highlight), the Ruler's angle and tolerance, p(parallel)
  gallery.json + gallery/<i>.png  diagrams with every query answered (as "read the whole figure"):
                                  the image, coordinates for the page's highlight previews, results,
                                  the model's and the Ruler's relation sets, contradictions
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from data.schema import read_jsonl
from experiments.run import REPO_ROOT, load_config
from web.export import DEFAULT_OUT, SITE_YAML
from web.figures import geometry
from web.server import core

GAME_BAND = (0.65, 1.4)          # angles from 0.65 x to 1.4 x the parallel tolerance: near the line


def save_png(im: Image.Image, path: Path, colours: int = 48) -> None:
    """Line art compresses well as a palette PNG (the green highlight and its halo survive 48 colours)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").quantize(colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def angle_deg(margin: float) -> float:
    return math.degrees(math.asin(min(float(margin), 1.0)))


def pick_game(rows: list[dict], vocab: list[str], tol_deg: float, n: int) -> list[int]:
    """Indices of seg_pair rows whose angle lies in the band around the tolerance, half on each side
    of it, evenly spread over the angle (deterministic)."""
    ip, iq = vocab.index("parallel"), vocab.index("perpendicular")
    lo, hi = GAME_BAND[0] * tol_deg, GAME_BAND[1] * tol_deg
    cand = [(angle_deg(r["margins"][ip]), i) for i, r in enumerate(rows)
            if r["kind"] == "seg_pair" and r["mask"][ip] and r["margins"][ip] is not None and not r["geo"][iq]
            and lo <= angle_deg(r["margins"][ip]) <= hi]
    out = []
    for side in ([c for c in cand if c[0] <= tol_deg], [c for c in cand if c[0] > tol_deg]):
        side.sort()
        k = min(n // 2, len(side))
        out += [side[j][1] for j in np.linspace(0, len(side) - 1, k).astype(int)] if k else []
    return sorted(set(out))


def game(e: core.Engine, cfg: dict, out: Path, n: int) -> dict:
    from analysis.phase7 import eval_rows
    rows, diagrams = eval_rows(cfg, "synth_extrap")
    tol = cfg["tolerances"]["parallel_deg"]
    idx = pick_game(rows, e.vocab, tol, n)
    raw = REPO_ROOT / cfg["paths"]["raw"]
    items, xs = [], []
    for k, i in enumerate(idx):
        r, d = rows[i], diagrams[rows[i]["diagram_id"]]
        with Image.open(raw / d.image_path) as im:
            p = core.Prepared(d, im.copy(), "construction")
        x, view = e.model_input(p, r["refs"])
        xs.append(x)
        save_png(view, out / "game" / f"{k:03d}.png")
        ip = e.vocab.index("parallel")
        items.append({"image": f"game/{k:03d}.png", "diagram": d.id, "refs": r["refs"],
                      "angle_deg": round(angle_deg(r["margins"][ip]), 2), "parallel": bool(r["geo"][ip])})
    probs = e.predict(xs)
    ip = e.vocab.index("parallel")
    for it, pr in zip(items, probs):
        it["p_parallel"] = round(float(pr[ip]), 4)
    return {"source": "synthetic extrapolation set (12-20 points), never seen by the demo model",
            "tolerance_deg": tol, "band_deg": [round(GAME_BAND[0] * tol, 2), round(GAME_BAND[1] * tol, 2)],
            "model": e.name, "pad": e.pad, "items": items,
            "summary": {"n": len(items), "parallel": sum(i["parallel"] for i in items),
                        "model_correct": sum((i["p_parallel"] > 0.5) == i["parallel"] for i in items)}}


def pick_gallery(cfg: dict, n: int) -> list:
    """n synthetic diagrams, half from the training-size pool (4-10 points) and half from the
    extrapolation set (12-20 points), alternating with and without circles."""
    proc = REPO_ROOT / cfg["paths"]["processed"]
    out = []
    for pool, k in (("synth_train", n - n // 2), ("synth_extrap", n // 2)):
        want_circle, taken = True, 0
        for d in islice(read_jsonl(proc / f"{pool}.jsonl"), 4000):
            if taken == k:
                break
            if bool(d.circles) != want_circle or len(d.segments) < 4:
                continue
            out.append(d)
            taken += 1
            want_circle = not want_circle
    return out


def gallery(e: core.Engine, cfg: dict, out: Path, n: int) -> dict:
    raw = REPO_ROOT / cfg["paths"]["raw"]
    items = []
    for k, d in enumerate(pick_gallery(cfg, n)):
        with Image.open(raw / d.image_path) as im:
            p = core.Prepared(d, im.copy(), "construction")
        res = e.read(p)
        save_png(p.image, out / "gallery" / f"{k:02d}.png", colours=32)
        items.append({"image": f"gallery/{k:02d}.png", "diagram": geometry(d),
                      "points": len(d.points), "pool": d.id.split(":")[1].split("_")[0], **res})
    return {"source": "synthetic diagrams drawn by the project's renderer, never seen by the demo model",
            "model": e.name, "pad": e.pad, "items": items}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--game", type=int, default=200)
    ap.add_argument("--gallery", type=int, default=12)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    site = yaml.safe_load(SITE_YAML.read_text())
    cfg = load_config(REPO_ROOT / site["demo_config"])
    run = site["demo_run"]
    e = core.Engine.from_checkpoint(cfg, REPO_ROOT / "runs" / run / "checkpoints" / "best.pt", args.device, run)
    out = Path(args.out)
    g = game(e, cfg, out, args.game)
    (out / "game.json").write_text(json.dumps(g, indent=1) + "\n")
    print(f"game: {g['summary']} -> {out / 'game.json'}")
    gl = gallery(e, cfg, out, args.gallery)
    (out / "gallery.json").write_text(json.dumps(gl, indent=1) + "\n")
    print(f"gallery: {len(gl['items'])} diagrams, " + ", ".join(
        f"{it['diagram']['id']} {it['queries_asked']}q {len(it['contradictions'])}c" for it in gl["items"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
