"""Matplotlib textbook-style renderer for synthetic diagrams (plan §4.2).

Black lines on white, point labels A, B, C... placed away from the lines through each point, and
the marks recorded by the generator for injected relations: right-angle squares (perpendicular),
tick marks (equal_length; 1-3 ticks per class) and arrowheads (parallel; 1-2 per class), so the
synthetic data has an "asserted" label set drawn in the image like the real data.

Style knobs (config `synthetic.render`, sampled per diagram from its own seed): line width, font
size and family (serif / sans, italic or not), label offset, point dots, mark size, optional scan
noise (Gaussian pixel noise + slight blur) and optional JPEG artefacts (encode/decode round trip).
Files are PNG, except noisy images, which are stored as JPEG quality 95 to keep the pools small. The slight rotation is applied to the coordinates by the generator, so
the stored annotation matches the image.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from data.schema import Diagram  # noqa: E402

FONTS = [("DejaVu Sans", "normal"), ("DejaVu Sans", "italic"), ("DejaVu Serif", "normal"), ("DejaVu Serif", "italic")]


def sample_style(rng: np.random.Generator, rcfg: dict) -> dict:
    u = lambda k: float(rng.uniform(*rcfg[k]))  # noqa: E731
    family, slant = FONTS[int(rng.integers(len(FONTS)))]
    return {
        "line_width": u("line_width"), "font_size": u("font_size"), "label_offset": u("label_offset_px"),
        "font_family": family, "font_style": slant, "dots": bool(rng.random() < rcfg["dot_prob"]),
        "mark_size": u("mark_size_px"),
        "scan_noise": float(rng.uniform(*rcfg["scan_noise_std"])) if rcfg["scan_noise"] and rng.random() < rcfg["scan_noise_prob"] else 0.0,
        "jpeg_quality": int(rng.integers(rcfg["jpeg_quality"][0], rcfg["jpeg_quality"][1] + 1)) if rcfg["jpeg_artifacts"] and rng.random() < rcfg["jpeg_prob"] else 0,
    }


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.zeros(2)


def _label_direction(d: Diagram, name: str) -> np.ndarray:
    """Unit vector pointing away from everything attached to the point."""
    p = np.asarray(d.points[name])
    dirs = []
    for a, b in d.meta["drawn"]:
        if name in (a, b):
            dirs.append(_unit(np.asarray(d.points[b if name == a else a]) - p))
    for g in d.lines:
        if name in g:
            dirs += [_unit(np.asarray(d.points[g[0]]) - p), _unit(np.asarray(d.points[g[-1]]) - p)]
    for c in d.circles:
        if name in c.points_on and c.center_xy:
            dirs.append(_unit(np.asarray(c.center_xy) - p))
        if c.center == name:
            return np.array([0.7, -0.7])
    if not dirs:
        return np.array([0.7, -0.7])
    # pick the candidate direction with the largest angular gap to all attached directions
    best, score = None, -2.0
    for k in range(16):
        cand = np.array([math.cos(2 * math.pi * k / 16), math.sin(2 * math.pi * k / 16)])
        s = -max(float(np.dot(cand, v)) for v in dirs if np.any(v))  if any(np.any(v) for v in dirs) else 1.0
        if s > score:
            best, score = cand, s
    return best


def _draw_marks(ax, d: Diagram, st: dict) -> None:
    P = {n: np.asarray(v) for n, v in d.points.items()}
    k = st["mark_size"]
    lw = st["line_width"] * 0.9
    for m in d.meta.get("marks", []):
        if m["type"] == "right_angle":
            v = P[m["vertex"]]
            u1, u2 = _unit(P[m["rays"][0]] - v), _unit(P[m["rays"][1]] - v)
            pts = np.array([v + k * u1, v + k * (u1 + u2), v + k * u2])
            ax.plot(pts[:, 0], pts[:, 1], color="black", lw=lw)
        elif m["type"] in ("ticks", "arrows"):
            a, b = P[m["seg"][0]], P[m["seg"][1]]
            t, nrm = _unit(b - a), _unit(np.array([-(b - a)[1], (b - a)[0]]))
            mid = (a + b) / 2
            for i in range(m["n"]):
                c = mid + (i - (m["n"] - 1) / 2) * (0.45 * k) * t
                if m["type"] == "ticks":
                    q = np.array([c - 0.55 * k * nrm, c + 0.55 * k * nrm])
                else:
                    q = np.array([c - 0.6 * k * t + 0.45 * k * nrm, c, c - 0.6 * k * t - 0.45 * k * nrm])
                ax.plot(q[:, 0], q[:, 1], color="black", lw=lw)


def render_array(d: Diagram, st: dict) -> np.ndarray:
    size = d.width
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    for a, b in d.meta["drawn"]:
        (x1, y1), (x2, y2) = d.points[a], d.points[b]
        ax.plot([x1, x2], [y1, y2], color="black", lw=st["line_width"], solid_capstyle="round")
    for c in d.circles:
        ax.add_patch(plt.Circle(c.center_xy, c.radius, fill=False, color="black", lw=st["line_width"]))
    _draw_marks(ax, d, st)
    for name, (x, y) in d.points.items():
        if st["dots"]:
            ax.plot(x, y, "o", color="black", ms=st["line_width"] * 2.2)
        off = _label_direction(d, name) * (st["label_offset"] + st["font_size"] * 0.75)
        ax.text(x + off[0], y + off[1], name, fontsize=st["font_size"], family=st["font_family"],
                style=st["font_style"], ha="center", va="center", color="black")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def post_process(arr: np.ndarray, st: dict, rng: np.random.Generator) -> Image.Image:
    im = Image.fromarray(arr)
    if st["scan_noise"] > 0:
        im = im.filter(ImageFilter.GaussianBlur(0.6))
        a = np.asarray(im).astype(np.float32) / 255.0
        a = np.clip(a + rng.normal(0, st["scan_noise"], a.shape), 0, 1)
        im = Image.fromarray((a * 255).astype(np.uint8))
    if st["jpeg_quality"]:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=st["jpeg_quality"])
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return im


def sample_style_for(rcfg: dict, seed) -> dict:
    return sample_style(np.random.default_rng([*np.atleast_1d(seed).tolist(), 7]), rcfg)


def render(d: Diagram, out_path: str | Path, rcfg: dict, seed) -> dict:
    """Render d to out_path with a style sampled from (seed); returns the style used.
    Noisy images are stored as JPEG (quality 95): pixel noise does not compress in PNG (~390 KB
    vs ~45 KB), and the noise imitates a scan anyway. out_path's suffix must match image_suffix()."""
    rng = np.random.default_rng([*np.atleast_1d(seed).tolist(), 7])
    st = sample_style(rng, rcfg)
    im = post_process(render_array(d, st), st, rng)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".jpg":
        im.save(out_path, "JPEG", quality=95)
    else:
        im.save(out_path)
    return st


def image_suffix(st: dict) -> str:
    return ".jpg" if st["scan_noise"] > 0 else ".png"
