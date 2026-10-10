"""Draw the queried primitives onto the image in the highlight colour (plan §5.3).

Colour: green (0, 200, 0) by default. The plan proposed red, but red / orange / cyan already occur
in 13-15% of real diagrams (right-angle marks, ticks, labels) and saturated green in none of 600
sampled images, so a green highlight cannot be confused with drawn marks.

Segments: a `width_px` core line in `color` over a semi-transparent halo (`halo_alpha`,
`halo_width_px`). Circles: the circle outline the same way. Points: a filled marker of radius
`point_radius_px` (with halo). For a collinear query [P, A, B] the line AB is highlighted as a
segment and P as a point. Everything else is left unchanged; `draw_overlay` also returns the mask
of pixels it touched (unit-tested).

The dataset draws the overlay after resizing the image to the model's long side, so the highlight
has the same thickness in every model input whatever the source resolution; coordinates are scaled
with `scale`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def flatten(im: Image.Image) -> Image.Image:
    """RGB on white (Geometry3K images are RGBA with a transparent background)."""
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def _primitives(d, refs: list, scale: float) -> list[tuple]:
    """(kind, geometry) in pixel coordinates of the scaled image."""
    P = lambda n: (d.points[n][0] * scale, d.points[n][1] * scale)  # noqa: E731
    out = []
    if refs and refs[0][0] == "pt" and len(refs) == 3 and all(r[0] == "pt" for r in refs):
        out.append(("seg", (P(refs[1][1]), P(refs[2][1]))))                  # collinear: line AB + point P
        out.append(("pt", P(refs[0][1])))
        return out
    for ref in refs:
        if ref[0] == "seg":
            out.append(("seg", (P(ref[1]), P(ref[2]))))
        elif ref[0] == "pt":
            out.append(("pt", P(ref[1])))
        elif ref[0] == "circle":
            c = d.circles[ref[1]]
            out.append(("circle", ((c.center_xy[0] * scale, c.center_xy[1] * scale), c.radius * scale)))
        elif ref[0] == "angle":
            a, v, b = (P(n) for n in ref[1:])
            out += [("seg", (v, a)), ("seg", (v, b))]
    return out


def _draw(draw: ImageDraw.ImageDraw, prims: list[tuple], width: float, colour: tuple, radius: float) -> None:
    w = max(1, int(round(width)))
    for kind, g in prims:
        if kind == "seg":
            draw.line([g[0], g[1]], fill=colour, width=w)
            for p in g:                                         # round caps
                draw.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=colour)
        elif kind == "circle":
            (cx, cy), r = g
            draw.ellipse([cx - r - w / 2, cy - r - w / 2, cx + r + w / 2, cy + r + w / 2], outline=colour, width=w)
        elif kind == "pt":
            rr = radius + (width - 3) / 2
            draw.ellipse([g[0] - rr, g[1] - rr, g[0] + rr, g[1] + rr], fill=colour)


def draw_overlay(image: Image.Image, d, refs: list, ocfg: dict, scale: float = 1.0) -> tuple[Image.Image, np.ndarray]:
    """Return (overlaid RGB image, boolean mask of the pixels the overlay touched)."""
    base = flatten(image)
    colour = tuple(int(c) for c in ocfg["color"])
    prims = _primitives(d, refs, scale)
    halo = Image.new("RGBA", base.size, (0, 0, 0, 0))
    _draw(ImageDraw.Draw(halo), prims, ocfg["halo_width_px"], (*colour, int(255 * ocfg["halo_alpha"])),
          ocfg["point_radius_px"] + (ocfg["halo_width_px"] - ocfg["width_px"]) / 2)
    core = Image.new("RGBA", base.size, (0, 0, 0, 0))
    _draw(ImageDraw.Draw(core), prims, ocfg["width_px"], (*colour, 255), ocfg["point_radius_px"])
    out = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), halo), core).convert("RGB")
    mask = (np.asarray(halo)[..., 3] > 0) | (np.asarray(core)[..., 3] > 0)
    return out, mask
