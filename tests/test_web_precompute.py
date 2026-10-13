"""web/precompute.py: the game's near-threshold pick and the palette PNGs (no model, no data)."""

import math

from PIL import Image, ImageDraw

from web import precompute

VOCAB = ["parallel", "perpendicular", "equal_length", "on_circle", "tangent", "collinear"]


def _row(deg, geo_par, kind="seg_pair", perp=0):
    m = math.sin(math.radians(deg))
    return {"kind": kind, "mask": [1, 1, 1, 0, 0, 0], "margins": [m, 0.9, 0.2, None, None, None],
            "geo": [geo_par, perp, 0, 0, 0, 0]}


def test_game_pick_is_balanced_near_the_tolerance():
    tol = 3.7
    rows = [_row(d, int(d <= tol)) for d in (0.5, 2.5, 3.0, 3.6, 3.8, 4.5, 5.1, 8.0)]
    rows += [_row(3.0, 1, kind="pt_circle"), _row(3.0, 1, perp=1)]          # wrong kind; perpendicular too
    idx = precompute.pick_game(rows, VOCAB, tol, 4)
    angles = [precompute.angle_deg(rows[i]["margins"][0]) for i in idx]
    assert all(precompute.GAME_BAND[0] * tol <= a <= precompute.GAME_BAND[1] * tol for a in angles)
    assert sum(a <= tol for a in angles) == 2 and sum(a > tol for a in angles) == 2
    assert 8 not in idx and 9 not in idx and 0 not in idx and 7 not in idx


def test_palette_png_keeps_the_green(tmp_path):
    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    d.line([(20, 20), (400, 30)], fill=(0, 200, 0), width=3)
    d.line([(20, 200), (400, 210)], fill="black", width=2)
    precompute.save_png(im, tmp_path / "x.png")
    out = Image.open(tmp_path / "x.png").convert("RGB")
    px = [out.getpixel((200, y)) for y in range(20, 30)]
    assert any(abs(r) < 30 and abs(g - 200) < 30 and b < 30 for r, g, b in px)
