"""data/overlay.py and data/augment.py (plan §5.3, §5.5)."""

import numpy as np
from PIL import Image, ImageDraw

from data.augment import Augment, ink_bbox
from data.overlay import draw_overlay
from data.schema import Circle, Diagram, circle_ref, pt, seg
from experiments.run import REPO_ROOT, load_config

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")
GREEN = tuple(CFG["overlay"]["color"])


def _diagram():
    return Diagram(id="synthetic:t", source="synthetic", split="train", image_path="", width=200, height=160,
                   points={"A": (20.0, 30.0), "B": (180.0, 30.0), "C": (100.0, 120.0), "O": (100.0, 80.0)},
                   circles=[Circle("O", (100.0, 80.0), 35.0, [])])


def _image():
    im = Image.new("RGB", (200, 160), "white")
    dr = ImageDraw.Draw(im)
    dr.line([(20, 30), (180, 30)], fill="black", width=2)
    dr.line([(20, 30), (100, 120)], fill=(30, 90, 200), width=2)
    rng = np.random.default_rng(0)
    return Image.fromarray(np.clip(np.asarray(im) + rng.integers(-3, 4, (160, 200, 3)), 0, 255).astype(np.uint8))


def test_overlay_only_changes_highlighted_pixels():
    base = _image()
    for refs in ([seg("A", "B"), seg("A", "C")], [pt("C"), circle_ref(0)], [seg("A", "B"), circle_ref(0)],
                 [pt("C"), pt("A"), pt("B")]):
        out, mask = draw_overlay(base, _diagram(), refs, CFG["overlay"])
        a, b = np.asarray(base), np.asarray(out)
        assert mask.any() and out.size == base.size
        assert np.array_equal(a[~mask], b[~mask]), refs                         # untouched outside the overlay
        core = (np.abs(b.astype(int) - np.array(GREEN)).sum(-1) < 10)
        assert core.sum() > 20 and mask[core].all()


def test_overlay_scales_coordinates():
    small = _image().resize((100, 80))
    _, mask = draw_overlay(small, _diagram(), [pt("C")], CFG["overlay"], scale=0.5)
    ys, xs = np.nonzero(mask)
    assert abs(xs.mean() - 50) < 2 and abs(ys.mean() - 60) < 2


def test_augment_keeps_lines_straight_and_size():
    im = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(im).line([(30, 40), (270, 160)], fill=GREEN, width=3)
    aug = Augment(CFG["augment"], 448)
    for k in range(6):
        out = aug(im, np.random.default_rng(k))
        assert max(out.size) <= 448 and out.mode == "RGB"
        a = np.asarray(out).astype(int)
        ys, xs = np.nonzero((a[..., 1] > 120) & (a[..., 0] < 120) & (a[..., 2] < 120))
        assert len(xs) > 30
        pts = np.stack([xs, ys], 1).astype(float)
        pts -= pts.mean(0)
        s = np.linalg.svd(pts, compute_uv=False)
        assert s[1] / s[0] < 0.08                                               # still a straight line


def test_ink_bbox_ignores_light_noise():
    a = np.full((100, 100, 3), 235, np.uint8)
    a[40:60, 30:35] = 0
    x0, y0, x1, y1 = ink_bbox(Image.fromarray(a), margin=0)
    assert (x0, y0, x1, y1) == (30, 40, 35, 60)
