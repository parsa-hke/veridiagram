"""Relation-preserving augmentations, applied after the overlay (plan §5.5).

Only similarity transforms and photometric changes, which keep every relation of the vocabulary
(parallel, perpendicular, equal length, on circle, tangent, collinear) true or false as before:
  rotation by U(-rotation_deg, rotation_deg) with the canvas expanded (nothing is cut off)
  horizontal / vertical flips (p 0.5 each)
  isotropic scale s ~ U(scale): the drawing is first cropped to its ink bounding box (+ margin), then
      resized so its long side is s x the model's long side, capped at the long side (so s > 1
      zooms in only as far as the ink allows, and nothing is ever cut)
  line-thickness jitter (p line_thickness_prob): a 3x3 min filter, i.e. dark strokes and the
      highlight thicken (morphological dilation of the ink)
  contrast ~ U(contrast), Gaussian pixel noise (p noise_prob, sd ~ U(gaussian_noise_std)),
  JPEG round trip (p jpeg_prob, quality ~ U(jpeg_quality))
No shear or perspective: they break parallelism and equal lengths. Because the overlay is drawn
first, the highlight transforms with the diagram.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

WHITE = (255, 255, 255)


def ink_bbox(im: Image.Image, threshold: int = 200, margin: int = 8) -> tuple[int, int, int, int]:
    """Bounding box of dark pixels (< threshold); 200 keeps scan noise and pale fills out of it."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.nonzero(a < threshold)
    if len(xs) == 0:
        return 0, 0, im.width, im.height
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(im.width, xs.max() + 1 + margin), min(im.height, ys.max() + 1 + margin))


class Augment:
    def __init__(self, acfg: dict, long_side: int):
        self.c = acfg
        self.L = long_side

    def __call__(self, im: Image.Image, rng: np.random.Generator) -> Image.Image:
        c = self.c
        im = im.convert("RGB")
        if c.get("line_thickness_jitter") and rng.random() < c.get("line_thickness_prob", 0.3):
            im = im.filter(ImageFilter.MinFilter(3))
        if c.get("rotation_deg"):
            ang = float(rng.uniform(-c["rotation_deg"], c["rotation_deg"]))
            im = im.rotate(ang, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=WHITE)
        if c.get("hflip") and rng.random() < 0.5:
            im = ImageOps.mirror(im)
        if c.get("vflip") and rng.random() < 0.5:
            im = ImageOps.flip(im)
        if c.get("scale"):
            im = im.crop(ink_bbox(im))
            s = float(rng.uniform(*c["scale"]))
            target = min(self.L, max(16, int(round(self.L * s))))
            k = target / max(im.size)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.BICUBIC)
        if c.get("contrast"):
            im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(*c["contrast"])))
        if c.get("gaussian_noise_std") and rng.random() < c.get("noise_prob", 0.5):
            sd = float(rng.uniform(*c["gaussian_noise_std"]))
            a = np.asarray(im).astype(np.float32) / 255.0
            im = Image.fromarray((np.clip(a + rng.normal(0, sd, a.shape), 0, 1) * 255).astype(np.uint8))
        if c.get("jpeg_quality") and rng.random() < c.get("jpeg_prob", 0.5):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=int(rng.integers(c["jpeg_quality"][0], c["jpeg_quality"][1] + 1)))
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        return im


def build_augment(cfg: dict) -> Augment | None:
    return Augment(cfg["augment"], cfg["input"]["long_side"]) if cfg["augment"].get("enabled") else None
