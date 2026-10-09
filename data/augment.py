"""Relation-preserving augmentations, applied after overlay (plan §5.5).

Rotation, flips, isotropic scale, line-thickness jitter, Gaussian/JPEG noise,
slight contrast. No shear or perspective: they break parallelism and equality.
"""

from __future__ import annotations


def build_augment(cfg: dict):
    raise NotImplementedError("Phase 5.5")
