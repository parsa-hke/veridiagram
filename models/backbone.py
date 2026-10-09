"""DINOv3 ViT-L/16 (fallback DINOv2-L/14) loading and feature extraction (plan §6.1).

Exposes patch tokens, CLS token, and pooled feature (CLS ++ mean patch token).
"""

from __future__ import annotations


def load_backbone(cfg: dict):
    raise NotImplementedError("Phase 6.1")
