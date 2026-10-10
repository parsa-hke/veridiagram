"""LayerNorm -> Linear -> per-relation sigmoid; multi-label BCE (plan §6.2).

The head outputs one logit per relation of the vocabulary; probabilities are sigmoid(logits).
`masked_bce` is BCE-with-logits with per-class positive weights, averaged over the applicable
entries only (the relation mask of each query: a point-circle query has no parallel label).
`pos_weights` derives the weights from the training-split census: for each relation,
(applicable negatives / positives), capped at `cap`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, in_dim: int, n_out: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_out)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(feats)))


def pos_weights(labels: np.ndarray, mask: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    labels, mask = np.asarray(labels, float), np.asarray(mask, float)
    pos = (labels * mask).sum(0)
    neg = ((1 - labels) * mask).sum(0)
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none",
                                              pos_weight=None if pos_weight is None else pos_weight.to(logits.device))
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
