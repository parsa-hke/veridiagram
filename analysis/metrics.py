"""Per-relation P/R/F1 and macro-F1 over applicable entries (plan §8.1; ECE and bootstrap CIs are
added in Phase 8). Inputs are (N, R) arrays: probabilities or binary predictions, labels, and the
applicability mask; entries with mask 0 are ignored.

macro-F1 averages the F1 of relations that have at least one positive label among the evaluated
entries (a relation with no positives has undefined recall and is reported, not averaged).
"""

from __future__ import annotations

import numpy as np


def per_relation(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray, vocab: list[str]) -> dict[str, dict]:
    pred, labels, mask = (np.asarray(a).astype(bool) for a in (pred, labels, mask))
    out = {}
    for j, r in enumerate(vocab):
        m = mask[:, j]
        p, y = pred[m, j], labels[m, j]
        tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[r] = {"precision": prec, "recall": rec, "f1": f1, "support": int(y.sum()), "n": int(m.sum()),
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def macro_f1(per: dict[str, dict]) -> float:
    vals = [v["f1"] for v in per.values() if v["support"] > 0]
    return float(np.mean(vals)) if vals else 0.0


def evaluate(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, vocab: list[str],
             threshold: float | np.ndarray = 0.5) -> dict:
    pred = np.asarray(probs) >= np.asarray(threshold)
    per = per_relation(pred, labels, mask, vocab)
    return {"macro_f1": macro_f1(per), "per_relation": per}
