"""Metrics (plan §8.1): per-relation precision, recall and F1; macro-F1; expected calibration error
(ECE, 15 bins); val-tuned operating thresholds; bootstrap 95% confidence intervals over diagrams
(1,000 resamples).

Inputs are (N, R) arrays over queries x relations: probabilities (or binary predictions), labels, and
the applicability mask; entries with mask 0 are ignored everywhere.

macro-F1 averages the F1 of relations that have at least one positive label among the evaluated
entries (a relation with no positives has undefined recall and is reported, not averaged).

ECE is the calibration of the predicted probability that a relation holds: applicable entries are
binned by probability into `n_bins` equal-width bins on [0, 1], and ECE is the frequency-weighted mean
over bins of |mean label - mean probability|. It is computed over all applicable entries pooled (every
relation is a yes/no question with the same meaning) and per relation.

Bootstrap: the diagram is the unit of resampling (queries of one diagram share its drawing and
coordinates, so they are not independent). Per-diagram confusion counts are resampled with
replacement, so one resample costs a matrix product. The statistic can be the macro-F1 of one run, or
the mean over several runs (seeds) that predicted the same queries: resampling the diagrams jointly
gives a CI of the seed-mean.
"""

from __future__ import annotations

import warnings

import numpy as np

CI_LEVEL = 0.95
DEFAULT_GRID = np.array([0.01, 0.03, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.97, 0.99])


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


# ----------------------------------------------------------------------------------------------- ECE

def ece(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the positive-class probability over the applicable entries."""
    p, y, m = np.asarray(probs, float), np.asarray(labels, float), np.asarray(mask).astype(bool)
    p, y = p[m], y[m]
    if p.size == 0:
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        sel = b == k
        if sel.any():
            total += sel.sum() / p.size * abs(y[sel].mean() - p[sel].mean())
    return float(total)


def ece_per_relation(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, vocab: list[str],
                     n_bins: int = 15) -> dict[str, float]:
    p, y, m = np.asarray(probs), np.asarray(labels), np.asarray(mask)
    return {r: ece(p[:, j:j + 1], y[:, j:j + 1], m[:, j:j + 1], n_bins) for j, r in enumerate(vocab)}


# ------------------------------------------------------------------------------------- thresholds

def tune_thresholds(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, vocab: list[str],
                    grid: np.ndarray = DEFAULT_GRID) -> np.ndarray:
    """Per-relation operating threshold that maximises F1 on the given (validation) entries.
    Ties go to the threshold nearest 0.5; a relation with no positives keeps 0.5."""
    p, y, m = np.asarray(probs, float), np.asarray(labels).astype(bool), np.asarray(mask).astype(bool)
    out = np.full(len(vocab), 0.5)
    for j in range(len(vocab)):
        pj, yj = p[m[:, j], j], y[m[:, j], j]
        if not yj.any():
            continue
        best, best_t = -1.0, 0.5
        for t in sorted(grid, key=lambda t: abs(t - 0.5)):
            pred = pj >= t
            tp, fp, fn = (pred & yj).sum(), (pred & ~yj).sum(), (~pred & yj).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best + 1e-12:
                best, best_t = f1, float(t)
        out[j] = best_t
    return out


# -------------------------------------------------------------------------------------- bootstrap

def diagram_counts(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray, diagram_ids: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per-diagram confusion counts: (unique diagram ids, counts of shape (D, R, 3) = tp, fp, fn)."""
    pred, labels, mask = (np.asarray(a).astype(bool) for a in (pred, labels, mask))
    ids, inv = np.unique(np.asarray(diagram_ids), return_inverse=True)
    tp, fp, fn = (pred & labels & mask), (pred & ~labels & mask), (~pred & labels & mask)
    counts = np.zeros((len(ids), pred.shape[1], 3))
    for k, a in enumerate((tp, fp, fn)):
        np.add.at(counts[:, :, k], inv, a.astype(float))
    return ids, counts


def f1_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F1 per relation from summed counts (..., R, 3); relations without positives get F1 = nan.
    Returns (f1, support)."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    support = tp + fn
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return np.where(support > 0, f1, np.nan), support


def macro_from_counts(counts: np.ndarray) -> np.ndarray:
    f1, _ = f1_from_counts(counts)
    with np.errstate(invalid="ignore"):
        return np.nanmean(f1, axis=-1)


def bootstrap_macro_f1(sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]], diagram_ids: np.ndarray,
                       vocab: list[str], n_resamples: int = 1000, seed: int = 0, level: float = CI_LEVEL) -> dict:
    """CI over diagrams of the macro-F1 averaged over `sets` = [(pred, labels, mask), ...] (one per seed;
    all on the same queries, in the same order as `diagram_ids`). Returns the point estimate (mean over
    sets of the full-data macro-F1), the CI bounds, the per-set points, and per-relation CIs of the
    set-mean F1."""
    ids = None
    counts = []
    for pred, labels, mask in sets:
        ids_s, c = diagram_counts(pred, labels, mask, diagram_ids)
        assert ids is None or (ids == ids_s).all()
        ids, counts = ids_s, counts + [c]
    counts = np.stack(counts)                                              # (S, D, R, 3)
    S, D = counts.shape[:2]
    full = counts.sum(1)                                                   # (S, R, 3)
    points = macro_from_counts(full)                                       # (S,)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(D, np.full(D, 1.0 / D), size=n_resamples).astype(float)   # (n, D)
    res = np.einsum("nd,sdrk->nsrk", w, counts)                            # (n, S, R, 3)
    macro = np.nanmean(macro_from_counts(res), axis=1)                     # mean over sets -> (n,)
    f1, _ = f1_from_counts(res)                                            # (n, S, R)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rel = np.nanmean(f1, axis=1)                                       # (n, R)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    f1_full, _ = f1_from_counts(full)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f1_mean = np.nanmean(f1_full, axis=0)
    per = {}
    for j, r in enumerate(vocab):
        col = rel[:, j]
        col = col[~np.isnan(col)]
        per[r] = {"f1": None if np.isnan(f1_mean[j]) else float(f1_mean[j]),
                  "lo": float(np.percentile(col, lo)) if col.size else None,
                  "hi": float(np.percentile(col, hi)) if col.size else None}
    macro = macro[~np.isnan(macro)]
    return {"macro_f1": float(np.nanmean(points)), "lo": float(np.percentile(macro, lo)), "hi": float(np.percentile(macro, hi)),
            "points": [float(x) for x in points], "n_diagrams": int(D), "n_resamples": int(n_resamples),
            "per_relation": per}
