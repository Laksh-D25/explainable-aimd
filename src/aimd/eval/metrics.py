"""Detection and calibration metrics.

Beyond the usual detection numbers, this module carries the calibration metrics
the architecture named but never implemented. The literature review makes the
case plainly: the worst deployment failure is a confident, wrong accusation
against a human artist, so a detector that is 99% accurate but badly calibrated
is not deployable. That makes ECE and the false-positive rate at the operating
threshold first-class results, not diagnostics.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    return y_true, y_score


def confusion(y_true, y_score, threshold: float = 0.5) -> dict[str, int]:
    y_true, y_score = _as_arrays(y_true, y_score)
    pred = y_score >= threshold
    return {
        "tp": int(np.sum(pred & (y_true == 1))),
        "fp": int(np.sum(pred & (y_true == 0))),
        "tn": int(np.sum(~pred & (y_true == 0))),
        "fn": int(np.sum(~pred & (y_true == 1))),
    }


def f1_score(y_true, y_score, threshold: float = 0.5) -> float:
    c = confusion(y_true, y_score, threshold)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return 0.0 if denom == 0 else 2 * c["tp"] / denom


def accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    c = confusion(y_true, y_score, threshold)
    total = sum(c.values())
    return 0.0 if total == 0 else (c["tp"] + c["tn"]) / total


def false_positive_rate(y_true, y_score, threshold: float = 0.5) -> float:
    """Rate of real songs wrongly flagged -- the metric that matters for trust."""
    c = confusion(y_true, y_score, threshold)
    denom = c["fp"] + c["tn"]
    return 0.0 if denom == 0 else c["fp"] / denom


def roc_auc(y_true, y_score) -> float:
    """AUROC via the rank-sum identity (ties averaged)."""
    y_true, y_score = _as_arrays(y_true, y_score)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ranks within tied score groups, or ties bias the estimate.
    sorted_scores = y_score[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def equal_error_rate(y_true, y_score) -> tuple[float, float]:
    """EER and the threshold at which it occurs.

    Reported because the speech anti-spoofing literature this project builds on
    (Zhang et al., Guo et al.) reports EER, so it is what makes those numbers
    comparable.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(y_score)
    best = (float("inf"), 0.5, 1.0)
    for t in thresholds:
        c = confusion(y_true, y_score, t)
        fpr = c["fp"] / max(c["fp"] + c["tn"], 1)
        fnr = c["fn"] / max(c["fn"] + c["tp"], 1)
        gap = abs(fpr - fnr)
        if gap < best[0]:
            best = (gap, float(t), (fpr + fnr) / 2)
    return best[2], best[1]


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE: mean |confidence - accuracy| over equal-width confidence bins.

    `y_prob` must be a probability, not a logit -- temperature scaling is what
    makes this number meaningful, so run it on calibrated outputs.
    """
    y_true, y_prob = _as_arrays(y_true, y_prob)
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("expected probabilities in [0, 1]; apply a sigmoid first")
    # Confidence is distance from the decision boundary, and the prediction it
    # supports; both classes contribute.
    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    correct = (y_prob >= 0.5).astype(int) == y_true

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0.5 else (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


def summary(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """The standard row of a results table."""
    eer, eer_threshold = equal_error_rate(y_true, y_prob)
    return {
        "f1": f1_score(y_true, y_prob, threshold),
        "accuracy": accuracy(y_true, y_prob, threshold),
        "auroc": roc_auc(y_true, y_prob),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "fpr": false_positive_rate(y_true, y_prob, threshold),
        "ece": expected_calibration_error(y_true, y_prob),
    }
