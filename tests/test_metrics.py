"""Metrics are validated against sklearn where an equivalent exists, and against
hand-computed values where one does not. A silently-wrong metric would be
reported as a research result, so these are checked rather than trusted."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import roc_auc_score as sk_auc

from aimd.eval.metrics import (
    accuracy,
    equal_error_rate,
    expected_calibration_error,
    f1_score,
    false_positive_rate,
    roc_auc,
    summary,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    # Signal plus noise, squashed to a probability.
    p = 1 / (1 + np.exp(-(y * 1.5 + rng.normal(0, 1.0, size=500))))
    return y, p


def test_roc_auc_matches_sklearn(scores):
    y, p = scores
    assert roc_auc(y, p) == pytest.approx(sk_auc(y, p), rel=1e-9)


def test_roc_auc_handles_ties(scores):
    """Constant scores must give 0.5, not a rank artefact."""
    y, _ = scores
    tied = np.full(len(y), 0.7)
    assert roc_auc(y, tied) == pytest.approx(0.5)
    assert roc_auc(y, tied) == pytest.approx(sk_auc(y, tied))


def test_f1_matches_sklearn(scores):
    y, p = scores
    assert f1_score(y, p) == pytest.approx(sk_f1(y, (p >= 0.5).astype(int)))


def test_auc_is_nan_for_a_single_class():
    assert np.isnan(roc_auc(np.ones(10), np.random.rand(10)))


def test_perfect_and_inverted_separation():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert f1_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_false_positive_rate_counts_only_real_songs():
    """FPR is the trust metric: real songs wrongly flagged as AI-generated."""
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1])  # 2 of 4 real songs flagged
    assert false_positive_rate(y, p) == pytest.approx(0.5)


def test_equal_error_rate_on_perfect_separation():
    eer, _ = equal_error_rate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert eer == pytest.approx(0.0)


def test_equal_error_rate_on_random_scores_is_near_half(scores):
    y, _ = scores
    rng = np.random.default_rng(1)
    eer, _ = equal_error_rate(y, rng.random(len(y)))
    assert 0.35 < eer < 0.65


def test_ece_is_zero_for_a_perfectly_calibrated_detector():
    """A detector that says 0.8 and is right 80% of the time has ECE ~ 0."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, size=20_000)
    y = (rng.random(20_000) < p).astype(int)  # correct exactly p of the time
    assert expected_calibration_error(y, p) < 0.02


def test_ece_is_large_for_a_confidently_wrong_detector():
    """The failure the project exists to prevent: certain, and wrong."""
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.99)  # certain every real song is fake
    assert expected_calibration_error(y, p) > 0.9


def test_ece_rejects_logits():
    with pytest.raises(ValueError, match="probabilities in"):
        expected_calibration_error(np.array([0, 1]), np.array([-2.0, 3.0]))


def test_summary_reports_the_full_row(scores):
    y, p = scores
    row = summary(y, p)
    assert set(row) == {"f1", "accuracy", "auroc", "eer", "eer_threshold", "fpr", "ece"}
    assert all(np.isfinite(v) for v in row.values())
    assert row["accuracy"] == pytest.approx(accuracy(y, p))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        f1_score(np.array([0, 1]), np.array([0.5]))
