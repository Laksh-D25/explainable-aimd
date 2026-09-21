"""Calibration is one of the four claimed contributions, so it gets tested like
a result rather than a utility."""

from __future__ import annotations

import numpy as np
import pytest

from aimd.calibrate.temperature import TemperatureScaler, threshold_for_target_fpr
from aimd.eval.metrics import expected_calibration_error, roc_auc


INFLATION = 4.0


def calibrated_logits(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels and logits that are calibrated *by construction*.

    The logit must be the true log-odds of its label, so the label is drawn
    from the probability the logit encodes. Deriving logits from labels the
    other way round (e.g. `2 * label - 1 + noise`) yields something whose true
    posterior is a rescaling of the logit, so the ideal temperature is not 1
    and a test asserting otherwise checks arithmetic rather than calibration.
    """
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.02, 0.98, size=n)
    labels = (rng.random(n) < prob).astype(int)
    return labels, np.log(prob / (1 - prob))


@pytest.fixture
def overconfident():
    """An overconfident detector: correct ranking, logits inflated by a known
    factor -- the typical failure mode of a network trained to convergence."""
    labels, honest = calibrated_logits()
    return labels, honest * INFLATION, honest


def test_scaling_reduces_calibration_error(overconfident):
    labels, inflated, _ = overconfident
    before = expected_calibration_error(labels, 1 / (1 + np.exp(-inflated)))
    after = expected_calibration_error(labels, TemperatureScaler().fit_transform(inflated, labels))
    assert after < before, f"ECE got worse: {before:.4f} -> {after:.4f}"
    assert after < 0.05


def test_recovers_the_inflation_factor(overconfident):
    """Scaling a calibrated logit by k must be undone by a temperature of k."""
    labels, inflated, _ = overconfident
    t = TemperatureScaler().fit(inflated, labels).temperature
    assert t == pytest.approx(INFLATION, rel=0.15), f"expected ~{INFLATION}, got {t:.2f}"


def test_ranking_is_unchanged(overconfident):
    """Temperature scaling is monotonic, so AUROC/F1/EER must be untouched --
    this is what makes it safe to apply after training."""
    labels, inflated, _ = overconfident
    before = roc_auc(labels, 1 / (1 + np.exp(-inflated)))
    after = roc_auc(labels, TemperatureScaler().fit_transform(inflated, labels))
    assert after == pytest.approx(before, rel=1e-9)


def test_temperature_stays_positive_on_adversarial_input():
    """log-T parameterisation must keep T > 0 even when labels are inverted."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=500)
    inverted = (logits < 0).astype(int)
    t = TemperatureScaler().fit(logits, inverted).temperature
    assert t > 0 and np.isfinite(t)


def test_already_calibrated_input_leaves_temperature_near_one():
    """Calibration must be close to a no-op on an already-honest detector."""
    labels, honest = calibrated_logits(seed=2)
    t = TemperatureScaler().fit(honest, labels).temperature
    assert t == pytest.approx(1.0, rel=0.12), f"expected ~1, got {t:.2f}"


def test_threshold_honours_a_false_positive_budget():
    """The operating point is chosen against an FPR budget, not left at 0.5:
    wrongly flagging a human artist is the asymmetric cost."""
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=5000)
    logits = labels * 2.0 - 1.0 + rng.normal(0, 1.0, size=5000)

    scaler = TemperatureScaler().fit(logits, labels)
    threshold = threshold_for_target_fpr(logits, labels, target_fpr=0.01, scaler=scaler)

    probs = scaler.transform(logits)
    realised_fpr = (probs[labels == 0] >= threshold).mean()
    assert realised_fpr <= 0.02, f"FPR budget 1% breached: {realised_fpr:.3f}"
    assert threshold > 0.5, "a 1% FPR budget should demand more than a coin flip"


def test_separable_validation_gets_a_margin_midpoint_threshold():
    """The failure this prevents: with perfect validation separation, the
    FPR-quantile threshold sits right against the real class. On unseen
    generators everything then crosses it -- FPR 1.00, and an F1 identical to a
    trivial all-positive baseline."""
    labels = np.array([0] * 50 + [1] * 50)
    logits = np.concatenate([np.full(50, -8.0), np.full(50, 8.0)])  # clean margin

    t = threshold_for_target_fpr(logits, labels, target_fpr=0.05)
    probs = 1 / (1 + np.exp(-logits))
    assert probs[:50].max() < t < probs[50:].min(), "threshold must sit inside the margin"
    assert t == pytest.approx((probs[:50].max() + probs[50:].min()) / 2)


def test_separable_threshold_survives_a_shift_that_breaks_the_quantile():
    """A modest shift toward 'fake' should not flip every real song."""
    labels = np.array([0] * 50 + [1] * 50)
    logits = np.concatenate([np.full(50, -8.0), np.full(50, 8.0)])
    t = threshold_for_target_fpr(logits, labels, target_fpr=0.05)

    # New data where real scores drifted upward but stay well below the margin.
    shifted_real = 1 / (1 + np.exp(-np.full(50, -4.0)))
    assert (shifted_real < t).all(), "midpoint threshold should still reject these"


def test_overlapping_validation_still_uses_the_fpr_budget():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=2000)
    logits = labels * 1.5 + rng.normal(0, 1.0, size=2000)  # overlapping
    t = threshold_for_target_fpr(logits, labels, target_fpr=0.05)
    probs = 1 / (1 + np.exp(-logits))
    assert (probs[labels == 0] >= t).mean() <= 0.08
