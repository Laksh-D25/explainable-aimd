"""Report figures.

Charts are checked for the things that silently go wrong: a figure that renders
but encodes the wrong numbers, or writes no accessible table alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aimd.eval.report import (
    faithfulness_curves,
    layer_profile,
    reliability_diagram,
    results_markdown,
    robustness_chart,
)
from aimd.xai.faithfulness import FaithfulnessCurve


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.5, 1.0, size=2000)
    labels = (rng.random(2000) < probs).astype(int)
    return labels, np.clip(probs**3, 0, 1), probs  # over-confident, then corrected


def test_reliability_diagram_writes_all_three_formats(calibration_data, tmp_path):
    labels, before, after = calibration_data
    out = reliability_diagram(labels, before, after, tmp_path / "calib")
    assert out.exists()
    for ext in (".pdf", ".png", ".csv"):
        assert out.with_suffix(ext).exists(), f"missing {ext}"


def test_reliability_csv_is_the_accessible_view(calibration_data, tmp_path):
    """Every figure ships its numbers -- that table is the accessible view and
    the appendix material."""
    labels, before, after = calibration_data
    out = reliability_diagram(labels, before, after, tmp_path / "calib")
    data = pd.read_csv(out.with_suffix(".csv"))
    assert set(data["series"]) == {"Uncalibrated", "Temperature-scaled"}
    assert {"confidence", "accuracy", "n"} <= set(data.columns)
    assert (data["accuracy"].between(0, 1)).all()


def test_reliability_handles_a_single_series(calibration_data, tmp_path):
    labels, before, _ = calibration_data
    out = reliability_diagram(labels, before, None, tmp_path / "one")
    assert set(pd.read_csv(out.with_suffix(".csv"))["series"]) == {"Uncalibrated"}


def test_robustness_chart_drops_clean_and_sorts(tmp_path):
    df = pd.DataFrame(
        {"f1": [1.0, 0.7, 0.9, 0.5], "f1_drop": [0.0, 0.3, 0.1, 0.5],
         "family": ["-", "codec", "noise", "filter"]},
        index=pd.Index(["clean", "mp3_32", "noise_low_snr", "lowpass_4k"], name="condition"),
    )
    out = robustness_chart(df, tmp_path / "rob")
    data = pd.read_csv(out.with_suffix(".csv"))
    assert "clean" not in set(data["condition"]), "clean is the reference, not a condition"
    assert data["f1_drop"].is_monotonic_increasing, "bars must be sorted"


def test_layer_profile_normalises_shapley_to_a_comparable_share(tmp_path):
    """The two measures share one axis, so they must be put on a common scale --
    a second y-axis would manufacture agreement from the scaling alone."""
    attention = np.full(13, 1 / 13)
    shapley = np.linspace(-2, 5, 13)
    out = layer_profile(attention, shapley, tmp_path / "layers")
    data = pd.read_csv(out.with_suffix(".csv"))
    assert len(data) == 13
    assert data["shapley_share"].sum() == pytest.approx(1.0)
    assert (data["shapley_share"] >= 0).all(), "share is of absolute contribution"
    assert data["shapley_l3"].tolist() == pytest.approx(shapley.tolist())


def test_layer_profile_survives_all_zero_shapley(tmp_path):
    out = layer_profile(np.full(13, 1 / 13), np.zeros(13), tmp_path / "zero")
    assert np.isfinite(pd.read_csv(out.with_suffix(".csv"))["shapley_share"]).all()


def test_faithfulness_curve_plots_both_series(tmp_path):
    curve = FaithfulnessCurve(
        np.linspace(0, 1, 5),
        np.array([5.0, 2.0, 0.5, 0.0, -0.5]),
        np.array([5.0, 4.5, 4.0, 3.0, 2.0]),
    )
    out = faithfulness_curves(curve, tmp_path / "faith")
    data = pd.read_csv(out.with_suffix(".csv"))
    assert {"fraction", "explanation", "random"} == set(data.columns)
    assert len(data) == 5


def test_results_markdown_states_the_baseline_comparison():
    """The base paper's 0.629 must appear, so the table cannot be read without it."""
    protocols = pd.DataFrame(
        {"n_songs": [100, 80], "f1": [0.95, 0.78], "accuracy": [0.9, 0.8],
         "auroc": [0.98, 0.85], "eer": [0.05, 0.2], "eer_threshold": [0.5, 0.5],
         "fpr": [0.02, 0.1], "ece": [0.01, 0.05]},
        index=pd.Index(["in_distribution", "cross_generator"], name="protocol"),
    )
    text = results_markdown(protocols)
    assert "0.629" in text
    assert "+0.151" in text, "the delta against the base paper must be stated"
    assert "| in_distribution |" in text
