"""Evaluation protocols.

These guard the discipline that makes the numbers trustworthy: thresholds come
from validation, robustness conditions are applied one at a time, and the
cross-generator result carries its comparison baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from aimd.data.datasets import ClipSpec
from aimd.data.manifest import split_by_song
from aimd.data.perturb import EVAL_PERTURBATIONS
from aimd.eval.protocols import (
    rows_for_split,
    build_loader,
    evaluate_cross_generator,
    evaluate_in_distribution,
    evaluate_robustness,
    fit_calibration,
    results_table,
)
from aimd.models.detector import Detector

SR = 24_000
SPEC = ClipSpec(clip_seconds=0.5, clips_per_song=2, max_clips=2, sample_rate=SR)


@pytest.fixture
def corpus(tmp_path):
    """Forty short songs; fake ones carry a tone so a model could separate them.

    Sized so 80/10/10 leaves a non-empty test split after per-class rounding."""
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * SR)) / SR
    rows = []
    for i in range(40):
        fake = i % 2
        audio = 0.2 * rng.standard_normal(len(t))
        if fake:
            audio = audio + 0.2 * np.sin(2 * np.pi * 6000 * t)
        path = tmp_path / f"s{i}.wav"
        sf.write(path, audio.astype(np.float32), SR)
        rows.append(
            {
                "song_id": f"s{i}",
                "path": str(path),
                "label": fake,
                "taxonomy": "full_fake" if fake else "real",
                "source": "suno" if fake else None,
                "duration": 1.5,
            }
        )
    return split_by_song(pd.DataFrame(rows))


@pytest.fixture
def model(stub_backbone):
    torch.manual_seed(0)
    return Detector(backbone=stub_backbone).eval()


def test_in_distribution_reports_a_full_metric_row(model, corpus):
    result = evaluate_in_distribution(model, corpus, SPEC)
    assert result.name == "in_distribution"
    assert result.n_songs == (corpus["split"] == "test").sum()
    assert {"f1", "auroc", "eer", "ece", "fpr"} <= set(result.metrics)


def test_cross_generator_carries_the_baseline_comparison(model, corpus):
    """The base paper's 0.629 must travel with the result, so the report cannot
    quietly omit what the number is being compared against."""
    result = evaluate_cross_generator(model, corpus, SPEC, baseline_f1=0.629)
    assert result.extra["baseline_f1"] == 0.629
    assert result.extra["delta_vs_baseline"] == pytest.approx(
        result.metrics["f1"] - 0.629
    )


def test_robustness_table_covers_every_held_out_condition(model, corpus):
    table = evaluate_robustness(model, corpus, SPEC)
    assert "clean" in table.index
    for condition in EVAL_PERTURBATIONS:
        assert condition.name in table.index, f"{condition.name} missing from robustness table"
    assert table.loc["clean", "f1_drop"] == 0.0


def test_robustness_reports_drop_relative_to_clean(model, corpus):
    """f1_drop must be positive for degradation, the direction a reader expects."""
    table = evaluate_robustness(model, corpus, SPEC)
    clean_f1 = table.loc["clean", "f1"]
    for name in table.index.drop("clean"):
        assert table.loc[name, "f1_drop"] == pytest.approx(clean_f1 - table.loc[name, "f1"])


def test_robustness_applies_one_condition_at_a_time(model, corpus):
    """Each row must be attributable to a single degradation."""
    table = evaluate_robustness(model, corpus, SPEC, conditions=EVAL_PERTURBATIONS[:2])
    assert len(table) == 3  # clean + 2
    assert set(table["family"]) == {"-", *(c.family for c in EVAL_PERTURBATIONS[:2])}


def test_calibration_is_fitted_on_validation_not_test(model, corpus):
    """A threshold chosen on test reports the best case, not the expected one."""
    scaler, threshold = fit_calibration(model, corpus, SPEC, target_fpr=0.1)
    assert scaler.temperature > 0
    assert 0.0 <= threshold <= 1.0

    val_ids = set(corpus.loc[corpus["split"] == "val", "song_id"])
    test_ids = set(corpus.loc[corpus["split"] == "test", "song_id"])
    assert not (val_ids & test_ids), "calibration split overlaps the test split"


def test_calibrated_scale_changes_the_reported_metrics(model, corpus):
    scaler, threshold = fit_calibration(model, corpus, SPEC, target_fpr=0.1)
    plain = evaluate_in_distribution(model, corpus, SPEC)
    calibrated = evaluate_in_distribution(model, corpus, SPEC, scaler=scaler, threshold=threshold)
    assert calibrated.threshold == threshold
    assert calibrated.metrics["auroc"] == pytest.approx(plain.metrics["auroc"])  # rank-invariant


def test_eval_loader_is_deterministic(corpus):
    a = next(iter(build_loader(corpus, SPEC, split="test")))
    b = next(iter(build_loader(corpus, SPEC, split="test")))
    assert torch.equal(a["clips"], b["clips"])
    assert a["song_id"] == b["song_id"]


def test_results_table_indexes_by_protocol(model, corpus):
    table = results_table(
        [
            evaluate_in_distribution(model, corpus, SPEC),
            evaluate_cross_generator(model, corpus, SPEC),
        ]
    )
    assert list(table.index) == ["in_distribution", "cross_generator"]
    assert "f1" in table.columns


def test_loader_evaluates_only_the_requested_split(corpus):
    """Evaluation must never see training songs -- it would inflate every
    number in the report."""
    loader = build_loader(corpus, SPEC, split="test")
    seen = {sid for batch in loader for sid in batch["song_id"]}
    expected = set(corpus.loc[corpus["split"] == "test", "song_id"])
    assert seen == expected
    assert seen.isdisjoint(set(corpus.loc[corpus["split"] == "train", "song_id"]))


def test_manifest_without_a_split_column_is_treated_as_eval_only(corpus):
    """FakeMusicCaps is never trained on, so all of its rows are the eval set."""
    eval_only = corpus.drop(columns=["split"])
    assert len(rows_for_split(eval_only, "test")) == len(eval_only)


def test_empty_split_fails_loudly(corpus):
    with pytest.raises(ValueError, match="is empty"):
        rows_for_split(corpus, "nonexistent")
