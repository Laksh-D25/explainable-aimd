"""XAI level 2a and the faithfulness harness.

The mechanics are tested here; whether the trained model's explanations are
actually faithful is an M7 result, not something a unit test can assert.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aimd.models.detector import Detector
from aimd.xai.faithfulness import (
    FaithfulnessCurve,
    deletion_curve,
    is_faithful,
    comprehensiveness,
)
from aimd.xai.relevance import relevance_times, temporal_relevance, top_regions


@pytest.fixture
def model(stub_backbone):
    torch.manual_seed(0)
    return Detector(backbone=stub_backbone).eval()


@pytest.fixture
def song():
    torch.manual_seed(1)
    return torch.randn(1, 2, 24_000)  # one song, 2 clips of 1 s


def test_relevance_shape_is_clips_by_frames(model, song):
    rel = temporal_relevance(model, song)
    assert rel.shape[0] == 2  # clips
    assert rel.shape[1] == 75  # 1 s at the stub's 75 Hz stride


def test_relevance_is_non_negative_and_peak_normalised(model, song):
    rel = temporal_relevance(model, song)
    assert (rel >= 0).all()
    assert rel.max() == pytest.approx(1.0)


def test_relevance_requires_a_single_song(model):
    with pytest.raises(ValueError, match="one song at a time"):
        temporal_relevance(model, torch.randn(2, 2, 24_000))


def test_relevance_leaves_the_model_unchanged(model, song):
    """Explanation must not mutate training state or leave gradients behind."""
    model.train()
    temporal_relevance(model, song)
    assert model.training, "explaining flipped the model out of training mode"
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.trainable_parameters())


def test_relevance_times_maps_frames_to_seconds():
    t = relevance_times(75, frame_rate=75)
    assert t[0] == 0.0 and t[-1] == pytest.approx(74 / 75)


def test_top_regions_returns_sorted_intervals():
    rel = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    regions = top_regions(rel, k=2, frame_rate=10)
    assert regions == sorted(regions)
    assert regions[0] == (pytest.approx(0.1), pytest.approx(0.2))  # index 1
    assert regions[1] == (pytest.approx(0.3), pytest.approx(0.4))  # index 3


def test_deletion_curve_endpoints(model, song):
    """At fraction 0 nothing is masked, so the score must be the clean logit."""
    rel = temporal_relevance(model, song)
    curve = deletion_curve(model, song, rel, steps=4)
    clean = model.forward(song).binary_logit.item()
    assert curve.scores[0] == pytest.approx(clean, abs=1e-4)
    assert curve.random_scores[0] == pytest.approx(clean, abs=1e-4)
    assert len(curve.scores) == 5


def test_insertion_curve_starts_from_silence_and_ends_clean(model, song):
    rel = temporal_relevance(model, song)
    curve = deletion_curve(model, song, rel, steps=4, keep=True)
    clean = model.forward(song).binary_logit.item()
    silent = model.forward(torch.zeros_like(song)).binary_logit.item()
    assert curve.scores[0] == pytest.approx(silent, abs=1e-4)
    assert curve.scores[-1] == pytest.approx(clean, abs=1e-4)


def test_random_baseline_is_actually_a_different_ordering(model, song):
    rel = temporal_relevance(model, song)
    curve = deletion_curve(model, song, rel, steps=4)
    assert not np.allclose(curve.scores, curve.random_scores)


def test_faithfulness_gate_compares_against_chance():
    """A steep deletion curve means nothing until it beats random masking --
    removing any audio degrades a detector somewhat."""
    fractions = np.linspace(0, 1, 5)
    faithful = FaithfulnessCurve(fractions, np.array([5.0, 2.0, 0.5, 0.0, 0.0]),
                                 np.array([5.0, 4.5, 4.0, 3.0, 2.0]))
    unfaithful = FaithfulnessCurve(fractions, np.array([5.0, 4.5, 4.0, 3.0, 2.0]),
                                   np.array([5.0, 2.0, 0.5, 0.0, 0.0]))
    assert is_faithful(faithful)
    assert not is_faithful(unfaithful)
    assert faithful.gap < 0 < unfaithful.gap


def test_comprehensiveness_is_the_logit_drop():
    curve = FaithfulnessCurve(np.linspace(0, 1, 3), np.array([4.0, 2.0, 1.0]), np.zeros(3))
    assert comprehensiveness(curve) == pytest.approx(3.0)


def test_deletion_requires_a_single_song(model, song):
    rel = temporal_relevance(model, song)
    with pytest.raises(ValueError, match="one song"):
        deletion_curve(model, song.repeat(2, 1, 1), rel)
